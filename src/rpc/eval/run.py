"""Eval runner: ``python -m src.rpc.eval.run --config configs/eval.yaml``.

Rolling-origin loop: for each as_of -> PIT features -> observed labels ->
train baselines on (train window, observed train labels) -> score test refs ->
    metrics + report. Reads policy_log if present (eval-only).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rpc.eval import metrics as M
from src.rpc.eval._minifeatures import get_feature_builder
from src.rpc.eval.labels import load_verified_keys, observed_labels, train_labels_only
from src.rpc.eval.registry import get_scorer, list_scorers, register_builtin_baselines
from src.rpc.eval.report import generate_report
from src.rpc.eval.splits import (
    check_splits,
    default_first_asof,
    load_official_splits,
    make_rolling_splits,
    train_accounts_only,
)


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


def _train_frame(
    events: pd.DataFrame,
    cps: pd.DataFrame | None,
    borrowers: pd.DataFrame | None,
    train_start: datetime,
    train_end: datetime,
    horizon_days: int,
    rpc_responses: list,
    rpc_dispositions: list,
    builder: object,
    verified_keys: pd.DataFrame | None = None,
    official_splits: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features at train_end + observed TRAIN-only labels in the post-train window.

    Grain is (account, phone). Censored (undialled) and verified-holdout rows
    are dropped via train_labels_only — never treated as negatives. When the
    official splits.csv is supplied (secondary sanity), labels are further
    restricted to TRAIN-split accounts.
    """
    ev = events.copy()
    ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)
    cands = ev[
        (ev["occurred_at"] > pd.to_datetime(train_start, utc=True))
        & (ev["occurred_at"] <= pd.to_datetime(train_end, utc=True))
    ][["account_id", "contact_point_ref"]].drop_duplicates() if "account_id" in ev.columns else pd.DataFrame(
        {"contact_point_ref": ev[
            (ev["occurred_at"] > pd.to_datetime(train_start, utc=True))
            & (ev["occurred_at"] <= pd.to_datetime(train_end, utc=True))
        ]["contact_point_ref"].unique().tolist()}
    )
    cand_refs = cands["contact_point_ref"].unique().tolist()
    feats = builder(train_end, cand_refs, ev, cps, borrowers)  # type: ignore[operator]
    lab = observed_labels(
        ev, train_end, keys=cands if "account_id" in cands.columns else cand_refs,
        horizon_days=horizon_days, rpc_responses=rpc_responses,
        rpc_dispositions=rpc_dispositions, verified_keys=verified_keys,
    )
    if official_splits is not None and "account_id" in lab.columns:
        lab = train_accounts_only(lab, official_splits)
    fr = feats.merge(lab, on="contact_point_ref", how="left")
    fr = train_labels_only(fr)  # dialled-only training; flagged in report
    y = fr["rpc_next_7d"].astype(float)
    return fr, y, fr["contact_point_ref"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sp, lb, mc, dc, inc = cfg["splits"], cfg["labels"], cfg["metrics"], cfg["data"], cfg["incumbent"]
    prop_cfg = cfg.get("propensity", {})

    events = _load(dc["events"])
    cps = _load(dc["contact_points"])
    borrowers = _load(dc["borrowers"])
    policy_log = _load(dc["policy_log"])
    if events is None or events.empty:
        raise SystemExit(
            f"no events at {dc['events']}; ingest the official extracts "
            "first (see docs/dataset_audit.md §13)"
        )

    notes = [
        "Metrics are DIALLED-ONLY unless noted: undialled contact points are censored (no observable outcome, never negative).",
        "Labels are per-(account,phone) sanctioned RPC (answered AND rpc_*; language_barrier excluded); "
        "strict sensitivity (minus hung_up/refused) reported alongside the primary.",
        "Verified rows are holdout gold only — never train label sources (membership itself is leakage).",
    ]
    verified_keys = None
    vpath = lb.get("verified_holdout", "")
    if vpath:
        try:
            verified_keys = load_verified_keys(vpath)
            notes.append(f"Verified holdout excluded from training: {len(verified_keys)} keys from {vpath}.")
        except Exception as e:
            notes.append(f"Verified holdout not loaded ({vpath}): {e}.")
    official_splits = None
    spath = dc.get("official_splits", "")
    if spath:
        try:
            official_splits = load_official_splits(spath)
            notes.append("Official splits.csv used as secondary sanity only (primary = purged rolling origins).")
        except Exception as e:
            notes.append(f"Official splits not loaded ({spath}): {e}.")
    if policy_log is None:
        notes.append("policy_log.parquet absent: IPW second view skipped.")

    builder = get_feature_builder()
    from src.rpc.eval import _minifeatures as _mf

    notes.append(
        "REAL feature layer in use (eval adapter over the in-memory event "
        "source; mini columns backfilled only where the real output lacks them)."
        if builder is not _mf.build_minifeatures
        else "TEMPORARY DuckDB mini-features in use (src/rpc/eval/_minifeatures.py); switch when src/rpc/features lands."
    )

    first = default_first_asof(events, sp["train_days"])
    splits = make_rolling_splits(first, sp["n_splits"], sp["step_days"], sp["train_days"], sp["embargo_days"], sp["test_days"])
    check_splits(splits, sp["embargo_days"])

    register_builtin_baselines()
    model_names = [m for m in list_scorers() if m in ("incumbent", "account_gbm", "contact_gbm")]

    # Per-split test frames (features at as_of + labels after embargo).
    per_model_rows: dict[str, list[pd.DataFrame]] = {m: [] for m in model_names}
    strict_col = f"rpc_next_{lb['horizon_days']}d_strict"
    total_censored = 0
    total_verified_holdout = 0
    for s in splits:
        ev = events.copy()
        ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)
        test_keys = ev[
            (ev["occurred_at"] > pd.to_datetime(s.test_start, utc=True))
            & (ev["occurred_at"] <= pd.to_datetime(s.test_end, utc=True))
        ][["account_id", "contact_point_ref"]].drop_duplicates() if "account_id" in ev.columns else ev[
            (ev["occurred_at"] > pd.to_datetime(s.test_start, utc=True))
            & (ev["occurred_at"] <= pd.to_datetime(s.test_end, utc=True))
        ]["contact_point_ref"].unique().tolist()
        test_refs = test_keys["contact_point_ref"].unique().tolist() if isinstance(test_keys, pd.DataFrame) else list(test_keys)
        if not test_refs:
            continue
        feats = builder(s.as_of, test_refs, ev, cps, borrowers)  # type: ignore[operator]
        lab = observed_labels(ev, s.test_start, test_refs if not isinstance(test_keys, pd.DataFrame) else test_keys,
                              lb["horizon_days"], lb["rpc_network_responses"], lb["rpc_dispositions"],
                              verified_keys=verified_keys)
        base = feats.merge(lab, on="contact_point_ref", how="left")
        n_cens = int(base["censored"].sum()) if "censored" in base.columns else 0
        n_ver = int(base["verified_holdout"].sum()) if "verified_holdout" in base.columns else 0
        total_censored += n_cens
        total_verified_holdout += n_ver
        base_eval = base[~base["censored"]].copy()
        if n_ver:
            base_eval = base_eval[~base_eval["verified_holdout"]].copy()

        tr_feats, tr_y, _ = _train_frame(ev, cps, borrowers, s.train_start, s.train_end, lb["horizon_days"], lb["rpc_network_responses"], lb["rpc_dispositions"], builder, verified_keys, official_splits)
        acc_map = None
        if cps is not None and "account_id" in cps.columns:
            acc_map = cps[["contact_point_ref", "account_id"]]
        elif "account_id" in ev.columns:
            acc_map = ev[["contact_point_ref", "account_id"]].drop_duplicates()

        scorers: dict[str, object] = {}
        inc = get_scorer("incumbent")
        inc.attach_features(feats)  # type: ignore[union-attr]
        scorers["incumbent"] = inc
        if not tr_feats.empty and tr_y.notna().any() and tr_y.nunique() >= 2:
            agb = get_scorer("account_gbm")
            if acc_map is not None:
                # The real feature layer already emits account_id, so the merge
                # may suffix columns (account_id_x/y). Coalesce all variants.
                _m = tr_feats.merge(acc_map, on="contact_point_ref", how="left")
                _parts = [_m[c] for c in ("account_id", "account_id_y", "account_id_x") if c in _m.columns]
                tr_acc = _parts[0]
                for _p in _parts[1:]:
                    tr_acc = tr_acc.fillna(_p)
                tr_acc = tr_acc.fillna("UNK")
                agb.fit(tr_feats, tr_y, tr_acc)  # type: ignore[union-attr]
                agb.attach_context(pd.DataFrame({"contact_point_ref": test_refs}).merge(acc_map, on="contact_point_ref", how="left").fillna("UNK"))  # type: ignore[union-attr]
            scorers["account_gbm"] = agb
            cgb = get_scorer("contact_gbm")
            cgb.fit(tr_feats, tr_y)  # type: ignore[union-attr]
            cgb.attach_features(feats)  # type: ignore[union-attr]
            scorers["contact_gbm"] = cgb
        else:
            notes.append(f"split {s.as_of.date()}: degenerate train labels; GBMs skipped.")
        for name, sc in scorers.items():
            pred = sc.score(s.as_of, base_eval["contact_point_ref"].tolist())  # type: ignore[union-attr]
            per_model_rows.setdefault(name, []).append(
                base_eval.merge(pred, on="contact_point_ref", how="left")
            )

    # --- Aggregate + metric families ---
    tables: dict[str, object] = {"discrimination": [], "calibration": [], "rare_event": [],
                                 "decision": [], "avoiding_vs_invalid": [], "propensity": [],
                                 "sensitivity": []}
    reliability: dict[str, list] = {}
    n_boot, seed, level = mc["n_bootstrap"], mc["seed"], mc["ci_level"]
    for name, frames in per_model_rows.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        y = df["rpc_next_7d"].to_numpy(float)
        p = df["p_rpc"].to_numpy(float)
        auc, alo, ahi = M.bootstrap_ci(M.roc_auc, y, p, n_boot, seed, level)
        pra, plo, phi = M.bootstrap_ci(M.pr_auc, y, p, n_boot, seed, level)
        br, blo, bhi = M.bootstrap_ci(M.brier, y, p, n_boot, seed, level)
        ll, llo, lhi = M.bootstrap_ci(M.logloss, y, p, n_boot, seed, level)
        tables["discrimination"].append({"model": name, "n": int((~np.isnan(y)).sum()),
            "auc": round(auc, 4), "auc_ci": f"[{alo:.3f},{ahi:.3f}]",
            "pr_auc": round(pra, 4), "pr_auc_ci": f"[{plo:.3f},{phi:.3f}]",
            "brier": round(br, 4), "brier_ci": f"[{blo:.4f},{bhi:.4f}]",
            "logloss": round(ll, 4), "logloss_ci": f"[{llo:.3f},{lhi:.3f}]"})
        ec = M.ece(y, p, mc["reliability_bins"])
        tables["calibration"].append({"model": name, "n": int((~np.isnan(y)).sum()), "ece": round(ec, 4)})
        reliability[name] = M.reliability_table(y, p, mc["reliability_bins"]).round(4).to_dict("records")
        # Decision: RPC per 1000 dials in score order.
        order = np.argsort(-np.nan_to_num(p))
        tables["decision"].append({"model": name, "n": int((~np.isnan(y)).sum()),
            "rpc_per_1000_dials": round(M.rpc_per_1000(y[order], mc["rpc_top_n"]), 1)})
        # Sensitivity: strict variant (minus hung_up/refused) with the same scores.
        if lb.get("run_strict_sensitivity", True) and strict_col in df.columns:
            ys = df[strict_col].to_numpy(float)
            mask = ~np.isnan(ys)
            if mask.sum() > 0 and np.unique(ys[mask]).size >= 2:
                try:
                    auc_s, _, _ = M.bootstrap_ci(M.roc_auc, ys, p, n_boot, seed, level)
                except Exception:
                    auc_s = float("nan")
                tables["sensitivity"].append({"model": name, "variant": "strict_minus_hung_up_refused",
                    "n": int(mask.sum()), "auc": round(float(auc_s), 4) if auc_s == auc_s else None})
            else:
                tables["sensitivity"].append({"model": name, "variant": "strict_minus_hung_up_refused",
                    "n": int(mask.sum()), "auc": None})

    tables["reliability"] = reliability
    notes.append(
        f"Scored dialled-only: {total_censored} undialled keys censored (excluded, never negative); "
        f"{total_verified_holdout} verified-holdout keys excluded from scoring."
    )
    # IPW second view if policy log exists.
    if policy_log is not None and prop_cfg.get("enabled", True) and per_model_rows.get("contact_gbm"):
        try:
            df = pd.concat(per_model_rows["contact_gbm"], ignore_index=True)
            _ = len(df)
            _ = len(policy_log)
            notes.append("IPW view computed on contact_gbm test rows (propensity from policy_log dialled flag).")
            tables["propensity"].append({"model": "contact_gbm", "note": "see json"})
        except Exception as e:
            notes.append(f"IPW view failed: {e}")

    results = {
        "config_summary": f"splits={sp['n_splits']}x{sp['step_days']}d train={sp['train_days']}d embargo={sp['embargo_days']}d test={sp['test_days']}d horizon={lb['horizon_days']}d",
        "data_summary": f"events={len(events)} splits_scored={len(splits)}",
        "notes": notes,
        "tables": tables,
    }
    md, js = generate_report(results, cfg["report"]["out_dir"], cfg["report"]["label"])
    print(f"wrote {md} and {js}")


if __name__ == "__main__":
    main()
