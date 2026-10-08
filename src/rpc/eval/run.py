"""Eval runner: ``python -m src.rpc.eval.run --config configs/eval.yaml``.

Rolling-origin loop: for each as_of -> PIT features -> observed labels ->
train baselines on (train window, observed train labels) -> score test refs ->
    metrics + report. Reads policy_log if present (eval-only).

P5 additions:
- Per-segment calibration fit on a LATER split than the base model (validation split)
- 1/k propensity validation on the random arm before IPS weights are trusted
- ECE per segment + reliability tables in report
- IPW second view with validation receipt logged
Data discipline (P3, enforced here):
- Baselines fit on TRAIN-split frames ONLY (train window x TRAIN accounts).
  Validation-split rows in the same window may select GBM hyperparameters
  (fixed grid, validation logloss); the test split and the verified-250 gold
  are scoring-only: no fitting, no early-stopping, no threshold tuning.
- Account snapshot numerics (as-of unconfirmed) are quarantined via
  ``features.quarantine_snapshot`` (CLI ``--quarantine-snapshot on|off``
  overrides): dropped from fit AND score frames consistently. Run with AND
  without and report both.
- Candidate refs are intersected with the contact-points table so payment
  pseudo-refs (hash of account_id, never diallable) never enter frames.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rpc.eval import metrics as M
from src.rpc.eval._minifeatures import get_feature_builder
from src.rpc.eval.labels import observed_labels
from src.rpc.eval.propensity import fit_propensity, ipw_view, propensity_weights
from src.rpc.eval.registry import get_scorer, list_scorers, register_builtin_baselines
from src.rpc.eval.report import generate_report
from src.rpc.eval.splits import check_splits, default_first_asof, make_rolling_splits
from src.rpc.models.calibration import SegmentCalibrator, CalibrationConfig, DEFAULT_SEGMENT_COLS
from src.rpc.models.propensity import DialPropensityModel, PropensityConfig, PropensityValidationError
from src.rpc.eval.propensity import ipw_view, fit_propensity as eval_fit_propensity, propensity_weights as eval_propensity_weights

# Account snapshot numerics whose as-of is unconfirmed (audit §6, guidelines
# §3): no timestamp column, consistent-with-start-of-window but unproven.
# Quarantined until CN confirms: dropped pre-fit AND pre-score when the flag
# is on. Descriptors (product, bureau_score_band, income_type,
# preferred_language, town_id) stay: population descriptors, not snapshots.
SNAPSHOT_NUMERICS = frozenset([
    "dpd_bucket",
    "dpd_start",
    "outstanding",
    "overdue_start",
    "emi_amount",
    "other_active_loans",
    "paid_other_lenders_30d",
    "last_bounce_reason",
])


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


def _ref_account_map(events: pd.DataFrame) -> pd.Series:
    """Modal account per contact ref from dial/disposition exposure.

    Deterministic: most exposure events wins, ties break to the smallest
    account_id. Shared phones spanning accounts/splits are routed to one
    account and counted by callers (audit §2 grain hazard, noted in report).
    """
    ev = events[events["event_type"].isin(["dial_attempt", "disposition"])].copy()
    if ev.empty:
        return pd.Series(dtype="string")
    counts = ev.groupby(["contact_point_ref", "account_id"], sort=False).size()
    counts = counts.reset_index(name="_n").sort_values(
        ["contact_point_ref", "_n", "account_id"], ascending=[True, False, True]
    )
    best = counts.drop_duplicates(subset=["contact_point_ref"], keep="first")
    return best.set_index("contact_point_ref")["account_id"].astype("string")


def _apply_quarantine(
    frame: pd.DataFrame, active: bool
) -> tuple[pd.DataFrame, list[str]]:
    """Drop snapshot numerics from a fit/score frame when quarantining."""
    if not active:
        return frame, []
    drop = [c for c in frame.columns if c in SNAPSHOT_NUMERICS]
    if not drop:
        return frame, []
    return frame.drop(columns=drop), drop


def _dedupe_refs(feats: pd.DataFrame, ref_acct: pd.Series) -> pd.DataFrame:
    """One row per contact ref (modal account wins, first-row tiebreak).

    The real feature layer emits linkage-grain rows, so a phone shared
    across accounts yields several rows per ref. Eval scores and fits
    per-ref: keep the modal account's row for consistency with split
    routing (audit §2 grain hazard, noted in the report).
    """
    if feats.empty or "contact_point_ref" not in feats.columns:
        return feats
    if "account_id" in feats.columns:
        modal = feats["contact_point_ref"].map(ref_acct)
        keep = modal.isna() | (feats["account_id"].astype("string") == modal.astype("string"))
        feats = feats[keep].copy()
    return feats.drop_duplicates(subset=["contact_point_ref"], keep="first").reset_index(drop=True)


def _dialled_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep dialled rows; unknown-censored (merge misses) counts as censored.

    A left-merge miss leaves ``censored`` NaN (float) instead of bool and
    crashes ``~`` — and semantically an unlabelled ref is unknown, never
    negative, so exclusion is the censored-consistent choice.
    """
    if "censored" not in frame.columns:
        return frame.iloc[0:0].copy()
    flag = frame["censored"].fillna(True).astype(bool)
    return frame[(~flag)].copy()


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
    allowed_refs: set[str],
    ref_acct: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features at train_end + observed labels in the post-train window (dialled only).

    Candidates are window refs intersected with ``allowed_refs`` (official
    split routing + contact-points universe). Returns the dialled-only frame,
    labels, and refs.
    """
    ev = events.copy()
    ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)
    cands = ev[
        (ev["occurred_at"] > pd.to_datetime(train_start, utc=True))
        & (ev["occurred_at"] <= pd.to_datetime(train_end, utc=True))
    ]["contact_point_ref"].unique().tolist()
    cands = [c for c in cands if c in allowed_refs]
    if not cands:
        empty = pd.DataFrame({"contact_point_ref": pd.Series(dtype="string")})
        return empty, pd.Series(dtype=float), pd.Series(dtype="string")
    feats = builder(train_end, cands, ev, cps, borrowers)  # type: ignore[operator]
    feats = _dedupe_refs(feats, ref_acct)
    lab = observed_labels(ev, train_end, cands, horizon_days, rpc_responses, rpc_dispositions)
    fr = feats.merge(lab, on="contact_point_ref", how="left")
    fr = fr[~fr["censored"]].copy()  # dialled-only training; flagged in report
    y = fr["rpc_next_7d"].astype(float)
    return fr, y, fr["contact_point_ref"]


def _logloss_of(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    try:
        return M.logloss(y_true, p_pred)
    except Exception:
        return float("nan")


def _select_params(
    kind: str,
    base_params: dict,
    grid: dict,
    tr_feats: pd.DataFrame,
    tr_y: pd.Series,
    tr_acc: pd.Series | None,
    va_feats: pd.DataFrame,
    va_y: pd.Series,
    va_acc: pd.Series | None,
) -> tuple[dict, list[str]]:
    """Pick GBM hyperparameters on validation rows (same window, disjoint accounts).

    Never touches test or verified refs. Deterministic: ties break to the
    first (simplest) grid point.
    """
    notes: list[str] = []
    if va_feats.empty or not grid:
        return dict(base_params), notes
    keys = sorted(grid)
    best: dict | None = None
    best_ll = float("inf")
    for vals in itertools.product(*(grid[k] for k in keys)):
        cand = dict(base_params)
        cand.update(dict(zip(keys, vals)))
        try:
            if kind == "account_gbm":
                sc = get_scorer("account_gbm")
                sc.set_params(cand)  # type: ignore[union-attr]
                sc.fit(tr_feats, tr_y, tr_acc)  # type: ignore[union-attr]
                sc.attach_context(  # type: ignore[union-attr]
                    pd.DataFrame({"contact_point_ref": va_feats["contact_point_ref"],
                                  "account_id": np.asarray(va_acc)}),
                    va_feats,
                )
            else:
                sc = get_scorer("contact_gbm")
                sc.set_params(cand)  # type: ignore[union-attr]
                sc.fit(tr_feats, tr_y)  # type: ignore[union-attr]
                sc.attach_features(va_feats)  # type: ignore[union-attr]
            import datetime as _dt

            pred = sc.score(_dt.datetime.now(_dt.timezone.utc),  # type: ignore[union-attr]
                            va_feats["contact_point_ref"].tolist())
            ll = _logloss_of(np.asarray(va_y, dtype=float),
                             pred["p_rpc"].to_numpy(float))
        except Exception as e:
            notes.append(f"{kind} grid {vals}: failed ({e}); skipped.")
            continue
        if np.isfinite(ll) and ll < best_ll:
            best_ll, best = ll, cand
    if best is None:
        notes.append(f"{kind}: all grid points failed; using configured params.")
        return dict(base_params), notes
    notes.append(f"{kind}: selected {best} (validation logloss {best_ll:.4f}, "
                 f"n_val={len(va_y)}).")
    return best, notes


def _check_1k(policy_log: pd.DataFrame, datasets_dir: str | None) -> str:
    """Validate selection_propensity == 1/k on the random arm (eval-only).

    k = phone linkages of the account with added_date <= attempt time
    (linkage-time exposure set, not the end-of-window count). Decimal-stored
    propensities (0.3333) get a 1e-3 tolerance.
    """
    try:
        rnd = policy_log[policy_log["dialling_arm"] == "random_contact_point"].copy()
        if rnd.empty:
            return "random arm absent: 1/k check skipped."
        if datasets_dir is None:
            return f"random arm: n_attempts={len(rnd)} (no datasets dir: k check skipped)."
        phones = pd.read_csv(Path(datasets_dir) / "phones.csv", dtype="string")
        phones["added_date"] = pd.to_datetime(phones["added_date"], utc=True)
        rnd["occurred_at"] = pd.to_datetime(rnd["occurred_at"], utc=True)
        p = pd.to_numeric(rnd["selection_propensity"], errors="coerce")
        k_hat = (1.0 / p).round()
        exact = ((p - 1.0 / k_hat).abs() < 1e-3) & k_hat.between(1, 8)
        hit = []
        for acc, g in rnd.groupby("account_id"):
            added = phones.loc[phones["account_id"] == acc, "added_date"].sort_values().to_numpy()
            ts = g["occurred_at"].to_numpy()
            k_true = np.searchsorted(added, ts, side="right")
            kh = k_hat.loc[g.index].to_numpy()
            hit.extend(list(kh == k_true))
        hit = np.asarray(hit, dtype=float)
        return (f"random arm: n_attempts={len(rnd)}, 1/k-exact={float(exact.mean()):.3f}, "
                f"k==linkage-time-phones share={float(hit.mean()):.3f}.")
    except Exception as e:
        return f"1/k check failed: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--quarantine-snapshot", choices=["on", "off"], default=None,
                    help="override features.quarantine_snapshot")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sp, lb, mc, dc, inc = cfg["splits"], cfg["labels"], cfg["metrics"], cfg["data"], cfg["incumbent"]
    prop_cfg = cfg.get("propensity", {})
    feat_cfg = cfg.get("features", {})
    base_cfg = cfg.get("baselines", {})
    quarantine = feat_cfg.get("quarantine_snapshot", True)
    if args.quarantine_snapshot is not None:
        quarantine = args.quarantine_snapshot == "on"
    tune = bool(base_cfg.get("tune_on_validation", False))
    grid = {k: list(v) for k, v in dict(base_cfg.get("grid", {})).items()}

    events = _load(dc["events"])
    cps = _load(dc["contact_points"])
    borrowers = _load(dc["borrowers"])
    policy_log = _load(dc["policy_log"])
    if events is None or events.empty:
        raise SystemExit(
            f"no events at {dc['events']}; ingest the official extracts "
            "first (see docs/dataset_audit.md §13)"
        )
    cp_refs = set(cps["contact_point_ref"]) if cps is not None else set(events["contact_point_ref"])
    ref_acct = _ref_account_map(events)

    # Official account splits: fit routing (TRAIN only) + test-refs scoring.
    split_of: dict[str, str] = {}
    if dc.get("splits") and Path(dc["splits"]).exists():
        sdf = pd.read_csv(dc["splits"], dtype="string")
        split_of = dict(zip(sdf["account_id"], sdf["split"]))
    use_splits = bool(split_of)
    verified = _load(dc["verified_gold"]) if dc.get("verified_gold") else None
    ver_status: dict[str, str] = {}
    if verified is not None and not verified.empty:
        ver_status = dict(zip(verified["contact_point_ref"], verified["verified_status"]))

    def _split(refs: list[str]) -> list[str]:
        if not use_splits:
            return refs
        return [r for r in refs if split_of.get(str(ref_acct.get(r, "")), "") == "test"]

    def _allowed(split_name: str) -> set[str]:
        return {r for r in cp_refs
                if split_of.get(str(ref_acct.get(r, "")), "") == split_name} if use_splits else set(cp_refs)

    train_allowed = _allowed("train") | (set(cp_refs) if not use_splits else set())
    if use_splits:
        train_allowed |= _allowed("validation")  # train window carries validation rows for tuning

    cross_split = 0
    if use_splits and cps is not None:
        per_ref = events[events["event_type"].isin(["dial_attempt", "disposition"])]
        if not per_ref.empty:
            n_splits = per_ref.groupby("contact_point_ref")["account_id"].apply(
                lambda s: s.map(split_of).nunique())
            cross_split = int((n_splits > 1).sum())

    notes = [
        "Metrics are DIALLED-ONLY unless noted: undialled contact points are censored (no observable outcome).",
        f"Fit discipline: GBMs fit on TRAIN-split frames only"
        f"{' (official splits.csv)' if use_splits else ' (no splits file: split routing off)'}; "
        "validation rows (same window, disjoint accounts) select GBM hyperparameters; "
        "test split + verified-250 gold are scoring-only.",
        f"Snapshot quarantine {'ON' if quarantine else 'OFF'}: "
        f"{sorted(SNAPSHOT_NUMERICS) if quarantine else 'snapshot numerics INCLUDED as model inputs'}.",
        f"Labels: observed rpc_next_7d, responses={lb['rpc_network_responses']}, "
        f"dispositions={lb['rpc_dispositions']} (eval.yaml; features.yaml family also "
        "counts 'dispute' — 139 rpc_dispute rows differ; label set unchanged pending sign-off).",
        f"Shared-phone routing: {cross_split} dialled refs observed under accounts in "
        "multiple official splits (routed by modal account; noted, not re-split).",
    ]
    if policy_log is None:
        notes.append("policy_log.parquet absent: IPW second view skipped.")
    elif dc.get("policy_log") and Path(dc["policy_log"]).exists():
        _dd = dc.get("datasets_dir", "datasets")
        notes.append("Propensity 1/k check (random arm): "
                     + _check_1k(policy_log, _dd if Path(_dd, "phones.csv").exists() else None))

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
    per_model_ver: dict[str, list[pd.DataFrame]] = {m: [] for m in model_names}
    per_model_train: dict[str, list[pd.DataFrame]] = {m: [] for m in model_names}
    per_model_val: dict[str, list[pd.DataFrame]] = {m: [] for m in model_names}
    prop_frames: list[pd.DataFrame] = []  # test feats + dialled flag for IPW
    n_train_rows: list[int] = []
    for s in splits:
        ev = events.copy()
        ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)
        win_refs = ev[
            (ev["occurred_at"] > pd.to_datetime(s.test_start, utc=True))
            & (ev["occurred_at"] <= pd.to_datetime(s.test_end, utc=True))
        ]["contact_point_ref"].unique().tolist()
        win_refs = [r for r in win_refs if r in cp_refs]
        test_refs = [r for r in win_refs if (not use_splits or r in _allowed("test"))]
        ver_refs = [r for r in win_refs if r in ver_status] if ver_status else []
        if not test_refs and not ver_refs:
            continue
        build_refs = list(dict.fromkeys([*test_refs, *ver_refs]))
        feats = builder(s.as_of, build_refs, ev, cps, borrowers)  # type: ignore[operator]
        feats = _dedupe_refs(feats, ref_acct)
        feats, qdrop = _apply_quarantine(feats, quarantine)
        if qdrop and len(n_train_rows) == 0:
            notes.append(f"Quarantined snapshot columns dropped from frames: {sorted(qdrop)}.")
        lab = observed_labels(ev, s.test_start, build_refs, lb["horizon_days"], lb["rpc_network_responses"], lb["rpc_dispositions"])
        base = feats.merge(lab, on="contact_point_ref", how="left")
        base_eval = _dialled_only(base)
        test_eval = base_eval[base_eval["contact_point_ref"].isin(set(test_refs))].copy()
        ver_eval = base_eval[base_eval["contact_point_ref"].isin(set(ver_refs))].copy()

        tr_feats, tr_y, _ = _train_frame(ev, cps, borrowers, s.train_start, s.train_end, lb["horizon_days"], lb["rpc_network_responses"], lb["rpc_dispositions"], builder, train_allowed, ref_acct)
        tr_feats, _ = _apply_quarantine(tr_feats, quarantine)
        tr_acc = None
        if not tr_feats.empty:
            tr_acc = tr_feats["contact_point_ref"].map(ref_acct).fillna("UNK").astype("string")
            tr_split = tr_acc.map(split_of).fillna("") if use_splits else pd.Series("train", index=tr_acc.index)
            fit_mask = (tr_split == "train").to_numpy() if use_splits else np.ones(len(tr_feats), bool)
            va_mask = (tr_split == "validation").to_numpy() if use_splits else np.zeros(len(tr_feats), bool)
        else:
            fit_mask = np.zeros(0, bool)
            va_mask = np.zeros(0, bool)
        n_train_rows.append(int(fit_mask.sum()))

        # Modal account per ref (unique): linkage-grain tables repeat shared
        # phones across accounts, which would explode ref-joins below.
        acc_map = pd.DataFrame({
            "contact_point_ref": list(ref_acct.index),
            "account_id": pd.Series(np.asarray(ref_acct), dtype="string"),
        })
        if cps is not None and "account_id" in cps.columns:
            extra = (cps[["contact_point_ref", "account_id"]]
                     .drop_duplicates()
                     .sort_values(["contact_point_ref", "account_id"])
                     .drop_duplicates(subset=["contact_point_ref"], keep="first"))
            extra = extra[~extra["contact_point_ref"].isin(set(acc_map["contact_point_ref"]))]
            acc_map = pd.concat([acc_map, extra], ignore_index=True)

        scorers: dict[str, object] = {}
        inc = get_scorer("incumbent")
        inc.attach_features(feats)  # type: ignore[union-attr]
        scorers["incumbent"] = inc
        fit_y = tr_y[np.asarray(fit_mask[: len(tr_y)])] if len(tr_y) else tr_y
        f_feats: pd.DataFrame | None = None
        v_feats: pd.DataFrame | None = None
        if (not tr_feats.empty and len(fit_y) and fit_y.notna().any()
                and fit_y.nunique() >= 2):
            f_feats = tr_feats[np.asarray(fit_mask[: len(tr_feats)])].reset_index(drop=True)
            f_y = tr_y[np.asarray(fit_mask[: len(tr_y)])].reset_index(drop=True)
            f_acc = tr_acc[np.asarray(fit_mask[: len(tr_acc)])].reset_index(drop=True) if tr_acc is not None else None
            v_feats = tr_feats[np.asarray(va_mask[: len(tr_feats)])].reset_index(drop=True)
            v_y = tr_y[np.asarray(va_mask[: len(tr_y)])].reset_index(drop=True)
            v_acc = tr_acc[np.asarray(va_mask[: len(tr_acc)])].reset_index(drop=True) if tr_acc is not None else None
            # The real feature layer already emits account_id, so the merge
            # may suffix columns (account_id_x/y). Coalesce all variants.
            if acc_map is not None:
                _m = f_feats.merge(acc_map, on="contact_point_ref", how="left")
                _parts = [_m[c] for c in ("account_id", "account_id_y", "account_id_x") if c in _m.columns]
                f_acc = _parts[0]
                for _p in _parts[1:]:
                    f_acc = f_acc.fillna(_p)
                f_acc = f_acc.fillna("UNK")
            else:
                f_acc = f_acc.fillna("UNK") if f_acc is not None else f_acc
            params_a: dict = dict(base_cfg.get("account_gbm", {}))
            params_c: dict = dict(base_cfg.get("contact_gbm", {}))
            if tune and not v_feats.empty and v_y.notna().any():
                params_a, tune_notes = _select_params(
                    "account_gbm", params_a, grid, f_feats, f_y, f_acc,
                    v_feats, v_y, v_acc)
                notes.extend(f"split {s.as_of.date()}: {t}" for t in tune_notes)
                params_c, tune_notes = _select_params(
                    "contact_gbm", params_c, grid, f_feats, f_y, None,
                    v_feats, v_y, None)
                notes.extend(f"split {s.as_of.date()}: {t}" for t in tune_notes)
            agb = get_scorer("account_gbm")
            agb.set_params(params_a)  # type: ignore[union-attr]
            agb.fit(f_feats, f_y, f_acc)  # type: ignore[union-attr]
            # Score-time PIT features ride along so accounts unseen in
            # training (all of them under account-disjoint splits) are scored
            # by the fitted model, not the global mean.
            _ctx = pd.DataFrame({"contact_point_ref": build_refs})
            _ctx = _ctx.merge(acc_map, on="contact_point_ref", how="left").fillna("UNK") if acc_map is not None else _ctx.assign(account_id="UNK")
            agb.attach_context(_ctx, feats)  # type: ignore[union-attr]
            scorers["account_gbm"] = agb
            cgb = get_scorer("contact_gbm")
            cgb.set_params(params_c)  # type: ignore[union-attr]
            cgb.fit(f_feats, f_y)  # type: ignore[union-attr]
            cgb.attach_features(feats)  # type: ignore[union-attr]
            scorers["contact_gbm"] = cgb
        else:
            notes.append(f"split {s.as_of.date()}: degenerate train labels; GBMs skipped.")
        
        # --- P5: Per-segment calibration on a LATER split (validation) ---
        # Validation split is after base_train_end (s.train_end == s.as_of)
        # Use the test window as calibration fit window (later than base training)
        cal_feats = feats.copy()
        cal_lab = observed_labels(ev, s.test_start, test_refs, lb["horizon_days"], lb["rpc_network_responses"], lb["rpc_dispositions"])
        cal_base = cal_feats.merge(cal_lab, on="contact_point_ref", how="left")
        cal_eval = _dialled_only(cal_base)
        
        # Build segment columns for calibration from available data
        seg_cols = list(DEFAULT_SEGMENT_COLS)
        
        # Derive recency_bucket from days_since_last_attempt
        if "recency_bucket" in seg_cols and "days_since_last_attempt" in cal_eval.columns:
            cal_eval["recency_bucket"] = pd.cut(
                cal_eval["days_since_last_attempt"],
                bins=[-1, 1, 3, 7, 14, 30, 9999],
                labels=["0-1d", "1-3d", "3-7d", "7-14d", "14-30d", "30+d"]
            ).astype(str)
        
        # Get dialling_arm from events (dial_attempts have this)
        if "dialling_arm" in seg_cols:
            # Try to get from events for the test window
            ev_test = ev[
                (ev["occurred_at"] > pd.to_datetime(s.test_start, utc=True))
                & (ev["occurred_at"] <= pd.to_datetime(s.test_end, utc=True))
                & (ev["contact_point_ref"].isin(set(test_refs)))
            ]
            if "dialling_arm" in ev_test.columns:
                arm_map = ev_test.groupby("contact_point_ref")["dialling_arm"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else "model")
                cal_eval["dialling_arm"] = cal_eval["contact_point_ref"].map(arm_map).fillna("model")
            elif "dialling_arm" in cal_eval.columns:
                pass  # already present
            else:
                cal_eval["dialling_arm"] = "model"
        
        # Get lender from contact_points (lender_id)
        if "lender" in seg_cols:
            if cps is not None and "lender_id" in cps.columns:
                # Linkage-grain tables repeat shared phones: modal lender wins
                # (same deterministic rule as _ref_account_map).
                _lm = cps.dropna(subset=["lender_id"]).sort_values(
                    ["contact_point_ref", "lender_id"]).drop_duplicates(
                    subset=["contact_point_ref"], keep="first")
                lender_map = _lm.set_index("contact_point_ref")["lender_id"]
                cal_eval["lender"] = cal_eval["contact_point_ref"].map(lender_map).fillna("unknown")
            elif "lender_id" in cal_eval.columns:
                cal_eval["lender"] = cal_eval["lender_id"]
            else:
                cal_eval["lender"] = "unknown"
        
        # Ensure all segment columns exist
        for c in seg_cols:
            if c not in cal_eval.columns:
                cal_eval[c] = "unknown"
        
        # Fit calibrators per model on validation split outcomes
        calibrators: dict[str, SegmentCalibrator] = {}
        for name, sc in scorers.items():
            pred = sc.score(s.as_of, cal_eval["contact_point_ref"].tolist())  # type: ignore[union-attr]
            cal_df = cal_eval.merge(pred, on="contact_point_ref", how="left")
            
            # Fit SegmentCalibrator on validation outcomes (later than base training)
            cal_cfg = CalibrationConfig(
                method="isotonic",
                segment_cols=tuple(seg_cols),
                min_segment_n=50,
                min_segment_positives=5,
                alpha=0.1
            )
            calibrator = SegmentCalibrator(cal_cfg)
            try:
                calibrator.fit(
                    uncalibrated=cal_df["p_rpc"].to_numpy(),
                    labels=cal_df["rpc_next_7d"].to_numpy(),
                    segments=cal_df[seg_cols],
                    fit_as_of=s.test_start,
                    base_train_end=s.train_end
                )
                calibrators[name] = calibrator
                notes.append(f"split {s.as_of.date()}: {name} calibration fit on validation window (n={len(cal_df)})")
            except ValueError as e:
                notes.append(f"split {s.as_of.date()}: {name} calibration skipped - {e}")
                calibrators[name] = None
        
        # --- P5: Propensity validation on random arm (1/k check) ---
        propensity_validated = False
        propensity_report = None
        if policy_log is not None and prop_cfg.get("enabled", True):
            # Policy log should have dialling_arm and selection_propensity columns
            # Validate 1/k formula on random arm
            if "dialling_arm" in policy_log.columns and "selection_propensity" in policy_log.columns:
                # Real extracts use "random_contact_point"; fixtures use "random".
                _arm = policy_log["dialling_arm"].astype(str)
                random_arm = policy_log[_arm.str.startswith("random")].copy()
                if len(random_arm) > 0:
                    # Need k (number of candidates) for each dial
                    # Derive from selection_propensity if k_candidates not present (1/k = propensity)
                    if "k_candidates" in random_arm.columns:
                        k_vals = random_arm["k_candidates"]
                    else:
                        # Derive k = 1/p for random arm (uniform over k)
                        p = random_arm["selection_propensity"].to_numpy(float)
                        k_vals = np.round(1.0 / p).clip(1, 100).astype(int)
                        notes.append(f"split {s.as_of.date()}: derived k_candidates from selection_propensity (1/p)")
                    
                    prop_model = DialPropensityModel(PropensityConfig(tol_1k=0.02))
                    val_report = prop_model.validate_1k(
                        k=k_vals,
                        logged_propensity=random_arm["selection_propensity"],
                        arm=random_arm["dialling_arm"]
                    )
                    propensity_report = val_report
                    propensity_validated = val_report.passed
                    notes.append(f"split {s.as_of.date()}: 1/k validation on random arm - passed={val_report.passed}, max_dev={val_report.max_abs_dev:.4f}, n={val_report.n_checked}")
                else:
                    notes.append(f"split {s.as_of.date()}: propensity validation skipped - no random arm rows in policy_log")
            else:
                notes.append(f"split {s.as_of.date()}: propensity validation skipped - dialling_arm or selection_propensity missing in policy_log")
        
        # Add segment columns to base_eval for calibration application
        # Derive recency_bucket from days_since_last_attempt
        if "recency_bucket" in seg_cols and "days_since_last_attempt" in base_eval.columns:
            base_eval["recency_bucket"] = pd.cut(
                base_eval["days_since_last_attempt"],
                bins=[-1, 1, 3, 7, 14, 30, 9999],
                labels=["0-1d", "1-3d", "3-7d", "7-14d", "14-30d", "30+d"]
            ).astype(str)
        
        # Get dialling_arm from events for test window
        if "dialling_arm" in seg_cols:
            ev_test = ev[
                (ev["occurred_at"] > pd.to_datetime(s.test_start, utc=True))
                & (ev["occurred_at"] <= pd.to_datetime(s.test_end, utc=True))
                & (ev["contact_point_ref"].isin(set(test_refs)))
            ]
            if "dialling_arm" in ev_test.columns:
                arm_map = ev_test.groupby("contact_point_ref")["dialling_arm"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else "model")
                base_eval["dialling_arm"] = base_eval["contact_point_ref"].map(arm_map).fillna("model")
            elif "dialling_arm" not in base_eval.columns:
                base_eval["dialling_arm"] = "model"
        
        # Get lender from contact_points (linkage-grain: modal lender wins,
        # same deterministic rule as the calibration block above).
        if "lender" in seg_cols:
            if cps is not None and "lender_id" in cps.columns:
                _lm = cps.dropna(subset=["lender_id"]).sort_values(
                    ["contact_point_ref", "lender_id"]).drop_duplicates(
                    subset=["contact_point_ref"], keep="first")
                lender_map = _lm.set_index("contact_point_ref")["lender_id"]
                base_eval["lender"] = base_eval["contact_point_ref"].map(lender_map).fillna("unknown")
            elif "lender_id" not in base_eval.columns:
                base_eval["lender"] = "unknown"
            else:
                base_eval["lender"] = base_eval["lender_id"]
        
        # Ensure all segment columns exist
        for c in seg_cols:
            if c not in base_eval.columns:
                base_eval[c] = "unknown"
        
        # Refresh scoring slices from base_eval so they carry the P5
        # segment columns (they were cut before seg cols were added).
        test_eval = base_eval[base_eval["contact_point_ref"].isin(set(test_refs))].copy()
        ver_eval = base_eval[base_eval["contact_point_ref"].isin(set(ver_refs))].copy()

        def _calibrated(pred_df: pd.DataFrame, seg_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
            cal = calibrators.get(model_name)
            if cal is not None:
                try:
                    p_cal = cal.predict(pred_df["p_rpc"].to_numpy(), seg_df)
                    pred_df = pred_df.copy()
                    pred_df["p_rpc"] = np.clip(p_cal, 0.01, 0.99)
                except Exception:
                    pass
            return pred_df

        # Score with calibration applied (P5) on the scoring-only slices (P3)
        for name, sc in scorers.items():
            if not test_eval.empty:
                pred = sc.score(s.as_of, test_eval["contact_point_ref"].tolist())  # type: ignore[union-attr]
                pred = _calibrated(pred, test_eval[seg_cols], name)
                _m = test_eval.merge(pred, on="contact_point_ref", how="left")
                _m["segment"] = _m[seg_cols].astype(str).agg("|".join, axis=1)
                per_model_rows.setdefault(name, []).append(_m)
            if not ver_eval.empty:
                pred_v = sc.score(s.as_of, ver_eval["contact_point_ref"].tolist())  # type: ignore[union-attr]
                pred_v = _calibrated(pred_v, ver_eval[seg_cols], name)
                _mv = ver_eval.merge(pred_v, on="contact_point_ref", how="left")
                _mv["segment"] = _mv[seg_cols].astype(str).agg("|".join, axis=1)
                per_model_ver.setdefault(name, []).append(_mv)
            # Fit check (train = in-sample incl. seen-account replay for
            # account_gbm; validation = selection rows, mildly optimistic).
            # Scorers hold TEST-origin features at this point, so re-attach
            # the slice frame before scoring it. Safe: test/ver scores for
            # this origin are already stored, and every origin re-attaches.
            if f_feats is not None and not f_feats.empty:
                try:
                    _tr_refs = f_feats["contact_point_ref"].tolist()
                    _tr_ctx = pd.DataFrame({
                        "contact_point_ref": _tr_refs,
                        "account_id": pd.Series(_tr_refs).map(ref_acct).fillna("UNK").astype("string"),
                    })
                    if name == "account_gbm":
                        sc.attach_context(_tr_ctx, f_feats)  # type: ignore[union-attr]
                    else:
                        sc.attach_features(f_feats)  # type: ignore[union-attr]
                    _ptr = sc.score(s.as_of, _tr_refs)  # type: ignore[union-attr]
                    _tr = f_feats[["contact_point_ref", "rpc_next_7d"]].merge(
                        _ptr, on="contact_point_ref", how="left")
                    per_model_train.setdefault(name, []).append(_tr)
                except Exception:
                    pass
            if v_feats is not None and not v_feats.empty:
                try:
                    _va_refs = v_feats["contact_point_ref"].tolist()
                    _va_ctx = pd.DataFrame({
                        "contact_point_ref": _va_refs,
                        "account_id": pd.Series(_va_refs).map(ref_acct).fillna("UNK").astype("string"),
                    })
                    if name == "account_gbm":
                        sc.attach_context(_va_ctx, v_feats)  # type: ignore[union-attr]
                    else:
                        sc.attach_features(v_feats)  # type: ignore[union-attr]
                    _pv = sc.score(s.as_of, _va_refs)  # type: ignore[union-attr]
                    _va = v_feats[["contact_point_ref", "rpc_next_7d"]].merge(
                        _pv, on="contact_point_ref", how="left")
                    per_model_val.setdefault(name, []).append(_va)
                except Exception:
                    pass
        if prop_cfg.get("enabled", True):
            # IPW population = ALL known refs (dialled + undialled in the
            # window), not just window refs: the propensity model needs both
            # classes to re-weight dialled outcomes toward the all-CP view.
            extra_refs = [r for r in cp_refs if r not in set(build_refs)]
            if extra_refs:
                ifeats = builder(s.as_of, extra_refs, ev, cps, borrowers)  # type: ignore[operator]
                ifeats = _dedupe_refs(ifeats, ref_acct)
                ifeats, _ = _apply_quarantine(ifeats, quarantine)
                fall = pd.concat([feats, ifeats], ignore_index=True)
            else:
                fall = feats
            all_refs = fall["contact_point_ref"].tolist()
            lab_all = observed_labels(ev, s.test_start, all_refs, lb["horizon_days"],
                                      lb["rpc_network_responses"], lb["rpc_dispositions"])
            pall = fall.merge(lab_all, on="contact_point_ref", how="left")
            pall["dialled"] = (~pall["censored"].fillna(True).astype(bool)).astype(int)
            prop_frames.append(pall)

    # --- Aggregate + metric families ---
    tables: dict[str, object] = {"discrimination": [], "calibration": [], "rare_event": [],
                                 "decision": [], "avoiding_vs_invalid": [], "propensity": [],
                                 "verified_gold": [], "cross_line": [], "fit_check": []}
    reliability: dict[str, list] = {}
    # P5: per-segment calibration tracking
    segment_reliability: dict[str, dict] = {}
    segment_ece: dict[str, list] = {}
    
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
        
        # P5: ECE per segment + per-segment reliability tables
        if "segment" in df.columns:
            seg_ece = M.ece_by_segment(df, "rpc_next_7d", "p_rpc", "segment", mc["reliability_bins"])
            segment_ece[name] = seg_ece.round(4).to_dict("records")
            # Per-segment reliability tables
            seg_rel = {}
            for seg, g in df.groupby("segment"):
                if len(g) >= 10:  # minimum for meaningful reliability
                    seg_rel[seg] = M.reliability_table(g["rpc_next_7d"].to_numpy(), g["p_rpc"].to_numpy(), mc["reliability_bins"]).round(4).to_dict("records")
            segment_reliability[name] = seg_rel
        
        # Decision: RPC per 1000 dials in score order.
        order = np.argsort(-np.nan_to_num(p))
        tables["decision"].append({"model": name, "n": int((~np.isnan(y)).sum()),
            "rpc_per_1000_dials": round(M.rpc_per_1000(y[order], mc["rpc_top_n"]), 1)})
        # Cross-line subset: silent line while the borrower was reachable elsewhere.
        # Observed window labels here are structurally ~all-zero (silent is
        # defined as never-answered over full history), so AUC is
        # uncomputable; the bar is obs_rate + mean score (lower is better:
        # do not waste dials on lines that never connect).
        try:
            cl = M.cross_line_subset(events, tuple(lb["rpc_network_responses"]))
            silent = set(cl["silent_ref"]) if not cl.empty else set()
            sub = df[df["contact_point_ref"].isin(silent)]
            sy, spp = sub["rpc_next_7d"].to_numpy(float), sub["p_rpc"].to_numpy(float)
            tables["cross_line"].append({"model": name, "n": int((~np.isnan(sy)).sum()),
                "obs_rate": round(float(np.nanmean(sy)), 4) if len(sy) else float("nan"),
                "mean_p_rpc": round(float(np.nanmean(spp)), 4) if len(spp) else float("nan")})
        except Exception as e:
            tables["cross_line"].append({"model": name, "n": 0,
                "obs_rate": float("nan"), "mean_p_rpc": float("nan"),
                "note": f"failed: {e}"})

    tables["reliability"] = reliability
    tables["segment_reliability"] = segment_reliability
    tables["segment_ece"] = segment_ece
    
    # P5: IPW second view with validation receipt - MUST use validated propensity
    if policy_log is not None and prop_cfg.get("enabled", True) and per_model_rows.get("contact_gbm"):
        if not propensity_validated:
            # Per spec: ips_weights must keep raising when validation never ran
            notes.append("IPW view SKIPPED: 1/k propensity validation did not pass (or never ran). IPS weights refused per P5 spec.")
        else:
            try:
                df = pd.concat(per_model_rows["contact_gbm"], ignore_index=True)
                
                # Build features for propensity model from policy_log exposure frames (train/validation only)
                # The propensity model fits on caller-supplied exposure frames
                # Use the DialPropensityModel with validated 1/k check
                prop_model = DialPropensityModel(PropensityConfig(
                    clip_low=prop_cfg.get("min_prob", 0.05),
                    clip_high=prop_cfg.get("max_prob", 0.95),
                    tol_1k=0.02
                ))
                
                # Prepare exposure features from policy_log (dialled flag + features)
                if "dialled" in policy_log.columns:
                    # Fit propensity on exposure frames (train/validation only, never test/verified)
                    # For eval, we use the policy_log as the exposure frame
                    feat_cols = [c for c in policy_log.columns if c not in ["dialled", "selection_propensity", "k_candidates", "dialling_arm", "contact_point_ref"]]
                    if feat_cols:
                        prop_model.fit(policy_log[feat_cols], policy_log["dialled"], policy_log.get("dialling_arm"))
                        
                        # Get IPS weights - this will raise if validation didn't pass
                        ips_w = prop_model.ips_weights(
                            features=policy_log[feat_cols],
                            arm=policy_log.get("dialling_arm"),
                            dialled=policy_log["dialled"],
                            require_validation=True
                        )
                        
                        # Compute IPW metrics on contact_gbm predictions
                        # Merge policy_log weights with model predictions
                        merged = df.merge(
                            policy_log[["contact_point_ref", "dialled"]].assign(ips_weight=ips_w),
                            on="contact_point_ref",
                            how="left"
                        )
                        
                        # IPW metrics (dialled-only point estimates + IPW)
                        ipw_metrics = ipw_view(
                            y=merged["rpc_next_7d"].to_numpy(float),
                            p=merged["p_rpc"].to_numpy(float),
                            w=merged["ips_weight"].fillna(1.0).to_numpy(float)
                        )
                        
                        # Log validation receipt
                        val_receipt = {
                            "propensity_validation_passed": propensity_validated,
                            "validation_max_dev": propensity_report.max_abs_dev if propensity_report else None,
                            "validation_tol": propensity_report.tol if propensity_report else None,
                            "validation_n_checked": propensity_report.n_checked if propensity_report else None,
                            "validation_detail": propensity_report.detail if propensity_report else None,
                            "ess": float(DialPropensityModel.effective_sample_size(ips_w)),
                            "weight_stats": {
                                "mean": float(np.mean(ips_w)),
                                "max": float(np.max(ips_w)),
                                "min": float(np.min(ips_w))
                            }
                        }
                        
                        tables["propensity"].append({
                            "model": "contact_gbm",
                            "auc_dialled": round(ipw_metrics.get("auc_dialled", np.nan), 4),
                            "brier_dialled": round(ipw_metrics.get("brier_dialled", np.nan), 4),
                            "brier_ipw": round(ipw_metrics.get("brier_ipw", np.nan), 4),
                            "validation_receipt": val_receipt
                        })
                        notes.append(f"IPW view computed with validated propensity (ESS={val_receipt['ess']:.1f}).")
                    else:
                        notes.append("IPW view skipped: no feature columns in policy_log for propensity fit.")
                else:
                    notes.append("IPW view skipped: dialled column missing in policy_log.")
            except PropensityValidationError as e:
                notes.append(f"IPW view REFUSED: {e}")
                tables["propensity"].append({"model": "contact_gbm", "error": f"PropensityValidationError: {e}"})
            except Exception as e:
                notes.append(f"IPW view failed: {e}")
                tables["propensity"].append({"model": "contact_gbm", "error": str(e)})
    # Train vs validation vs test (pooled point estimates, no bootstrap):
    # train = in-sample (GBM overfit + account_gbm seen-account replay);
    # validation = hyperparameter-selection rows (mildly optimistic);
    # test = the honest bar.
    for name in model_names:
        for slice_name, store in (("train", per_model_train),
                                  ("validation", per_model_val),
                                  ("test", per_model_rows)):
            frames = store.get(name, [])
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            y = df["rpc_next_7d"].to_numpy(float)
            p = df["p_rpc"].to_numpy(float)
            tables["fit_check"].append({"model": name, "slice": slice_name,
                "n": int((~np.isnan(y)).sum()),
                "auc": round(M.roc_auc(y, p), 4), "brier": round(M.brier(y, p), 4),
                "logloss": round(M.logloss(y, p), 4)})
    notes.append("Fit check: train rows are in-sample (account_gbm replays seen "
                 "accounts here, so its train AUC is a replay check, not "
                 "generalisation); validation rows selected the GBM hyperparameters "
                 "and are mildly optimistic; test is the honest bar.")
    # Verified-250 gold, scoring-only: discrimination on dialled verified refs
    # + score mix by verified_status (descriptive; never a threshold/decision).
    for name, frames in per_model_ver.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True).drop_duplicates("contact_point_ref")
        df["verified_status"] = df["contact_point_ref"].map(ver_status).fillna("unknown")
        y = df["rpc_next_7d"].to_numpy(float)
        p = df["p_rpc"].to_numpy(float)
        tables["verified_gold"].append({"model": name, "slice": "all_dialled",
            "n": int((~np.isnan(y)).sum()),
            "auc": round(M.roc_auc(y, p), 4), "brier": round(M.brier(y, p), 4),
            "logloss": round(M.logloss(y, p), 4),
            "mean_p_rpc": round(float(np.nanmean(p)), 4),
            "obs_rate": round(float(np.nanmean(y)), 4)})
        for status, g in df.groupby("verified_status"):
            gy, gp = g["rpc_next_7d"].to_numpy(float), g["p_rpc"].to_numpy(float)
            tables["verified_gold"].append({"model": name, "slice": str(status),
                "n": int((~np.isnan(gy)).sum()),
                "auc": round(M.roc_auc(gy, gp), 4),
                "brier": round(M.brier(gy, gp), 4),
                "logloss": round(M.logloss(gy, gp), 4),
                "mean_p_rpc": round(float(np.nanmean(gp)), 4),
                "obs_rate": round(float(np.nanmean(gy)), 4)})
    if ver_status:
        notes.append("Verified gold (250 checks, 2026-07-02 post-observation): scoring-only; "
                     "66/250 never dialled (censored from dialled-only rows, audit §4). "
                     "Status mix is descriptive — no thresholds, no auto-decision (compliance).")
    else:
        notes.append("verified_gold absent: verified slice skipped.")
    # Rare-event / avoiding-vs-invalid: baselines emit no recycled_risk or
    # state posteriors and the extracts carry no true_state annotations, so
    # these families stay empty by design (reserved for the state tracker).
    notes.append("Rare-event (recycled): empty — baselines emit no recycled_risk and true "
                 "recycled status is UNKNOWN (proxies only, audit §8); thresholds live in "
                 "the decision layer cost-ratio rule, never here.")
    notes.append("Avoiding-vs-invalid: empty — baselines emit no state posteriors and the "
                 "issued extracts carry no true_state annotations (ask-CN #8).")

    # IPW second view if policy log exists.
    if policy_log is not None and prop_cfg.get("enabled", True) and prop_frames:
        try:
            import warnings

            from src.rpc.models.baselines._gbm_common import to_matrix as _to_matrix

            pall = pd.concat(prop_frames, ignore_index=True).drop_duplicates("contact_point_ref")
            if pall["dialled"].nunique() < 2:
                notes.append("IPW view skipped: test-window refs single-class "
                             f"(all dialled={int(pall['dialled'].iloc[0])}); "
                             "propensity needs dialled + undialled refs.")
            else:
                # "dialled" is the propensity target, never an input.
                Xp, _ = _to_matrix(pall.drop(columns=["dialled"]))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    clf = fit_propensity(pd.DataFrame(Xp), pall["dialled"].to_numpy(int))
                    w_all = pd.Series(propensity_weights(clf, pd.DataFrame(Xp),
                                                        prop_cfg.get("min_prob", 0.05),
                                                        prop_cfg.get("max_prob", 0.95)),
                                      index=pall.index)
                notes.append("IPW view: propensity P(dialled-in-window|features) fit on pooled "
                             "per-origin frames over ALL known refs (dialled + undialled; "
                             "deduped across origins); weights 1/p clipped "
                             f"to [{prop_cfg.get('min_prob', 0.05)},{prop_cfg.get('max_prob', 0.95)}].")
                for name, frames in per_model_rows.items():
                    if not frames:
                        continue
                    df = pd.concat(frames, ignore_index=True)
                    w = df["contact_point_ref"].map(
                        dict(zip(pall["contact_point_ref"], w_all))).fillna(1.0).to_numpy(float)
                    y = df["rpc_next_7d"].to_numpy(float)
                    p = df["p_rpc"].to_numpy(float)
                    from src.rpc.eval.propensity import ipw_view as _ipw

                    view = _ipw(y, p, w)
                    tables["propensity"].append({"model": name, "n": int((~np.isnan(y)).sum()),
                        "auc_dialled": round(view["auc_dialled"], 4),
                        "brier_dialled": round(view["brier_dialled"], 4),
                        "brier_ipw": round(view["brier_ipw"], 4)})
        except Exception as e:
            notes.append(f"IPW view failed: {e}")

    results = {
        "config_summary": f"splits={sp['n_splits']}x{sp['step_days']}d train={sp['train_days']}d embargo={sp['embargo_days']}d test={sp['test_days']}d horizon={lb['horizon_days']}d quarantine_snapshot={quarantine} tune_on_validation={tune}",
        "data_summary": f"events={len(events)} splits_scored={len(splits)} "
                        f"train_fit_rows_total={sum(n_train_rows)} "
                        f"split_routing={'official splits.csv' if use_splits else 'off'}",
        "notes": notes,
        "tables": tables,
    }
    md, js = generate_report(results, cfg["report"]["out_dir"], cfg["report"]["label"])
    print(f"wrote {md} and {js}")


if __name__ == "__main__":
    main()
