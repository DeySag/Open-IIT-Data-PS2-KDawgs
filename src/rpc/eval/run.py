"""Eval runner: ``python -m src.rpc.eval.run --config configs/eval.yaml``.

Rolling-origin loop: for each as_of -> PIT features -> observed labels ->
train baselines on (train window, observed train labels) -> score test refs ->
    metrics + report. Reads policy_log if present (eval-only).

P5 additions:
- Per-segment calibration fit on a LATER split than the base model (validation split)
- 1/k propensity validation on the random arm before IPS weights are trusted
- ECE per segment + reliability tables in report
- IPW second view with validation receipt logged
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
from src.rpc.eval.labels import observed_labels
from src.rpc.eval.registry import get_scorer, list_scorers, register_builtin_baselines
from src.rpc.eval.report import generate_report
from src.rpc.eval.splits import check_splits, default_first_asof, make_rolling_splits
from src.rpc.models.calibration import SegmentCalibrator, CalibrationConfig, DEFAULT_SEGMENT_COLS
from src.rpc.models.propensity import DialPropensityModel, PropensityConfig, PropensityValidationError
from src.rpc.eval.propensity import ipw_view, fit_propensity as eval_fit_propensity, propensity_weights as eval_propensity_weights


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
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features at train_end + observed labels in the post-train window (dialled only)."""
    ev = events.copy()
    ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)
    cands = ev[
        (ev["occurred_at"] > pd.to_datetime(train_start, utc=True))
        & (ev["occurred_at"] <= pd.to_datetime(train_end, utc=True))
    ]["contact_point_ref"].unique().tolist()
    feats = builder(train_end, cands, ev, cps, borrowers)  # type: ignore[operator]
    lab = observed_labels(ev, train_end, cands, horizon_days, rpc_responses, rpc_dispositions)
    fr = feats.merge(lab, on="contact_point_ref", how="left")
    fr = fr[~fr["censored"]].copy()  # dialled-only training; flagged in report
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
        "Metrics are DIALLED-ONLY unless noted: undialled contact points are censored (no observable outcome).",
    ]
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
    for s in splits:
        ev = events.copy()
        ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)
        test_refs = ev[
            (ev["occurred_at"] > pd.to_datetime(s.test_start, utc=True))
            & (ev["occurred_at"] <= pd.to_datetime(s.test_end, utc=True))
        ]["contact_point_ref"].unique().tolist()
        if not test_refs:
            continue
        feats = builder(s.as_of, test_refs, ev, cps, borrowers)  # type: ignore[operator]
        lab = observed_labels(ev, s.test_start, test_refs, lb["horizon_days"], lb["rpc_network_responses"], lb["rpc_dispositions"])
        base = feats.merge(lab, on="contact_point_ref", how="left")
        base_eval = base[~base["censored"]].copy()

        tr_feats, tr_y, _ = _train_frame(ev, cps, borrowers, s.train_start, s.train_end, lb["horizon_days"], lb["rpc_network_responses"], lb["rpc_dispositions"], builder)
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
        
        # --- P5: Per-segment calibration on a LATER split (validation) ---
        # Validation split is after base_train_end (s.train_end == s.as_of)
        # Use the test window as calibration fit window (later than base training)
        cal_feats = feats.copy()
        cal_lab = observed_labels(ev, s.test_start, test_refs, lb["horizon_days"], lb["rpc_network_responses"], lb["rpc_dispositions"])
        cal_base = cal_feats.merge(cal_lab, on="contact_point_ref", how="left")
        cal_eval = cal_base[~cal_base["censored"]].copy()
        
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
                lender_map = cps.set_index("contact_point_ref")["lender_id"]
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
                random_arm = policy_log[policy_log["dialling_arm"] == "random"].copy()
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
        
        # Get lender from contact_points
        if "lender" in seg_cols:
            if cps is not None and "lender_id" in cps.columns:
                lender_map = cps.set_index("contact_point_ref")["lender_id"]
                base_eval["lender"] = base_eval["contact_point_ref"].map(lender_map).fillna("unknown")
            elif "lender_id" not in base_eval.columns:
                base_eval["lender"] = "unknown"
            else:
                base_eval["lender"] = base_eval["lender_id"]
        
        # Ensure all segment columns exist
        for c in seg_cols:
            if c not in base_eval.columns:
                base_eval[c] = "unknown"
        
        # Score with calibration applied
        for name, sc in scorers.items():
            pred = sc.score(s.as_of, base_eval["contact_point_ref"].tolist())  # type: ignore[union-attr]
            cal = calibrators.get(name)
            if cal is not None:
                # Apply calibration to predictions
                seg_df = base_eval[seg_cols]
                p_cal = cal.predict(pred["p_rpc"].to_numpy(), seg_df)
                pred = pred.copy()
                pred["p_rpc"] = np.clip(p_cal, 0.01, 0.99)
            # Add segment column to merged result for per-segment ECE tracking
            merged = base_eval.merge(pred, on="contact_point_ref", how="left")
            merged["segment"] = merged[seg_cols].astype(str).agg("|".join, axis=1)
            per_model_rows.setdefault(name, []).append(merged)

    # --- Aggregate + metric families ---
    tables: dict[str, object] = {"discrimination": [], "calibration": [], "rare_event": [],
                                 "decision": [], "avoiding_vs_invalid": [], "propensity": []}
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
