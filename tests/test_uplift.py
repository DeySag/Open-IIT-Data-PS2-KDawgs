"""Unit + censoring + leakage tests for the trace-outcome (uplift) model."""

from __future__ import annotations

import pandas as pd

from src.rpc.models.uplift import TraceOutcomeModel, load_uplift_params, trace_outcome_labels

N_POSITIVE = 20
N_NEGATIVE = 20
CLIP_FLOOR = 0.01
LABEL_WINDOW_DAYS = 30


def _traces_payments() -> tuple[pd.DataFrame, pd.DataFrame]:
    traces = pd.DataFrame([
        {"trace_id": "T1", "account_id": "A1", "trace_date": "2026-05-01"},
        {"trace_id": "T2", "account_id": "A2", "trace_date": "2026-05-01"},
        {"trace_id": "T3", "account_id": "A3", "trace_date": "2026-07-20"},
    ])
    payments = pd.DataFrame([
        {"account_id": "A1", "payment_ts": "2026-05-10", "amount": 100.0},
        {"account_id": "A2", "payment_ts": "2026-08-10", "amount": 100.0},
    ])
    return traces, payments


def test_labels_window_and_censoring() -> None:
    traces, payments = _traces_payments()
    lab = trace_outcome_labels(traces, payments, window_days=30).set_index("trace_id")
    assert lab.loc["T1", "paid"] == 1.0  # 9d after trace, inside window
    assert lab.loc["T2", "paid"] == 0.0  # 100d after trace, outside window
    assert lab.loc["T3", "censored"]  # window exceeds the payment feed
    assert pd.isna(lab.loc["T3", "paid"])  # censored is unknown, never 0


def test_model_learns_separation_and_is_deterministic() -> None:
    feats = pd.DataFrame([
        {"trace_id": f"H{i}", "n_attempts": 20.0, "consec_failures": 15.0}
        for i in range(N_POSITIVE)
    ] + [
        {"trace_id": f"L{i}", "n_attempts": 2.0, "consec_failures": 1.0}
        for i in range(N_NEGATIVE)
    ])
    labs = pd.DataFrame([
        {"trace_id": f"H{i}", "paid": 1.0, "censored": False} for i in range(N_POSITIVE)
    ] + [
        {"trace_id": f"L{i}", "paid": 0.0, "censored": False} for i in range(N_NEGATIVE)
    ])
    m = TraceOutcomeModel({"window_days": 30, "gbm": {"n_estimators": 10,
                           "learning_rate": 0.1, "num_leaves": 7,
                           "min_child_samples": 5}, "seed": 42}).fit(feats, labs)
    got = m.predict(feats).set_index("trace_id")["p_recover_30d"]
    assert got[[f"H{i}" for i in range(N_POSITIVE)]].mean() > \
        got[[f"L{i}" for i in range(N_NEGATIVE)]].mean()
    m2 = TraceOutcomeModel(m.params).fit(feats, labs)
    assert (m.predict(feats)["p_recover_30d"] == m2.predict(feats)["p_recover_30d"]).all()


def test_censored_excluded_and_degenerate_falls_back() -> None:
    feats = pd.DataFrame([{"trace_id": "C1", "n_attempts": 5.0}])
    labs = pd.DataFrame([{"trace_id": "C1", "paid": float("nan"), "censored": True}])
    m = TraceOutcomeModel().fit(feats, labs)
    assert m.predict(feats)["p_recover_30d"].iloc[0] == CLIP_FLOOR  # clipped base 0.0


def test_banned_and_post_treatment_never_features() -> None:
    feats = pd.DataFrame([
        {"trace_id": "T1", "n_attempts": 5.0, "result": "new_phone_found",
         "verified_status": "borrower_number", "paid": 1.0},
        {"trace_id": "T2", "n_attempts": 6.0, "result": "no_new_info",
         "verified_status": "borrower_number", "paid": 0.0},
    ])
    labs = pd.DataFrame([
        {"trace_id": "T1", "paid": 1.0, "censored": False},
        {"trace_id": "T2", "paid": 0.0, "censored": False},
    ])
    m = TraceOutcomeModel().fit(feats, labs)
    assert "result" not in m._columns
    assert "verified_status" not in m._columns
    assert "paid" not in m._columns


def test_params_come_from_config() -> None:
    assert load_uplift_params()["window_days"] == LABEL_WINDOW_DAYS
