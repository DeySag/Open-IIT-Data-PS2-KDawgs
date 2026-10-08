"""Unit + PIT + leakage tests for the time-slot RPC model."""

from __future__ import annotations

import pandas as pd

from src.rpc.models.slot import (
    SlotRPCModel,
    attach_best_slot,
    attempt_is_rpc,
    load_slot_params,
    to_slot,
)

SLOTS = {"morning": [8, 12], "afternoon": [12, 16], "evening": [16, 19]}
TZ = "Asia/Kolkata"
N_MORNING_RPC = 20
N_EVENING_RPC = 4
N_SLOT_ATTEMPTS = 40
N_FITTED_ROWS = 80


def _attempts() -> pd.DataFrame:
    # Morning: 40 dials, 20 RPC. Evening: 40 dials, 4 RPC. IST wall-clock.
    rows = []
    for i in range(N_SLOT_ATTEMPTS):
        rows.append({
            "occurred_at": pd.Timestamp("2026-04-01 09:00", tz=TZ),
            "network_response": "answered",
            "disposition": "RPC" if i < N_MORNING_RPC else "no_answer",
            "lender_id": "L01",
            "seq": i,
        })
    for i in range(N_SLOT_ATTEMPTS):
        rows.append({
            "occurred_at": pd.Timestamp("2026-04-01 17:00", tz=TZ),
            "network_response": "answered",
            "disposition": "RPC" if i < N_EVENING_RPC else "no_answer",
            "lender_id": "L01",
            "seq": 100 + i,
        })
    return pd.DataFrame(rows)


def test_slot_binning_boundaries() -> None:
    assert to_slot("2026-04-01 08:00+05:30", SLOTS, TZ) == "morning"
    assert to_slot("2026-04-01 11:59+05:30", SLOTS, TZ) == "morning"
    assert to_slot("2026-04-01 12:00+05:30", SLOTS, TZ) == "afternoon"
    assert to_slot("2026-04-01 18:59+05:30", SLOTS, TZ) == "evening"
    assert to_slot("2026-04-01 19:00+05:30", SLOTS, TZ) == "off_hours"
    assert to_slot("2026-04-01 03:00+05:30", SLOTS, TZ) == "off_hours"


def test_rpc_requires_answered_and_sanctioned() -> None:
    assert attempt_is_rpc("answered", "RPC")
    assert attempt_is_rpc("answered", "promise_to_pay")
    assert not attempt_is_rpc("answered", "no_answer")
    # rpc_* without answer is quarantined, never positive
    assert not attempt_is_rpc("no_answer", "rpc_ptp")
    assert not attempt_is_rpc("busy", "RPC")


def test_morning_beats_evening_and_clipping() -> None:
    params = load_slot_params()
    params["min_samples"] = 10
    m = SlotRPCModel(params).fit(_attempts(), pd.Timestamp("2026-04-02", tz="UTC"))
    assert m.multiplier("morning", "L01") > 1.0
    assert m.multiplier("evening", "L01") < 1.0
    assert params["clip_lo"] <= m.multiplier("morning", "L01") <= params["clip_hi"]
    assert m.best_slot("L01") == "morning"
    # apply() keeps probabilities in [0, 1]
    assert 0.0 <= m.apply(0.5, "morning", "L01") <= 1.0


def test_thin_and_unseen_fall_back_to_neutral() -> None:
    params = load_slot_params()
    params["min_samples"] = 1000
    m = SlotRPCModel(params).fit(_attempts(), pd.Timestamp("2026-04-02", tz="UTC"))
    assert m.multiplier("morning", "L01") == 1.0
    assert m.best_slot("L01") is None
    # unseen segment falls back rather than inventing signal
    params2 = load_slot_params()
    params2["min_samples"] = 10
    m2 = SlotRPCModel(params2).fit(_attempts(), pd.Timestamp("2026-04-02", tz="UTC"))
    assert m2.multiplier("morning", "L99") == 1.0


def test_pit_excludes_future_rows() -> None:
    df = _attempts()
    future = pd.DataFrame([{
        "occurred_at": pd.Timestamp("2026-05-01 09:00", tz=TZ),
        "network_response": "answered",
        "disposition": "RPC",
        "lender_id": "L01",
        "seq": 999,
    }])
    df = pd.concat([df, future], ignore_index=True)
    params = load_slot_params()
    params["min_samples"] = 10
    m = SlotRPCModel(params).fit(df, pd.Timestamp("2026-04-02", tz="UTC"))
    assert m._global_n == N_FITTED_ROWS  # future row excluded


def test_banned_columns_never_read() -> None:
    df = _attempts()
    with_banned = df.copy()
    with_banned["verified_status"] = "borrower_number"
    with_banned["true_state"] = "valid"
    params = load_slot_params()
    params["min_samples"] = 10
    as_of = pd.Timestamp("2026-04-02", tz="UTC")
    a = SlotRPCModel(params).fit(df, as_of).to_dict()
    b = SlotRPCModel(load_slot_params() | {"min_samples": 10}).fit(with_banned, as_of).to_dict()
    assert a["rates"] == b["rates"]


def test_attach_best_slot_sets_neutral_when_thin() -> None:
    class _Scorer:
        def __init__(self) -> None:
            self.mults: dict[str, float] = {}

        def attach_slot_multipliers(self, mults: dict[str, float]) -> None:
            self.mults = mults

    params = load_slot_params()
    params["min_samples"] = 10
    m = SlotRPCModel(params).fit(_attempts(), pd.Timestamp("2026-04-02", tz="UTC"))
    sc = _Scorer()
    attach_best_slot(sc, m, ["P1", "P2"], "L01")
    assert set(sc.mults) == {"P1", "P2"}
    assert all(v > 1.0 for v in sc.mults.values())  # best slot lifts
    sc2 = _Scorer()
    attach_best_slot(sc2, m, ["P9"], "L99")  # unseen segment -> neutral
    assert sc2.mults == {}
