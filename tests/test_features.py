"""Tests for the point-in-time feature layer.

Covers the 11 required checks: leakage, late arrival, duplicates,
micro-fixture hand-computed values, never-attempted rows, cross-line
avoiding-vs-invalid signal, lender-local sharing, no ground truth, text
extraction, determinism/schema/registry match.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.rpc.features.features import build_features, build_training_table
from src.rpc.features.labels import build_labels
from src.rpc.features.source import DataFrameEventSource
from src.rpc.features.spec import (
    KEY_COLUMNS,
    META_COLUMNS,
    build_registry,
    feature_names,
    load_feature_config,
    snapshot_id,
)
from src.rpc.features.text import extract_switched_off_months_text, load_patterns

T = "2026-08-15T00:00:00+00:00"  # a Saturday; also in configs holidays
T2 = "2026-08-21T00:00:00+00:00"

HIDDEN_STATE_TOKENS = (
    "valid_reachable",
    "avoiding",
    "temp_unreachable",
    "switched_off_long",
    "recycled",
    "invalid",
    "true_state",
    "ground_truth",
    "shared_reason",
)


def _ev(event_id, event_type, lender, borrower, account, ref, occurred, received, payload):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "lender_id": lender,
        "borrower_id": borrower,
        "account_id": account,
        "contact_point_ref": ref,
        "occurred_at": occurred,
        "received_at": received,
        "payload": json.dumps(payload),
    }


def _cp(ref, borrower, lender, type_, source, primary, created):
    return {
        "contact_point_ref": ref,
        "borrower_id": borrower,
        "lender_id": lender,
        "type": type_,
        "value_hash": ref,
        "source": source,
        "is_primary": primary,
        "created_at": created,
    }


def _bor(borrower, lender, product="msme", bucket="31-60", out=50000.0, secured=True):
    return {
        "borrower_id": borrower,
        "lender_id": lender,
        "product": product,
        "dpd_bucket": bucket,
        "dpd_days": 45,
        "outstanding": out,
        "secured": secured,
    }


def make_fixture(extra_events: list[dict] | None = None):
    """Hand-built micro-fixture (~30 events). See module docstring/tests."""
    L1, L2 = "LENDER_001", "LENDER_002"
    B1, B2, B9 = "BORR_0001", "BORR_0002", "BORR_0009"
    A1, A2 = "ACC_0001", "ACC_0002"
    events = [
        # CPA: silent line (B1)
        _ev("d0", "dial_attempt", L1, B1, A1, "CP_A", "2026-08-08T05:00:00+00:00",
            "2026-08-08T05:05:00+00:00", {"network_response": "busy", "ring_seconds": 1.0}),
        _ev("d1", "dial_attempt", L1, B1, A1, "CP_A", "2026-08-14T04:00:00+00:00",
            "2026-08-14T04:05:00+00:00", {"network_response": "no_answer", "ring_seconds": 30.0}),
        # duplicate id, later received_at, conflicting payload -> must be ignored
        _ev("d1", "dial_attempt", L1, B1, A1, "CP_A", "2026-08-14T04:00:00+00:00",
            "2026-08-14T06:00:00+00:00", {"network_response": "answered", "ring_seconds": 2.0}),
        _ev("d2", "dial_attempt", L1, B1, A1, "CP_A", "2026-08-14T08:00:00+00:00",
            "2026-08-14T08:05:00+00:00", {"network_response": "switched_off", "ring_seconds": 0.5}),
        _ev("d3", "dial_attempt", L1, B1, A1, "CP_A", "2026-08-10T06:00:00+00:00",
            "2026-08-10T06:05:00+00:00", {"network_response": "no_answer", "ring_seconds": 1.0}),
        _ev("d4", "dial_attempt", L1, B1, A1, "CP_A", "2026-07-01T06:00:00+00:00",
            "2026-07-01T06:05:00+00:00", {"network_response": "no_answer", "ring_seconds": 25.0}),
        # late arrival: occurred <= T but received after T
        _ev("dLate", "dial_attempt", L1, B1, A1, "CP_A", "2026-08-13T06:00:00+00:00",
            "2026-08-20T00:00:00+00:00", {"network_response": "no_answer", "ring_seconds": 20.0}),
        # CPB: answered line (B1)
        _ev("d5", "dial_attempt", L1, B1, A1, "CP_B", "2026-08-14T07:00:00+00:00",
            "2026-08-14T07:05:00+00:00", {"network_response": "answered", "ring_seconds": 4.0}),
        _ev("d6", "dial_attempt", L1, B1, A1, "CP_B", "2026-08-13T03:00:00+00:00",
            "2026-08-13T03:05:00+00:00", {"network_response": "busy", "ring_seconds": 1.0}),
        # dispositions on CP_A
        _ev("x1", "disposition", L1, B1, A1, "CP_A", "2026-08-11T10:00:00+00:00",
            "2026-08-11T10:05:00+00:00", {"disposition": "wrong_number", "agent_id": "AG1",
                                          "remarks": "Wrong number hai, kisi aur ka number"}),
        _ev("x2", "disposition", L1, B1, A1, "CP_A", "2026-08-12T10:00:00+00:00",
            "2026-08-12T10:05:00+00:00", {"disposition": "third_party", "agent_id": "AG1",
                                          "remarks": "Phone uthaya bhai ne"}),
        _ev("x3", "disposition", L1, B1, A1, "CP_A", "2026-08-13T10:00:00+00:00",
            "2026-08-13T10:05:00+00:00", {"disposition": "switched_off", "agent_id": "AG2",
                                          "remarks": "Number band hai 2 mahine se"}),
        # dispositions on CP_B
        _ev("x4", "disposition", L1, B1, A1, "CP_B", "2026-08-14T07:30:00+00:00",
            "2026-08-14T07:35:00+00:00", {"disposition": "RPC", "agent_id": "AG1",
                                          "remarks": "Spoke to borrower, will pay"}),
        _ev("x5", "disposition", L1, B1, A1, "CP_B", "2026-08-14T09:00:00+00:00",
            "2026-08-14T09:05:00+00:00", {"disposition": "callback", "agent_id": "AG1",
                                          "remarks": "Baad me call karo"}),
        # bot transcripts on CP_B
        _ev("b1", "bot_transcript", L1, B1, A1, "CP_B", "2026-08-14T07:05:00+00:00",
            "2026-08-14T07:06:00+00:00", {"transcript": "Haan main borrower bol raha hoon",
                                          "who_answered": "borrower"}),
        _ev("b2", "bot_transcript", L1, B1, A1, "CP_B", "2026-08-14T07:06:00+00:00",
            "2026-08-14T07:07:00+00:00", {"transcript": "Phone uthaya bhai ne, kaun bol raha hai"}),
        # payment confirming CP_B (within 7d after answered d5 / RPC x4)
        _ev("p1", "payment", L1, B1, A1, "CP_B", "2026-08-14T12:00:00+00:00",
            "2026-08-14T12:05:00+00:00", {"amount": 1000.0, "payment_mode": "upi"}),
        # contact point update for CP_A
        _ev("u1", "contact_point_update", L1, B1, A1, "CP_A", "2026-08-01T00:00:00+00:00",
            "2026-08-01T01:00:00+00:00", {"source": "skip_trace", "contact_type": "phone",
                                          "contact_value": "+9112345", "is_primary": True}),
        # field visits on CP_C (address)
        _ev("v1", "field_visit", L1, B1, A1, "CP_C", "2026-08-12T05:00:00+00:00",
            "2026-08-12T06:00:00+00:00", {"outcome": "locked_premises", "dwell_seconds": 120,
                                          "visit_time": "2026-08-12T05:00:00+00:00"}),
        _ev("v2", "field_visit", L1, B1, A1, "CP_C", "2026-08-05T05:00:00+00:00",
            "2026-08-05T06:00:00+00:00", {"outcome": "met_borrower", "dwell_seconds": 300,
                                          "visit_time": "2026-08-05T05:00:00+00:00"}),
        # shared ref SH: B2 line + B1 row (no dials) + same hash at L2 (must not link)
        _ev("d7", "dial_attempt", L1, B2, A2, "SH", "2026-08-14T05:00:00+00:00",
            "2026-08-14T05:05:00+00:00", {"network_response": "no_answer", "ring_seconds": 10.0}),
        _ev("d8", "dial_attempt", L1, B2, A2, "CP_B2", "2026-08-14T05:30:00+00:00",
            "2026-08-14T05:35:00+00:00", {"network_response": "answered", "ring_seconds": 5.0}),
    ]
    if extra_events:
        events.extend(extra_events)
    cps = pd.DataFrame([
        _cp("CP_A", B1, L1, "phone", "KYC", True, "2026-01-01T00:00:00+00:00"),
        _cp("CP_B", B1, L1, "phone", "bureau", False, "2026-02-01T00:00:00+00:00"),
        _cp("CP_C", B1, L1, "address", "KYC", False, "2026-03-01T00:00:00+00:00"),
        _cp("CP_N", B1, L1, "phone", "KYC", False, "2026-04-01T00:00:00+00:00"),
        _cp("SH", B1, L1, "phone", "KYC", False, "2026-01-10T00:00:00+00:00"),
        _cp("SH", B2, L1, "phone", "KYC", True, "2026-01-15T00:00:00+00:00"),
        _cp("CP_B2", B2, L1, "phone", "bureau", False, "2026-02-15T00:00:00+00:00"),
        _cp("SH", B9, L2, "phone", "KYC", True, "2026-01-20T00:00:00+00:00"),
    ])
    borrowers = pd.DataFrame([_bor(B1, L1), _bor(B2, L1), _bor(B9, L2)])
    return DataFrameEventSource(pd.DataFrame(events), cps, borrowers)


def row_for(out: pd.DataFrame, lender: str, borrower: str, ref: str) -> pd.Series:
    sub = out[(out["lender_id"] == lender) & (out["borrower_id"] == borrower)
              & (out["contact_point_ref"] == ref)]
    assert len(sub) == 1, f"expected 1 row for {(lender, borrower, ref)}, got {len(sub)}"
    return sub.iloc[0]


L1, B1, B2 = "LENDER_001", "BORR_0001", "BORR_0002"


# 1. Leakage ------------------------------------------------------------------
def test_leakage_future_events_do_not_change_snapshot():
    src1 = make_fixture()
    out1 = build_features(T, src1)
    extra = [
        _ev("dFut", "dial_attempt", L1, B1, "ACC_0001", "CP_A", "2026-08-16T06:00:00+00:00",
            "2026-08-16T06:05:00+00:00", {"network_response": "answered", "ring_seconds": 3.0}),
        _ev("xFut", "disposition", L1, B1, "ACC_0001", "CP_A", "2026-08-17T10:00:00+00:00",
            "2026-08-17T10:05:00+00:00", {"disposition": "RPC", "remarks": "future rpc"}),
        _ev("pFut", "payment", L1, B1, "ACC_0001", "CP_A", "2026-08-18T00:00:00+00:00",
            "2026-08-18T00:05:00+00:00", {"amount": 500.0, "payment_mode": "cash"}),
    ]
    out2 = build_features(T, make_fixture(extra))
    assert_frame_equal(out1, out2, check_dtype=True)


# 2. Late arrival ---------------------------------------------------------------
def test_late_arrival_excluded_then_included():
    out_t = build_features(T, make_fixture())
    r = row_for(out_t, L1, B1, "CP_A")
    # dLate (occurred 08-13, received 08-20) must be excluded at T
    assert r["n_attempts_14d"] == 4
    assert r["n_no_answer_14d"] == 2  # d1, d3 (dLate excluded)
    out_t2 = build_features(T2, make_fixture())
    r2 = row_for(out_t2, L1, B1, "CP_A")
    # at T2 the late event is visible (received 08-20 <= T2)
    assert r2["n_attempts_14d"] == 5
    assert r2["n_no_answer_14d"] == 3


# 3. Duplicates ------------------------------------------------------------------
def test_duplicate_event_ids_do_not_double_count():
    out = build_features(T, make_fixture())
    r = row_for(out, L1, B1, "CP_A")
    # duplicate d1 (later received_at, conflicting answered payload) ignored
    assert r["n_attempts_1d"] == 2
    assert r["n_answered_1d"] == 0
    assert r["last_response_type"] == "switched_off"


# 4. Micro-fixture hand-computed values --------------------------------------------
def test_micro_fixture_window_slot_and_days_since():
    out = build_features(T, make_fixture())
    a = row_for(out, L1, B1, "CP_A")
    # windows (d0 08-08 busy, d1 08-14 no_answer, d2 08-14 switched_off,
    #          d3 08-10 no_answer, d4 07-01 no_answer; dLate excluded)
    assert a["n_attempts_1d"] == 2
    assert a["answer_rate_1d"] == 0.0
    assert a["n_no_answer_1d"] == 1
    assert a["n_switched_off_1d"] == 1
    assert a["n_attempts_3d"] == 2  # d3 (08-10) outside 3d
    assert a["n_attempts_7d"] == 4  # + d0 (08-08), + d3 (08-10)
    assert a["n_no_answer_7d"] == 2  # d1, d3
    assert a["n_busy_7d"] == 1  # d0
    assert a["n_attempts_14d"] == 4
    assert a["n_attempts_30d"] == 4  # d4 (07-01) outside 30d
    assert a["n_no_answer_30d"] == 2
    # slots (IST = UTC+5:30; d0 10:30 morn, d1 09:30 morn, d2 13:30 aft,
    #        d3 11:30 morn, d4 11:30 morn)
    assert a["n_attempts_morning_1d"] == 1
    assert a["n_attempts_afternoon_1d"] == 1
    assert a["n_attempts_evening_1d"] == 0
    assert a["answer_rate_morning_1d"] == 0.0
    assert pd.isna(a["answer_rate_evening_1d"])
    assert a["n_attempts_morning_30d"] == 3  # d0, d1, d3 (d4 outside 30d)
    assert a["n_attempts_afternoon_30d"] == 1  # d2
    # weekend: only d0 (Sat 08-08) is on a weekend
    assert a["weekend_attempt_share_7d"] == pytest.approx(1 / 4)
    assert a["weekend_attempt_share_30d"] == pytest.approx(1 / 4)
    # ring (threshold 3.0s from configs)
    assert a["ring_seconds_mean_1d"] == pytest.approx(15.25)
    assert a["ring_seconds_std_1d"] == pytest.approx(20.859, abs=1e-3)
    assert a["short_ring_rate_1d"] == pytest.approx(0.5)
    assert a["ring_seconds_mean_30d"] == pytest.approx(8.125)
    assert a["ring_seconds_std_30d"] == pytest.approx(14.585, abs=1e-3)
    assert a["short_ring_rate_30d"] == pytest.approx(0.75)
    # scalars
    assert a["last_response_type"] == "switched_off"
    assert a["consecutive_failures"] == 5  # never answered
    assert a["consecutive_same_response"] == 1
    assert a["days_since_first_attempt"] == 45  # 07-01 -> 08-15
    assert a["days_since_last_attempt"] == 1
    assert pd.isna(a["days_since_last_answer"])
    assert pd.isna(a["days_since_last_rpc"])
    assert a["mean_gap_between_attempts_days"] == pytest.approx(11.0208, abs=1e-3)
    # system fail share on 08-14 (IST): d1,d2,d5,d7,d8 -> 3 fail / 5
    assert a["system_fail_rate_on_last_attempt_day"] == pytest.approx(0.6)
    # dispositions + cues
    assert a["n_wrong_number"] == 1
    assert a["n_third_party"] == 1
    assert a["n_rpc"] == 0
    assert a["wrong_number_rate"] == pytest.approx(1 / 3)
    assert a["last_disposition"] == "switched_off"
    assert a["days_since_last_disposition"] == 2
    assert a["remark_wrongnumber_cue_count"] == 1
    assert a["remark_thirdparty_cue_count"] == 1
    assert a["remark_switchedoff_cue_count"] == 1
    assert a["remark_avoidance_cue_count"] == 0
    assert a["switched_off_months_max"] == 2
    # record history (update u1 overrides table source KYC)
    assert a["source"] == "skip_trace"
    assert a["record_age_days"] == 226  # 01-01 -> 08-15
    assert a["n_updates"] == 1
    assert a["days_since_last_update"] == 14
    assert bool(a["confirmed_by_payment"]) is False
    assert pd.isna(a["days_since_confirmed"])
    # agent reliability: last disp agent AG2 (1 disp, 0 wrong) ; AG1 1/4
    assert a["agent_wrong_number_rate"] == pytest.approx(0.0)
    assert a["account_id"] == "ACC_0001"
    assert a["dpd_bucket"] == "31-60"
    assert a["is_holiday"] == True  # noqa: E712 (2026-08-15 in configs)


def test_micro_fixture_cross_line_and_confirmations():
    out = build_features(T, make_fixture())
    a = row_for(out, L1, B1, "CP_A")
    b = row_for(out, L1, B1, "CP_B")
    # other lines for A = B's dials (d5 answered 08-14, d6 busy 08-13)
    assert a["other_lines_attempts_1d"] == 1
    assert a["other_lines_answered_1d"] == 1
    assert a["other_lines_answer_rate_1d"] == pytest.approx(1.0)
    assert a["other_lines_attempts_7d"] == 2
    assert a["other_lines_answer_rate_7d"] == pytest.approx(0.5)
    assert a["days_since_last_other_line_answer"] == 1
    # mirror: other lines for B = A's dials, all unanswered
    assert b["other_lines_attempts_1d"] == 2
    assert b["other_lines_answered_1d"] == 0
    assert b["other_lines_answer_rate_1d"] == pytest.approx(0.0)
    assert pd.isna(b["days_since_last_other_line_answer"])  # A never answered
    # B answered itself: streaks reset, answer recency set
    assert b["consecutive_failures"] == 0
    assert b["last_response_type"] == "answered"
    assert b["days_since_last_answer"] == 1
    assert b["days_since_last_rpc"] == 1
    assert b["n_attempts_1d"] == 1
    assert b["answer_rate_1d"] == pytest.approx(1.0)
    assert b["n_bot_calls"] == 2
    assert b["n_bot_who_borrower"] == 1
    assert b["n_bot_who_other"] == 1
    assert b["n_bot_who_unknown"] == 0
    assert b["n_bot_whoisthis_cue"] == 1
    assert b["n_bot_name_mismatch"] == 0
    assert b["last_who_answered"] == "other"
    assert b["remark_avoidance_cue_count"] == 1
    # payment p1 confirms B (within 7d after answered d5 / RPC x4), not A
    assert bool(b["confirmed_by_payment"]) is True
    assert b["days_since_confirmed"] == 1
    assert b["n_payments_1d"] == 1
    assert b["n_payments_30d"] == 1
    assert b["days_since_last_payment"] == 1
    assert a["n_payments_1d"] == 1  # borrower-level payments are shared
    # agent: B's last disp agent AG1 -> 1 wrong / 4 disps
    assert b["agent_wrong_number_rate"] == pytest.approx(0.25)


def test_micro_fixture_address_and_shared():
    out = build_features(T, make_fixture())
    c = row_for(out, L1, B1, "CP_C")
    assert c["contact_point_type"] == "address"
    assert c["n_visits"] == 2
    assert c["n_visits_locked_premises"] == 1
    assert c["n_visits_met_borrower"] == 1
    assert c["n_visits_nobody_of_that_name"] == 0
    assert c["last_visit_outcome"] == "locked_premises"
    assert c["days_since_last_visit"] == 3
    assert c["gps_dwell_mean_seconds"] == pytest.approx(210.0)
    assert c["visit_hour_mean"] == pytest.approx(10.5)
    # address rows still get cross-line telephony of the borrower's phones
    assert c["other_lines_attempts_30d"] == 6  # A:4 + B:2
    assert c["other_lines_answered_30d"] == 1
    # shared: B1-SH row linked to B2, L2 SH isolated
    sh1 = row_for(out, L1, B1, "SH")
    sh2 = row_for(out, L1, B2, "SH")
    sh9 = row_for(out, "LENDER_002", "BORR_0009", "SH")
    assert sh1["is_shared"] == True  # noqa: E712
    assert sh1["n_borrowers_sharing_cp"] == 2
    assert sh1["n_accounts_sharing_cp"] == 2
    assert sh2["is_shared"] == True  # noqa: E712
    assert sh9["is_shared"] == False  # noqa: E712
    assert sh9["n_borrowers_sharing_cp"] == 1
    assert sh1["connected_component_size"] == 2
    assert sh2["connected_component_size"] == 2
    assert sh9["connected_component_size"] == 1
    # borrower phone inventory + ranks (CPA 1, SH 2, CPB 3, CPC 4, CPN 5)
    assert sh1["n_phone_cps_for_borrower"] == 4
    assert row_for(out, L1, B1, "CP_A")["cp_rank_within_borrower"] == 1
    assert sh1["cp_rank_within_borrower"] == 2
    assert row_for(out, L1, B1, "CP_N")["cp_rank_within_borrower"] == 5
    # phone rows carry null (not 0) field features
    ph = row_for(out, L1, B1, "CP_A")
    for col in ("n_visits", "n_visits_locked_premises", "last_visit_outcome",
                "days_since_last_visit", "gps_dwell_mean_seconds", "visit_hour_mean"):
        assert pd.isna(ph[col]), col


# 5. Never-attempted --------------------------------------------------------------
def test_never_attempted_row_semantics():
    out = build_features(T, make_fixture())
    n = row_for(out, L1, B1, "CP_N")
    assert bool(n["has_any_attempt"]) is False
    assert n["n_attempts_30d"] == 0
    assert n["n_answered_30d"] == 0
    assert pd.isna(n["answer_rate_30d"])
    assert pd.isna(n["hangup_rate_30d"])
    assert pd.isna(n["answer_rate_morning_30d"])
    assert pd.isna(n["weekend_attempt_share_30d"])
    assert pd.isna(n["ring_seconds_mean_30d"])
    assert pd.isna(n["last_response_type"])
    assert pd.isna(n["consecutive_failures"])
    assert pd.isna(n["consecutive_same_response"])
    assert pd.isna(n["days_since_last_attempt"])
    assert pd.isna(n["days_since_first_attempt"])
    assert pd.isna(n["mean_gap_between_attempts_days"])
    assert pd.isna(n["system_fail_rate_on_last_attempt_day"])
    # cross-line still defined (borrower's other lines exist)
    assert n["other_lines_attempts_30d"] == 6
    assert n["other_lines_answer_rate_30d"] == pytest.approx(1 / 6)


# 6. Cross-line is covered in test_micro_fixture_cross_line_and_confirmations;
#    this pins the silent-vs-answered direction explicitly.
def test_cross_line_direction():
    out = build_features(T, make_fixture())
    a = row_for(out, L1, B1, "CP_A")  # silent
    b = row_for(out, L1, B1, "CP_B")  # answered
    assert a["other_lines_answered_7d"] > 0  # B answered while A silent
    assert b["other_lines_answered_7d"] == 0  # B's others exclude B itself


# 7. Shared contacts are lender-local -----------------------------------------------
def test_shared_contacts_lender_local():
    out = build_features(T, make_fixture())
    sh9 = row_for(out, "LENDER_002", "BORR_0009", "SH")
    assert sh9["n_borrowers_sharing_cp"] == 1
    assert sh9["connected_component_size"] == 1


# 8. No ground truth ------------------------------------------------------------------
def test_no_ground_truth_in_output():
    out = build_features(T, make_fixture())
    banned = set(load_feature_config()["banned_columns"]) | set(HIDDEN_STATE_TOKENS)
    cols = set(out.columns)
    assert not (banned & {c.lower() for c in cols}), banned & {c.lower() for c in cols}
    for token in HIDDEN_STATE_TOKENS:
        assert not any(token in c for c in out.columns), token
    # string-valued outputs carry only observable vocab, never hidden states
    for col in ("last_response_type", "last_disposition", "last_who_answered",
                "last_visit_outcome", "source"):
        vals = set(out[col].dropna().astype(str).str.lower().tolist())
        for token in ("valid_reachable", "avoiding", "temp_unreachable",
                      "switched_off_long", "recycled", "invalid"):
            assert token not in vals, (col, token)
    # the pattern file itself never matches on hidden-state names
    raw = json.dumps(load_patterns(), default=str).lower()
    for token in ("valid_reachable", "avoiding", "temp_unreachable",
                  "switched_off_long", "recycled", "true_state", "ground_truth"):
        assert token not in raw, token
    # feature code never imports the label module (one-way dependency):
    # static scan of every feature-code file (labels.py itself excluded).
    import pathlib
    import re
    repo = pathlib.Path(__file__).resolve().parents[1]
    import_re = re.compile(r"^\s*(import|from)\s+[\w.]*labels\b", re.MULTILINE)
    for mod in ("features.py", "source.py", "spec.py", "text.py", "build.py", "__init__.py"):
        src = (repo / "src" / "rpc" / "features" / mod).read_text()
        assert not import_re.search(src), mod


# 9. Text extractor ----------------------------------------------------------------------
@pytest.mark.parametrize(("phrase", "months"), [
    ("number band hai 2 mahine se", 2),
    ("Number Band Hai Do Mahine Se", 2),
    ("nambar bnd he 3 mahina se", 3),
    ("phone band hai 1 saal se", 12),
    ("mobile switched off hai 4 months se", 4),
    ("no answer aa raha hai", None),
])
def test_text_extractor_switched_off_months(phrase, months):
    assert extract_switched_off_months_text(phrase) == months


@pytest.mark.parametrize(("phrase", "group"), [
    ("number band hai 2 mahine se", "switched_off_cue"),
    ("PHONE BAND HAI", "switched_off_cue"),
    ("wrong number hai", "wrong_number_cue"),
    ("Rong number bata raha hai", "wrong_number_cue"),
    ("galat number lag gaya", "wrong_number_cue"),
    ("phone uthaya bhai ne", "third_party_cue"),
    ("BHAI NE PHONE UTHAYA", "third_party_cue"),
    ("baad me call karo", "avoidance_cue"),
    ("Call Mat Karo dobara", "avoidance_cue"),
    ("kaun bol raha hai", "who_is_this_cue"),
    ("Kaun Hai, aap kaun", "who_is_this_cue"),
    ("naam galat hai is number par", "name_mismatch_cue"),
    ("hindi nahi samajh aa rahi", "language_mismatch_cue"),
])
def test_text_extractor_cues(phrase, group):
    from src.rpc.features.text import _compiled
    assert _compiled(group).search(phrase), (phrase, group)


# 10. Determinism and schema ---------------------------------------------------------------
def test_determinism_schema_and_snapshot():
    src = make_fixture()
    o1 = build_features(T, src)
    o2 = build_features(T, make_fixture())
    assert_frame_equal(o1, o2, check_dtype=True)
    assert list(o1.dtypes) == list(o2.dtypes)
    assert (o1["feature_snapshot_id"] == o2["feature_snapshot_id"]).all()
    assert (o1["event_watermark"] == o2["event_watermark"]).all()
    # snapshot id changes with as_of ...
    o3 = build_features(T2, make_fixture())
    assert o3["feature_snapshot_id"].iloc[0] != o1["feature_snapshot_id"].iloc[0]
    # ... and with config content
    assert snapshot_id("cfg-a", T, str(o1["event_watermark"].iloc[0])) != \
        snapshot_id("cfg-b", T, str(o1["event_watermark"].iloc[0]))
    # same code path: single-date training table equals build_features
    tt = build_training_table([T], make_fixture())
    assert_frame_equal(tt, o1, check_dtype=True)
    # stable key order + metadata present
    assert list(o1.columns[:5]) == KEY_COLUMNS
    assert list(o1.columns[-2:]) == META_COLUMNS


# 11. Registry matches output ---------------------------------------------------------------
def test_registry_matches_output_columns():
    src = make_fixture()
    out = build_features(T, src)
    include_agent = bool(out.attrs.get("include_agent_feature"))
    expected = feature_names(include_agent_feature=include_agent)
    got = [c for c in out.columns if c not in KEY_COLUMNS + META_COLUMNS]
    assert got == expected
    # dtypes declared in the registry hold in the output
    reg = {f.name: f.dtype for f in build_registry(include_agent_feature=include_agent)}
    for col in got:
        assert str(out[col].dtype) == reg[col], col


def test_labels_module_basic():
    src = make_fixture()
    lab = build_labels(T, src)
    # no RPC dispositions in (T, T+7d] in the fixture -> all False, structure holds
    assert list(lab.columns) == ["lender_id", "borrower_id", "contact_point_ref",
                                 "as_of", "rpc_next_7d", "was_dialled_next_7d"]
    # future RPC labels the right row once a future disposition exists
    extra = [_ev("xF", "disposition", L1, B1, "ACC_0001", "CP_A", "2026-08-16T10:00:00+00:00",
                 "2026-08-16T10:05:00+00:00", {"disposition": "RPC"})]
    lab2 = build_labels(T, make_fixture(extra))
    hit = lab2[(lab2["borrower_id"] == B1) & (lab2["contact_point_ref"] == "CP_A")]
    assert bool(hit["rpc_next_7d"].iloc[0]) is True
