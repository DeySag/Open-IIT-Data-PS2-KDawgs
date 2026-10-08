"""P7 decision tests: per-lender costs, self-cure haircut, fast-path real wiring.

Fast-path cases mirror real issued-extract rows (wrong_number / third_party
dispositions from dial_attempts.csv) through the real contracts envelope,
detector, and suppression store: suppression is added, idempotent on replay,
and reversible only via a CN-signoff removal request (entry stays in force).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.rpc.contracts import Disposition, DispositionPayload, EventType, InputEvent, ReasonCode
from src.rpc.decision.reason_codes import DecisionReason, REASON_CODE_INFO, to_contract_reason
from src.rpc.decision.voi import (
    compute_voi,
    load_costs_config,
    resolve_costs_for_lender,
    scale_recovery_gain,
    trace_cost_for,
)
from src.rpc.serve.fast_path import RecycledSignalDetector
from src.rpc.serve.suppression import SuppressionStore

UTC = timezone.utc


def _disp_event(ref: str, disposition: Disposition, lender: str = "L01") -> InputEvent:
    return InputEvent(
        event_id=uuid4(),
        event_type=EventType.DISPOSITION,
        lender_id=lender,
        borrower_id="AC000001",
        account_id="AC000001",
        contact_point_ref=ref,
        occurred_at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        received_at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        payload=DispositionPayload(disposition=disposition),
    )


# --- per-lender costs (TRAIN-observed means in costs.yaml) ---

def test_per_lender_costs_resolve_and_fallback():
    costs = load_costs_config()
    assert trace_cost_for("digital", resolve_costs_for_lender(costs, "L01")) == 103.7
    assert trace_cost_for("digital", resolve_costs_for_lender(costs, "L04")) == 92.4
    # L05 had too few TRAIN traces -> global fallback; unknown labels too.
    assert trace_cost_for("digital", resolve_costs_for_lender(costs, "L05")) == 104.4
    assert trace_cost_for("digital", resolve_costs_for_lender(costs, "L99")) == 104.4
    # Explicit override still wins; input never mutated.
    assert trace_cost_for("digital", resolve_costs_for_lender(costs, "L01"), override=63.0) == 63.0
    assert trace_cost_for("digital", costs) == 100.0


# --- self-cure gain haircut ---

def test_selfcure_haircut_scales_gain_not_costs():
    costs = load_costs_config()
    scaled = scale_recovery_gain(costs, 0.34)
    assert scaled["recovery_curves"]["by_bucket"]["31-60"] == {"reached": 0.085, "not_reached": 0.0102}
    assert costs["recovery_curves"]["by_bucket"]["31-60"] == {"reached": 0.25, "not_reached": 0.03}
    assert scale_recovery_gain(costs, 1.0)["recovery_curves"] == costs["recovery_curves"]
    base = compute_voi(0.8, "31-60", "unsecured_retail", False, 100000.0, "digital", costs)
    cut = compute_voi(0.8, "31-60", "unsecured_retail", False, 100000.0, "digital", scaled)
    assert cut["voi"] < base["voi"]
    assert cut["trace_cost"] == base["trace_cost"]  # costs untouched


# --- fast-path real wiring: suppress, idempotent, reversible-only-via-request ---

def test_fast_path_wrong_number_suppresses_real_shape():
    det = RecycledSignalDetector(None, type("C", (), {"recycled_risk_threshold": 0.5})())
    store = SuppressionStore("p7-test")
    res = det.detect(_disp_event("cp_real_1", Disposition.WRONG_NUMBER))
    assert [r.reason for r in res] == ["recycled"]
    entry = store.add(res[0].contact_point_ref, "L01", res[0].reason, res[0].evidence)
    assert entry is not None and store.is_suppressed("cp_real_1", "L01")
    # Idempotent replay of the same evidence adds nothing.
    again = store.add(res[0].contact_point_ref, "L01", res[0].reason, res[0].evidence)
    assert again is None


def test_fast_path_third_party_and_reversible_only_via_request():
    det = RecycledSignalDetector(None, type("C", (), {"recycled_risk_threshold": 0.5})())
    store = SuppressionStore("p7-test")
    res = det.detect(_disp_event("cp_real_2", Disposition.THIRD_PARTY))
    assert [r.reason for r in res] == ["third_party"]
    store.add(res[0].contact_point_ref, "L01", res[0].reason, res[0].evidence)
    req = store.request_removal("cp_real_2", "L01", reason="reviewed-valid", requester="ops")
    assert req["status"] == "pending_cn_signoff"
    assert store.is_suppressed("cp_real_2", "L01")  # stays in force until CN signs off


def test_fast_path_ignores_non_signal_disposition():
    det = RecycledSignalDetector(None, type("C", (), {"recycled_risk_threshold": 0.5})())
    assert det.detect(_disp_event("cp_ok", Disposition.RPC)) == []


# --- reason codes stable across versions ---

def test_reason_codes_stable_and_contract_coerced():
    frozen = {"VALID_CONTINUE", "TEMP_UNREACHABLE_BACKOFF", "AVOIDING_SWITCH_CHANNEL",
              "SWITCHED_OFF_MOVE_OR_TRACE", "INVALID_TRACE", "RECYCLED_SUPPRESS",
              "THIRD_PARTY_RESTRICT", "ADDRESS_ABSENT_CHANGE_TIME", "ADDRESS_MOVED_TRACE",
              "ADDRESS_FABRICATED_TRACE_FLAG", "ADDRESS_UNRESOLVED_REVIEW"}
    assert frozen <= set(REASON_CODE_INFO)  # drivers documented for every code
    for name in frozen:
        assert to_contract_reason(DecisionReason(name)) == ReasonCode(name)  # 1:1, never renamed
    for ext in ("GUARDRAIL_NO_CONSENT", "DEFERRED_TRACE_WAIT", "LOW_VOI_DEPRIORITISED"):
        assert isinstance(to_contract_reason(ext), ReasonCode)  # extensions coerce, never crash
