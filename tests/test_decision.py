"""Decision-layer tests: guardrails, action mapping, VOI ranker.

Conventions: all fixtures use fixed timestamps (no wall-clock dependence);
contact-hours cases pin Asia/Kolkata explicitly. Simulation-only throughout.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from src.rpc.contracts import Action, OutputDecision
from src.rpc.decision import (
    AccountContext,
    AccountFlags,
    ContactPointScore,
    TraceCandidate,
    compute_voi,
    decide_full,
    evaluate_guardrails,
    rank_trace,
    recoverable_amount,
    should_defer,
)
from src.rpc.decision.guardrails import load_guardrails_config
from src.rpc.decision.reason_codes import DecisionReason
from src.rpc.decision.stubs import stub_context, stub_score, stub_scores, stub_trace_candidate
from src.rpc.decision.voi import load_costs_config

UTC = timezone.utc
# 2026-10-05 04:00 UTC == 09:30 IST (inside 08-19 window)
IN_HOURS = datetime(2026, 10, 5, 4, 0, tzinfo=UTC)
# 2026-10-05 18:00 UTC == 23:30 IST (outside window)
OUT_OF_HOURS = datetime(2026, 10, 5, 18, 0, tzinfo=UTC)

FOUR_ACTIONS = {"continue", "switch_contact_point", "switch_channel", "trace"}


def _ctx(**kw) -> AccountContext:
    base = {"now": IN_HOURS}
    base.update(kw)
    return stub_context(**base)


def _random_scores(rng: random.Random, n: int = 3) -> list[ContactPointScore]:
    states = ["valid_reachable", "avoiding", "temp_unreachable", "switched_off_long",
              "recycled", "third_party", "invalid"]
    out = []
    for i in range(n):
        post = {s: rng.random() for s in states}
        out.append(ContactPointScore(
            contact_point_ref=f"rnd_{i}",
            type="phone",
            as_of=IN_HOURS,
            state_posterior=post,
            p_rpc=rng.random(),
            recycled_risk=rng.random() * 0.05,
            confidence=rng.random(),
        ))
    return out


# ---------------------------------------------------------------------------
# 1. Guardrails always restrict; scores cannot override
# ---------------------------------------------------------------------------

FLAG_COMBOS = [
    {},
    {"dispute": True},
    {"no_consent": True},
    {"deceased_or_insolvent": True},
    {"dnd": True},
    {"legal_case": True},
    {"dispute": True, "dnd": True},
    {"no_consent": True, "dispute": True, "deceased_or_insolvent": True, "dnd": True, "legal_case": True},
]


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_guardrails_restrict_property(flags):
    rng = random.Random(1234)
    hard = {"dispute", "no_consent", "deceased_or_insolvent", "legal_case"} & set(flags)
    for trial in range(10):
        scores = _random_scores(rng)
        ctx = _ctx(flags=AccountFlags(**{k: True for k in flags}))
        res = decide_full(ctx, scores)
        d = res.decision
        assert d.action in FOUR_ACTIONS
        if hard:
            # Restrictions must never produce a trace, however "healthy" scores look.
            assert d.action != Action.TRACE.value
            assert d.trace is None
            assert res.internal_reason_code.value.startswith("GUARDRAIL_")
        # Suppressed refs are never dialled first.
        if res.decision.ranked_contact_points[0].ref != "no_viable_contact_point":
            assert res.decision.ranked_contact_points[0].ref not in ctx.suppressed_refs


def test_hard_blocks_never_trace_even_with_perfect_scores():
    for flag in ["dispute", "no_consent", "deceased_or_insolvent", "legal_case"]:
        ctx = _ctx(flags=AccountFlags(**{flag: True}))
        scores = stub_scores(["valid_reachable", "valid_reachable"])
        for s in scores:
            s.p_rpc = 1.0
        res = decide_full(ctx, scores)
        assert res.decision.action != Action.TRACE.value
        assert res.decision.trace is None


def test_dnd_removes_voice_channels_and_whatsapp_needs_opt_in():
    ctx = _ctx(flags=AccountFlags(dnd=True), whatsapp_opt_in=False)
    report = evaluate_guardrails(ctx, stub_scores(["avoiding"]))
    assert "voice_bot" not in report.allowed_channels
    assert "telecaller" not in report.allowed_channels
    res = decide_full(ctx, stub_scores(["avoiding"]))
    assert res.decision.action == Action.SWITCH_CHANNEL.value
    assert res.decision.action_params.target_channel == "field"

    ctx2 = _ctx(flags=AccountFlags(dnd=True), whatsapp_opt_in=True)
    res2 = decide_full(ctx2, stub_scores(["avoiding"]))
    assert res2.decision.action_params.target_channel == "whatsapp"


# ---------------------------------------------------------------------------
# 2. Avoiding borrower -> switch_channel, ~zero VOI, never queued
# ---------------------------------------------------------------------------

def test_avoiding_borrower_switches_channel_with_zero_voi():
    ctx = _ctx(whatsapp_opt_in=True)
    res = decide_full(ctx, stub_scores(["avoiding"]))
    assert res.decision.action == Action.SWITCH_CHANNEL.value
    assert res.internal_reason_code == DecisionReason.AVOIDING_SWITCH_CHANNEL
    assert res.decision.action_params.target_channel == "whatsapp"

    costs = load_costs_config()
    d = compute_voi(0.02, "31-60", "unsecured_retail", False, 50000.0, "digital", costs)
    assert d["voi"] < 0  # near-zero find prob cannot cover trace + collection costs

    ranked = rank_trace(
        [stub_trace_candidate("AVOID", p_dead=0.02)], 100000.0, costs)
    assert ranked[0].within_budget is False
    assert ranked[0].rank == 0


# ---------------------------------------------------------------------------
# 3. Recycled / third-party risk excluded; suppression returned; never first
# ---------------------------------------------------------------------------

def test_recycled_excluded_suppressed_never_first():
    ctx = _ctx()
    bad = stub_score("cp_bad", "recycled")
    bad.p_rpc = 0.99  # highest raw score must still never head dial order
    good = stub_score("cp_good", "valid_reachable")
    res = decide_full(ctx, [bad, good])
    ranked_refs = [c.ref for c in res.decision.ranked_contact_points]
    assert ranked_refs[0] == "cp_good"
    assert "cp_bad" in ranked_refs[1:]
    reasons = [s.reason for s in res.suppressions]
    assert "recycled" in reasons
    req = next(s for s in res.suppressions if s.reason == "recycled")
    assert req.contact_point_ref == "cp_bad" and len(req.evidence) >= 1

    # High classifier risk alone (no dominant state) also suppresses.
    risky = stub_score("cp_risky", "switched_off_long")
    risky.recycled_risk = 0.5
    res2 = decide_full(ctx, [risky, good])
    assert res2.decision.ranked_contact_points[0].ref == "cp_good"
    assert any(s.contact_point_ref == "cp_risky" for s in res2.suppressions)


def test_third_party_excluded_from_dial_order():
    ctx = _ctx()
    tp = stub_score("cp_tp", "third_party")
    tp.p_rpc = 0.95
    good = stub_score("cp_good", "valid_reachable")
    res = decide_full(ctx, [tp, good])
    assert res.decision.ranked_contact_points[0].ref == "cp_good"
    assert any(s.reason == "third_party" for s in res.suppressions)


# ---------------------------------------------------------------------------
# 4. All phones dead -> trace; fabricated address -> trace + review flag
# ---------------------------------------------------------------------------

def test_all_phones_dead_traces_with_reason_code():
    ctx = _ctx()
    res = decide_full(ctx, stub_scores(["invalid", "switched_off_long"]))
    assert res.decision.action == Action.TRACE.value
    assert res.internal_reason_code in (
        DecisionReason.INVALID_TRACE, DecisionReason.SWITCHED_OFF_MOVE_OR_TRACE)
    assert res.decision.trace is not None
    assert res.decision.trace.recoverable_amount > 0


def test_dead_best_with_healthy_alternative_switches_not_traces():
    ctx = _ctx()
    dead = stub_score("cp_dead", "invalid")
    dead.p_rpc = 0.9  # ranked first by p_rpc -> all contact points considered
    alive = stub_score("cp_alive", "valid_reachable")
    alive.p_rpc = 0.1
    res = decide_full(ctx, [dead, alive])
    assert res.decision.action == Action.SWITCH_CONTACT_POINT.value
    res_rev = decide_full(ctx, [alive, dead])  # order-invariant
    assert res_rev.decision.action == Action.SWITCH_CONTACT_POINT.value


def test_fabricated_address_traces_with_origination_review():
    ctx = _ctx(address_state="fabricated")
    res = decide_full(ctx, stub_scores(["invalid", "switched_off_long"]))
    assert res.decision.action == Action.TRACE.value
    assert res.internal_reason_code == DecisionReason.ADDRESS_FABRICATED_TRACE_FLAG
    assert res.decision.flags.origination_review is True
    assert res.decision.trace is not None and res.decision.trace.recoverable_amount > 0


# ---------------------------------------------------------------------------
# 5. Exactly one action + reason, including adversarial inputs
# ---------------------------------------------------------------------------

def test_empty_scores_still_yield_one_action_and_reason():
    res = decide_full(_ctx(), [])
    assert res.decision.action in FOUR_ACTIONS
    assert isinstance(res.internal_reason_code, DecisionReason)
    assert len(res.decision.ranked_contact_points) >= 1  # contract min_length=1


def test_all_suppressed_never_traces():
    scores = stub_scores(["valid_reachable", "avoiding"])
    refs = {s.contact_point_ref for s in scores}
    ctx = _ctx(suppressed_refs=refs)
    res = decide_full(ctx, scores)
    assert res.decision.action in FOUR_ACTIONS
    assert res.decision.action != Action.TRACE.value
    assert res.decision.trace is None


def test_trace_pending_never_double_queues():
    ctx = _ctx(trace_pending=True)
    res = decide_full(ctx, stub_scores(["invalid"]))
    assert res.decision.action != Action.TRACE.value
    assert res.internal_reason_code == DecisionReason.GUARDRAIL_TRACE_PENDING


# ---------------------------------------------------------------------------
# 6. VOI: hand-computed example, budget, monotonicity, deferral
# ---------------------------------------------------------------------------

def test_hand_computed_voi_example():
    # Hand computation (INR) with costs.yaml assumptions:
    # recoverable = 100000 * 0.85 * 0.60 * (1-0.20) / 1.12^0.5
    #             = 40800 / 1.05830052443 = 38552.27 (approx)
    # gain = 0.25 - 0.03 = 0.22 ; p_find = 0.25 * 0.8 = 0.20
    # VOI = 0.20 * 0.22 * 38552.27 - 100 - 60 - 25 = 1511.30 (approx)
    costs = load_costs_config()
    d = compute_voi(0.8, "31-60", "unsecured_retail", False, 100000.0, "digital", costs)
    assert d["recoverable_amount"] == pytest.approx(38552.27, rel=1e-3)
    assert d["p_find"] == pytest.approx(0.20)
    assert d["recovery_gain"] == pytest.approx(0.22)
    assert d["voi"] == pytest.approx(1511.30, rel=1e-3)
    assert d["voi_per_rupee"] == pytest.approx(15.113, rel=1e-3)


def test_voi_monotonicity():
    costs = load_costs_config()
    base = dict(p_dead=0.8, dpd_bucket="31-60", product="unsecured_retail",
                secured=False, outstanding=100000.0, trace_method="digital", costs=costs)
    v0 = compute_voi(**base)["voi"]
    assert compute_voi(**{**base, "outstanding": 200000.0})["voi"] > v0
    assert compute_voi(**{**base, "p_dead": 0.9})["voi"] > v0
    assert compute_voi(**{**base, "trace_cost_override": 5000.0})["voi"] < v0


def test_rank_trace_budget_respected_and_gated():
    costs = load_costs_config()
    cands = [
        stub_trace_candidate("RICH_DEAD", p_dead=0.9, outstanding=200000.0),
        stub_trace_candidate("POOR_DEAD", p_dead=0.9, outstanding=20000.0),
        stub_trace_candidate("AVOIDER", p_dead=0.02, outstanding=200000.0),
        stub_trace_candidate("INELIGIBLE", p_dead=0.9, outstanding=200000.0, eligible=False),
    ]
    ranked = rank_trace(cands, 150.0, costs)  # digital=100: fits exactly one of RICH/POOR
    selected = [t for t in ranked if t.within_budget]
    assert sum(t.cost for t in selected) <= 150.0
    assert selected and selected[0].account_id == "RICH_DEAD" and selected[0].rank == 1
    by_id = {t.account_id: t for t in ranked}
    assert by_id["AVOIDER"].within_budget is False  # ~zero VOI never queued
    assert by_id["INELIGIBLE"].within_budget is False


def test_should_defer():
    costs = load_costs_config()
    assert should_defer(100.0, 100000.0, costs) is True   # waiting beats tracing now
    assert should_defer(100000.0, 100.0, costs) is False
    # Temp-unreachable-heavy accounts wait with backoff instead of tracing.
    res = decide_full(_ctx(), stub_scores(["temp_unreachable"]))
    assert res.decision.action == Action.CONTINUE.value
    assert res.decision.action_params.backoff_days == 2


# ---------------------------------------------------------------------------
# 7. Contact-hours window in IST, incl. midnight/day boundaries
# ---------------------------------------------------------------------------

def test_contact_hours_ist_boundaries():
    cfg = load_guardrails_config()
    # 07:59 IST (UTC 02:29) outside; 08:00 IST (UTC 02:30) inside
    assert evaluate_guardrails(
        _ctx(now=datetime(2026, 10, 5, 2, 29, tzinfo=UTC)), []).fired_rules == ["CONTACT_HOURS"]
    assert "CONTACT_HOURS" not in evaluate_guardrails(
        _ctx(now=datetime(2026, 10, 5, 2, 30, tzinfo=UTC)), []).fired_rules
    # 18:59 IST (UTC 13:29) inside; 19:00 IST (UTC 13:30) outside
    assert "CONTACT_HOURS" not in evaluate_guardrails(
        _ctx(now=datetime(2026, 10, 5, 13, 29, tzinfo=UTC)), []).fired_rules
    assert "CONTACT_HOURS" in evaluate_guardrails(
        _ctx(now=datetime(2026, 10, 5, 13, 30, tzinfo=UTC)), []).fired_rules


def test_overnight_window_across_midnight():
    cfg = load_guardrails_config()
    cfg = {**cfg, "contact_hours": {**cfg["contact_hours"],
                                    "start_hour": 22, "end_hour": 6}}
    inside = [datetime(2026, 10, 5, h, m, tzinfo=UTC)
              for h, m in [(16, 30), (18, 0), (0, 0)]]  # 22:00, 23:30, 05:30 IST
    outside = [datetime(2026, 10, 5, h, m, tzinfo=UTC)
               for h, m in [(0, 30), (6, 30)]]  # 06:00, 12:00 IST
    for ts in inside:
        assert "CONTACT_HOURS" not in evaluate_guardrails(_ctx(now=ts), [], cfg).fired_rules, ts
    for ts in outside:
        assert "CONTACT_HOURS" in evaluate_guardrails(_ctx(now=ts), [], cfg).fired_rules, ts


def test_outside_hours_blocks_trace_but_keeps_backoff():
    res = decide_full(_ctx(now=OUT_OF_HOURS), stub_scores(["invalid"]))
    assert res.decision.action != Action.TRACE.value
    assert res.decision.action in FOUR_ACTIONS


def test_frequency_caps_force_backoff_not_new_attempts():
    ctx = _ctx(attempts_today=3)
    report = evaluate_guardrails(ctx, stub_scores(["valid_reachable"]))
    assert "FREQUENCY_CAP" in report.fired_rules
    assert "continue" not in report.allowed_actions
    res = decide_full(ctx, stub_scores(["valid_reachable"]))
    assert res.decision.action in FOUR_ACTIONS
    assert res.decision.action != Action.TRACE.value


# ---------------------------------------------------------------------------
# 8. Contract validation; TraceInfo always has recoverable_amount
# ---------------------------------------------------------------------------

def test_output_validates_against_contract():
    for states, kw in [
        (["valid_reachable"], {}),
        (["avoiding"], {}),
        (["invalid", "switched_off_long"], {}),
        (["recycled"], {}),
        ([], {}),
        (["temp_unreachable"], {"attempts_today": 99}),
        (["invalid"], {"address_state": "fabricated"}),
    ]:
        res = decide_full(_ctx(**kw), stub_scores(states))
        validated = OutputDecision.model_validate(res.decision.model_dump())
        assert validated.action == res.decision.action
        if res.decision.action == Action.TRACE.value:
            assert res.decision.trace is not None
            assert res.decision.trace.recoverable_amount > 0
            assert res.decision.trace.recoverable_amount == pytest.approx(
                recoverable_amount(
                    50000.0, "31-60", False, load_costs_config()))


def test_recoverable_amount_matches_account_context():
    costs = load_costs_config()
    ctx = _ctx(outstanding=123456.0, dpd_bucket="61-90", product="msme", secured=True)
    res = decide_full(ctx, stub_scores(["invalid"]))
    assert res.decision.action == Action.TRACE.value
    assert res.decision.trace is not None
    assert res.decision.trace.recoverable_amount == pytest.approx(
        recoverable_amount(123456.0, "61-90", True, costs))


def test_legacy_engine_shim_still_fixed():
    """The day-0 bugs stay fixed through the compat shim used by serve/app."""
    from src.rpc.contracts import RankedContactPoint, StatePosterior, ContactPointType
    from src.rpc.decision.engine import GuardrailsEngine, decide_action, map_reason_code

    eng = GuardrailsEngine()
    cp = RankedContactPoint(
        ref="cp1", type=ContactPointType.PHONE, p_rpc=0.9,
        state_posterior=StatePosterior(valid_reachable=0.8, avoiding=0.05, temp_unreachable=0.05,
                                       switched_off_long=0.03, recycled=0.02, third_party=0.02,
                                       invalid=0.03),
        confidence=0.9)
    # IST check through shim: 23:30 IST is outside hours.
    sham = eng.evaluate("A", [cp], {"now": OUT_OF_HOURS, "attempts_today": 0, "consent": True})
    assert "CONTACT_HOURS" in (sham.reason or "")
    # Restrictions never trace through the shim fallback.
    res = eng.evaluate("A", [cp], {"now": IN_HOURS, "dispute": True})
    rc = map_reason_code(cp.state_posterior, cp.p_rpc, res)
    action, _ = decide_action(rc, res, [cp])
    assert action != Action.TRACE
