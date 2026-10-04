"""Per-account action mapping: ranked contact points -> exactly one action.

Order of evaluation: guardrails (restrict-only) -> risk exclusions
(suppression) -> avoidance/channel logic -> dead-line handling -> trace
eligibility + VOI gate + deferral -> contract OutputDecision.

A recycled signal never traces: it yields a suppression request and the
account moves to another contact point/channel (or a parked
switch_channel/field when nothing viable remains).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import yaml

from src.rpc.contracts import (
    Action,
    ActionParams,
    ContactPointType,
    Flags,
    OutputDecision,
    RankedContactPoint,
    StatePosterior,
    SuppressionEntry,
    TraceInfo,
)
from src.rpc.decision.guardrails import evaluate_guardrails
from src.rpc.decision.reason_codes import DecisionReason, to_contract_reason
from src.rpc.decision.types import AccountContext, ContactPointScore
from src.rpc.decision.voi import compute_voi, load_costs_config, should_defer

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GUARDRAILS_PATH = REPO_ROOT / "configs" / "guardrails.yaml"


def load_decision_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_GUARDRAILS_PATH) as f:
        cfg = yaml.safe_load(f)["guardrails"]
    return cfg.get("decision", {})


@dataclass
class DecisionResult:
    decision: OutputDecision
    suppressions: list[SuppressionEntry]
    internal_reason_code: DecisionReason
    fired_rules: list[str]
    excluded_refs: dict[str, str]


def _recycled_threshold(decision_cfg: dict, costs: dict) -> float:
    """Threshold from the explicit cost ratio (NOT 0.5).

    Missed-recycled errors cost ``cost_ratio`` x a false suppression, so the
    cost-optimal cutoff is P* = C_fp / (C_fp + C_fn) = 1 / (1 + ratio).
    With ratio 100 -> ~0.0099: suppress on even a weak credible signal.
    """
    ratio = float(costs.get("recycled_threshold", {}).get("cost_ratio_miss_vs_false_suppress", 100.0))
    derived = 1.0 / (1.0 + ratio)
    return float(decision_cfg.get("recycled_threshold", derived))


def _placeholder_ranked(now: datetime) -> RankedContactPoint:
    """Contract requires >=1 ranked contact point even with no viable line.
    Zero scores signal do-not-dial; the serving layer must not dial it."""
    return RankedContactPoint(
        ref="no_viable_contact_point",
        type=ContactPointType.PHONE,
        p_rpc=0.0,
        state_posterior=StatePosterior(
            valid_reachable=0.0, avoiding=0.0, temp_unreachable=0.0,
            switched_off_long=0.0, recycled=0.0, third_party=0.0, invalid=1.0,
        ),
        confidence=0.0,
    )


def _to_ranked(s: ContactPointScore) -> RankedContactPoint:
    ctype = ContactPointType.PHONE if s.type == "phone" else ContactPointType.ADDRESS
    return RankedContactPoint(
        ref=s.contact_point_ref,
        type=ctype,
        p_rpc=s.p_rpc,
        state_posterior=StatePosterior(**s.state_posterior),
        confidence=s.confidence,
    )


def _suppression_for(score: ContactPointScore, ctx: AccountContext, model_version: str,
                     reason: str) -> SuppressionEntry:
    evidence = list(score.evidence_ids) or [
        uuid5(NAMESPACE_URL, f"rpc:{score.contact_point_ref}")
    ]  # placeholder when the model ships no event ids; serving layer must
       # reconcile with real event ids (documented assumption).
    return SuppressionEntry(
        contact_point_ref=score.contact_point_ref,
        lender_id=ctx.lender_id,
        reason=reason,  # type: ignore[arg-type]
        evidence=evidence,
        added_at=ctx.now,
        model_version=model_version,
    )


def _pick_channel(ctx: AccountContext, allowed_channels: set[str]) -> str:
    """WhatsApp only with explicit opt-in; voice/telecaller honour DND."""
    if ctx.whatsapp_opt_in and "whatsapp" in allowed_channels:
        return "whatsapp"
    if "sms" in allowed_channels and not ctx.flags.dnd:
        return "sms"
    if "field" in allowed_channels:
        return "field"
    return sorted(allowed_channels)[0] if allowed_channels else "field"


def decide_full(
    ctx: AccountContext,
    scores: list[ContactPointScore],
    guard_cfg: dict | None = None,
    cost_cfg: dict | None = None,
) -> DecisionResult:
    from src.rpc.decision.guardrails import load_guardrails_config

    guardrails_raw = guard_cfg or load_guardrails_config()
    decision_cfg = guardrails_raw.get("decision", {}) if guard_cfg is None else guard_cfg.get("decision", guard_cfg)
    costs = cost_cfg or load_costs_config()
    model_version = str(decision_cfg.get("model_version", "decision-v0.1.0"))
    validity_hours = int(decision_cfg.get("validity_hours", 24))
    recycled_thr = _recycled_threshold(decision_cfg, costs)
    third_party_thr = float(decision_cfg.get("third_party_threshold", 0.5))
    avoidance_thr = float(decision_cfg.get("avoidance_threshold", 0.5))
    backoff_days = int(decision_cfg.get("temp_unreachable_backoff_days", 2))
    deferral_days = int(decision_cfg.get("deferral_days", 7))

    report = evaluate_guardrails(ctx, scores, guardrails_raw)
    allowed = set(report.allowed_actions)
    suppressions: list[SuppressionEntry] = []
    excluded: dict[str, str] = dict(report.excluded_refs)

    def parked(reason: DecisionReason) -> DecisionResult:
        return _final(
            ctx, scores, [], Action.SWITCH_CHANNEL, ActionParams(target_channel="field"),
            reason, suppressions, report, excluded, model_version, validity_hours,
            origination_review=False, trace_info=None,
        )

    # --- hard blocks: never outreach, never trace ---
    if report.hard_blocked:
        assert report.block_reason is not None
        return parked(report.block_reason)

    # --- risk exclusions over ALL contact points (never just the first) ---
    viable: list[ContactPointScore] = []
    for s in scores:
        if s.contact_point_ref in excluded:
            continue
        # Cost-ratio cutoff applies to the dedicated recycled-risk classifier
        # score (NOT to background posterior mass: a healthy line always
        # carries a little recycled uncertainty). A recycled-dominant state
        # posterior is independently a credible fast-path signal.
        if s.recycled_risk >= recycled_thr or s.dominant_state == "recycled":
            excluded[s.contact_point_ref] = DecisionReason.RECYCLED_SUPPRESS.value
            suppressions.append(_suppression_for(s, ctx, model_version, "recycled"))
            continue
        if s.third_party_mass >= third_party_thr:
            excluded[s.contact_point_ref] = DecisionReason.THIRD_PARTY_RESTRICT.value
            suppressions.append(_suppression_for(s, ctx, model_version, "third_party"))
            continue
        viable.append(s)

    # Rank by p_rpc, confidence breaks ties.
    viable.sort(key=lambda s: (s.p_rpc, s.confidence), reverse=True)
    ranked_all = [_to_ranked(s) for s in viable] + [
        _to_ranked(s) for s in scores if s.contact_point_ref in excluded
    ]

    def trace_path(p_dead: float, driver: DecisionReason) -> DecisionResult:
        """Shared trace gate: deferral -> VOI gate -> eligibility -> trace/park."""
        if ctx.trace_pending:
            return parked(DecisionReason.GUARDRAIL_TRACE_PENDING)
        if "trace" not in allowed:
            return parked(_guardrail_reason_for(report))
        d = compute_voi(p_dead, ctx.dpd_bucket, ctx.product, ctx.secured,
                        ctx.outstanding, "digital", costs)
        min_ratio = float(costs.get("voi", {}).get("min_voi_per_rupee", 0.5))
        if d["voi_per_rupee"] < min_ratio or d["voi"] <= 0:
            return parked(DecisionReason.LOW_VOI_DEPRIORITISED)
        if should_defer(d["voi"], _ev_wait(ctx, viable), costs):
            params = ActionParams(
                next_attempt_after=ctx.now + timedelta(days=deferral_days),
                backoff_days=deferral_days,
            )
            return _final(ctx, scores, ranked_all, Action.CONTINUE, params,
                          DecisionReason.DEFERRED_TRACE_WAIT, suppressions, report,
                          excluded, model_version, validity_hours,
                          origination_review=False, trace_info=None)
        info = TraceInfo(
            voi_per_rupee=max(0.0, d["voi_per_rupee"]),
            rank=0,  # portfolio rank assigned by rank_trace, not per-account
            est_cost=d["trace_cost"],
            trace_method="digital",
            recoverable_amount=d["recoverable_amount"],
        )
        return _final(ctx, scores, ranked_all, Action.TRACE, ActionParams(),
                      driver, suppressions, report, excluded, model_version,
                      validity_hours,
                      origination_review=ctx.address_state in ("fabricated", "incomplete"),
                      trace_info=info)

    # --- no viable contact point ---
    if not viable:
        addr = ctx.address_state
        if addr == "valid_absent":
            params = ActionParams(next_attempt_after=ctx.now + timedelta(days=1), backoff_days=1)
            return _final(ctx, scores, ranked_all or [_placeholder_ranked(ctx.now)],
                          Action.CONTINUE, params, DecisionReason.ADDRESS_ABSENT_CHANGE_TIME,
                          suppressions, report, excluded, model_version, validity_hours,
                          origination_review=False, trace_info=None)
        if addr == "hard_to_find":
            params = ActionParams(next_attempt_after=ctx.now + timedelta(days=backoff_days),
                                  backoff_days=backoff_days)
            return _final(ctx, scores, ranked_all or [_placeholder_ranked(ctx.now)],
                          Action.CONTINUE, params, DecisionReason.ADDRESS_UNRESOLVED_REVIEW,
                          suppressions, report, excluded, model_version, validity_hours,
                          origination_review=False, trace_info=None)
        if addr == "moved":
            return trace_path(_p_dead_all(scores), DecisionReason.ADDRESS_MOVED_TRACE)
        if addr in ("fabricated", "incomplete"):
            return trace_path(_p_dead_all(scores), DecisionReason.ADDRESS_FABRICATED_TRACE_FLAG)
        if "trace" in allowed and not ctx.trace_pending and not _all_recycled(scores):
            return trace_path(_p_dead_all(scores), DecisionReason.INVALID_TRACE)
        rule = (DecisionReason.GUARDRAIL_SUPPRESSED if excluded
                else DecisionReason.GUARDRAIL_TRACE_PENDING if ctx.trace_pending
                else DecisionReason.INVALID_TRACE)
        if rule is DecisionReason.INVALID_TRACE:
            return parked(DecisionReason.GUARDRAIL_SUPPRESSED)
        return parked(rule)

    best = viable[0]
    healthy_rest = [s for s in viable[1:] if s.dominant_state == "valid_reachable"]
    avoidance = max((s.state_posterior["avoiding"] for s in viable), default=0.0)
    callable_best = best.dominant_state in ("valid_reachable", "temp_unreachable")

    # --- avoiding borrower on callable lines: switch channel, never redial ---
    if best.dominant_state == "avoiding" or (avoidance >= avoidance_thr and callable_best):
        action, reason = Action.SWITCH_CHANNEL, DecisionReason.AVOIDING_SWITCH_CHANNEL
        params = ActionParams(target_channel=_pick_channel(ctx, report.allowed_channels))  # type: ignore[arg-type]
        return _constrain(ctx, scores, ranked_all, action, params, reason,
                          suppressions, report, excluded, model_version, validity_hours,
                          allowed, backoff_days)

    # --- healthy best line: keep dialling at the best slot ---
    if best.dominant_state == "valid_reachable":
        params = ActionParams(next_attempt_after=ctx.now)
        return _constrain(ctx, scores, ranked_all, Action.CONTINUE, params,
                          DecisionReason.VALID_CONTINUE, suppressions, report,
                          excluded, model_version, validity_hours, allowed, backoff_days)

    # --- temporarily unreachable: backoff, or move to a healthy alternative ---
    if best.dominant_state == "temp_unreachable":
        if healthy_rest:
            return _constrain(ctx, scores, ranked_all, Action.SWITCH_CONTACT_POINT,
                              ActionParams(next_attempt_after=ctx.now),
                              DecisionReason.TEMP_UNREACHABLE_BACKOFF, suppressions,
                              report, excluded, model_version, validity_hours,
                              allowed, backoff_days)
        params = ActionParams(next_attempt_after=ctx.now + timedelta(days=backoff_days),
                              backoff_days=backoff_days)
        return _constrain(ctx, scores, ranked_all, Action.CONTINUE, params,
                          DecisionReason.TEMP_UNREACHABLE_BACKOFF, suppressions,
                          report, excluded, model_version, validity_hours,
                          allowed, backoff_days)

    # --- third-party line: restricted use only ---
    if best.dominant_state == "third_party":
        if len(viable) > 1:
            return _constrain(ctx, scores, ranked_all, Action.SWITCH_CONTACT_POINT,
                              ActionParams(next_attempt_after=ctx.now),
                              DecisionReason.THIRD_PARTY_RESTRICT, suppressions,
                              report, excluded, model_version, validity_hours,
                              allowed, backoff_days)
        return trace_path(_p_dead_all(scores), DecisionReason.THIRD_PARTY_RESTRICT)

    # --- dead lines (switched_off_long / invalid): move or trace ---
    if best.dominant_state in ("switched_off_long", "invalid"):
        if healthy_rest or [s for s in viable[1:] if s.dominant_state in ("valid_reachable", "temp_unreachable")]:
            driver = (DecisionReason.SWITCHED_OFF_MOVE_OR_TRACE
                      if best.dominant_state == "switched_off_long" else DecisionReason.INVALID_TRACE)
            return _constrain(ctx, scores, ranked_all, Action.SWITCH_CONTACT_POINT,
                              ActionParams(next_attempt_after=ctx.now), driver,
                              suppressions, report, excluded, model_version,
                              validity_hours, allowed, backoff_days)
        driver = (DecisionReason.SWITCHED_OFF_MOVE_OR_TRACE
                  if best.dominant_state == "switched_off_long" else DecisionReason.INVALID_TRACE)
        if ctx.address_state in ("fabricated", "incomplete"):
            driver = DecisionReason.ADDRESS_FABRICATED_TRACE_FLAG
        elif ctx.address_state == "moved":
            driver = DecisionReason.ADDRESS_MOVED_TRACE
        return trace_path(_p_dead_all(scores), driver)

    # Defensive default (unreachable in practice): park, never strand.
    return parked(DecisionReason.GUARDRAIL_SUPPRESSED)


def _p_dead_all(scores: list[ContactPointScore]) -> float:
    if not scores:
        return 1.0  # no contact info at all: trace value conditioned on dead
    return max(s.p_dead for s in scores)


def _all_recycled(scores: list[ContactPointScore]) -> bool:
    return bool(scores) and all(s.dominant_state == "recycled" for s in scores)


def _ev_wait(ctx: AccountContext, viable: list[ContactPointScore]) -> float:
    """Expected value of waiting one cycle: temp_unreachable mass suggests
    self-cure without a trace (simulation-only heuristic)."""
    if not viable:
        return 0.0
    return max(s.state_posterior["temp_unreachable"] for s in viable) * max(0.0, ctx.outstanding) * 0.02


def _constrain(
    ctx: AccountContext,
    scores: list[ContactPointScore],
    ranked_all: list[RankedContactPoint],
    action: Action,
    params: ActionParams,
    reason: DecisionReason,
    suppressions: list[SuppressionEntry],
    report: object,
    excluded: dict[str, str],
    model_version: str,
    validity_hours: int,
    allowed: set[str],
    backoff_days: int,
) -> DecisionResult:
    """Guardrails narrow only: if the mapped action is disallowed, fall back
    to continue-with-backoff, then switch_contact_point, then the parked
    switch_channel. Never fall back TO trace."""
    from src.rpc.decision.guardrails import GuardrailReport as _R

    assert isinstance(report, _R)
    if action.value not in allowed:
        if "continue" in allowed:
            action = Action.CONTINUE
            params = ActionParams(
                next_attempt_after=ctx.now + timedelta(days=backoff_days),
                backoff_days=backoff_days,
            )
            reason = _guardrail_reason_for(report)
        elif "switch_contact_point" in allowed:
            action = Action.SWITCH_CONTACT_POINT
            params = ActionParams(next_attempt_after=ctx.now)
            reason = _guardrail_reason_for(report)
        else:
            return _final(ctx, scores, ranked_all, Action.SWITCH_CHANNEL,
                          ActionParams(target_channel="field"),
                          _guardrail_reason_for(report), suppressions, report,
                          excluded, model_version, validity_hours,
                          origination_review=False, trace_info=None)
    if action == Action.SWITCH_CHANNEL and not params.target_channel:
        params = ActionParams(target_channel=_pick_channel(ctx, report.allowed_channels))  # type: ignore[arg-type]
    return _final(ctx, scores, ranked_all, action, params, reason, suppressions,
                  report, excluded, model_version, validity_hours,
                  origination_review=False, trace_info=None)


def _guardrail_reason_for(report: object) -> DecisionReason:
    from src.rpc.decision.guardrails import GuardrailReport as _R

    assert isinstance(report, _R)
    mapping = {
        "CONTACT_HOURS": DecisionReason.GUARDRAIL_CONTACT_HOURS,
        "FREQUENCY_CAP": DecisionReason.GUARDRAIL_FREQ_CAP,
        "TRACE_PENDING": DecisionReason.GUARDRAIL_TRACE_PENDING,
        "DND_CHANNEL_RESTRICT": DecisionReason.GUARDRAIL_DND,
        "SUPPRESSION_ALL": DecisionReason.GUARDRAIL_SUPPRESSED,
        "EMPTY_FALLBACK_PARK": DecisionReason.GUARDRAIL_SUPPRESSED,
    }
    for rule in report.fired_rules:
        if rule in mapping:
            return mapping[rule]
    return DecisionReason.GUARDRAIL_SUPPRESSED


def _final(
    ctx: AccountContext,
    scores: list[ContactPointScore],
    ranked: list[RankedContactPoint],
    action: Action,
    params: ActionParams,
    reason: DecisionReason,
    suppressions: list[SuppressionEntry],
    report: object,
    excluded: dict[str, str],
    model_version: str,
    validity_hours: int,
    origination_review: bool,
    trace_info: TraceInfo | None,
) -> DecisionResult:
    from src.rpc.decision.guardrails import GuardrailReport as _R

    assert isinstance(report, _R)
    # TraceInfo (when present) always carries recoverable_amount from account context.
    ranked_out = ranked or [_placeholder_ranked(ctx.now)]
    decision = OutputDecision(
        account_id=ctx.account_id,
        lender_id=ctx.lender_id,
        as_of=ctx.now,
        valid_until=ctx.now + timedelta(hours=validity_hours),
        model_version=model_version,
        feature_snapshot_id="stub_snapshot",
        action=action,
        action_params=params,
        reason_code=to_contract_reason(reason),
        ranked_contact_points=ranked_out,
        trace=trace_info,
        flags=Flags(origination_review=origination_review),
    )
    return DecisionResult(
        decision=decision,
        suppressions=suppressions,
        internal_reason_code=reason,
        fired_rules=list(report.fired_rules),
        excluded_refs=dict(excluded),
    )


def decide(
    ctx: AccountContext,
    scores: list[ContactPointScore],
    guard_cfg: dict | None = None,
    cost_cfg: dict | None = None,
) -> OutputDecision:
    """Primary interface: per-account decision. Suppression requests that
    accompany the decision are available via :func:`decide_full` (the serving
    layer persists them near-real-time + daily reconciliation)."""
    return decide_full(ctx, scores, guard_cfg, cost_cfg).decision
