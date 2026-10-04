"""Decision engine: public entry points plus a backwards-compatible shim.

New code should import from :mod:`src.rpc.decision.actions`,
:mod:`src.rpc.decision.guardrails`, :mod:`src.rpc.decision.voi`,
:mod:`src.rpc.decision.reason_codes` and :mod:`src.rpc.decision.types`
directly (re-exported here). The legacy names below (``GuardrailsEngine``,
``map_reason_code``, ``decide_action``) are kept for the serving layer and
are reimplemented on the corrected logic. Fixes vs the day-0 boilerplate:

- contact hours use Asia/Kolkata from config (was UTC ``datetime.now``);
- suppressed / no-consent / dispute / deceased accounts can never yield
  ``trace`` (the old empty-allowed fallback returned ``TRACE``);
- recycled maps to suppression + move, never ``trace``;
- all of an account's contact points are considered, not just the first;
- ``TraceInfo`` always carries ``recoverable_amount`` from account context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.rpc.contracts import (
    Action,
    ActionParams,
    ContactPointType,
    PhoneState,
    RankedContactPoint,
    ReasonCode,
    StatePosterior,
)
from src.rpc.decision import guardrails as _guardrails_mod
from src.rpc.decision.actions import decide, decide_full
from src.rpc.decision.guardrails import GuardrailReport, evaluate_guardrails
from src.rpc.decision.reason_codes import DecisionReason, to_contract_reason
from src.rpc.decision.types import AccountContext, AccountFlags, ContactPointScore
from src.rpc.decision.voi import rank_trace


# ---------------------------------------------------------------------------
# Legacy shim (kept for src.rpc.serve.app; same signatures, fixed behaviour)
# ---------------------------------------------------------------------------

class GuardrailDecision(str):  # kept simple: plain constants holder
    ALLOW = "allow"
    RESTRICT = "restrict"
    SUPPRESS = "suppress"


@dataclass
class GuardrailResult:
    decision: str
    allowed_actions: set[Action]
    reason: str | None = None
    excluded_refs: dict[str, str] = field(default_factory=dict)


class GuardrailsEngine:
    """Hard compliance rules - evaluated first, models can only narrow."""

    def __init__(self, config_path: str = "configs/guardrails.yaml"):
        from src.rpc.decision.guardrails import load_guardrails_config

        self.config = load_guardrails_config(config_path)
        self._config_path = config_path

    def _to_ctx(self, account_id: str, context: dict[str, Any]) -> AccountContext:
        flags = AccountFlags(
            dispute=bool(context.get("dispute", False)),
            no_consent=not context.get("consent", True),
            deceased_or_insolvent=bool(context.get("deceased_or_insolvent", False)),
            dnd=bool(context.get("dnd", False)),
            legal_case=bool(context.get("legal_case", False)),
        )
        return AccountContext(
            account_id=account_id,
            lender_id=str(context.get("lender_id", "UNKNOWN")),
            borrower_id=str(context.get("borrower_id", account_id)),
            dpd_bucket=str(context.get("dpd_bucket", "31-60")),
            product=str(context.get("product", "unsecured_retail")),
            secured=bool(context.get("secured", False)),
            outstanding=float(context.get("outstanding", 0.0)),
            now=context.get("now", datetime.now().astimezone()),
            flags=flags,
            suppressed_refs=set(context.get("suppression", {}).keys()) | set(context.get("suppressed_refs", set())),
            attempts_today=int(context.get("attempts_today", 0)),
            attempts_week=int(context.get("attempts_week", 0)),
            trace_pending=bool(context.get("trace_pending", False)),
            whatsapp_opt_in=bool(context.get("whatsapp_opt_in", False)),
        )

    @staticmethod
    def _to_scores(contact_points: list[RankedContactPoint]) -> list[ContactPointScore]:
        return [
            ContactPointScore(
                contact_point_ref=cp.ref,
                type=cp.type.value if isinstance(cp.type, ContactPointType) else str(cp.type),
                as_of=datetime.now().astimezone(),
                state_posterior=cp.state_posterior.model_dump(),
                p_rpc=cp.p_rpc,
                recycled_risk=cp.state_posterior.recycled,
                confidence=cp.confidence,
            )
            for cp in contact_points
        ]

    def evaluate(
        self, account_id: str, contact_points: list[RankedContactPoint], context: dict[str, Any]
    ) -> GuardrailResult:
        """Evaluate all guardrails for an account (per-CP suppression; IST hours)."""
        ctx = self._to_ctx(account_id, context)
        scores = self._to_scores(contact_points)
        report = evaluate_guardrails(ctx, scores, self.config)
        action_map = {
            "continue": Action.CONTINUE,
            "switch_contact_point": Action.SWITCH_CONTACT_POINT,
            "switch_channel": Action.SWITCH_CHANNEL,
            "trace": Action.TRACE,
        }
        if report.hard_blocked:
            return GuardrailResult(
                decision=GuardrailDecision.SUPPRESS,
                allowed_actions={action_map[a] for a in report.allowed_actions},
                reason=(report.block_reason.value if report.block_reason else "blocked"),
                excluded_refs=dict(report.excluded_refs),
            )
        decision = GuardrailDecision.ALLOW if len(report.fired_rules) == 0 else GuardrailDecision.RESTRICT
        return GuardrailResult(
            decision=decision,
            allowed_actions={action_map[a] for a in report.allowed_actions},
            reason=";".join(report.fired_rules) if report.fired_rules else None,
            excluded_refs=dict(report.excluded_refs),
        )


def map_reason_code(
    state_posterior: StatePosterior,
    p_rpc: float,
    guardrail_result: GuardrailResult,
) -> ReasonCode:
    """Map model outputs to a contract reason code (legacy entry point)."""
    _ = p_rpc
    if guardrail_result.decision == GuardrailDecision.SUPPRESS:
        return ReasonCode.THIRD_PARTY_RESTRICT
    dominant = state_posterior.dominant_state()
    mapping = {
        PhoneState.VALID_REACHABLE: ReasonCode.VALID_CONTINUE,
        PhoneState.TEMP_UNREACHABLE: ReasonCode.TEMP_UNREACHABLE_BACKOFF,
        PhoneState.AVOIDING: ReasonCode.AVOIDING_SWITCH_CHANNEL,
        PhoneState.SWITCHED_OFF_LONG: ReasonCode.SWITCHED_OFF_MOVE_OR_TRACE,
        PhoneState.INVALID: ReasonCode.INVALID_TRACE,
        PhoneState.RECYCLED: ReasonCode.RECYCLED_SUPPRESS,
        PhoneState.THIRD_PARTY: ReasonCode.THIRD_PARTY_RESTRICT,
    }
    return mapping.get(dominant, ReasonCode.VALID_CONTINUE)


def decide_action(
    reason_code: ReasonCode,
    guardrail_result: GuardrailResult,
    contact_points: list[RankedContactPoint],
) -> tuple[Action, ActionParams]:
    """Determine action from reason code, respecting guardrails.

    Fixed: recycled/suppressed restrictions never produce TRACE; the
    last-resort fallback is the parked switch_channel, never trace.
    """
    _ = contact_points
    action_map = {
        ReasonCode.VALID_CONTINUE: Action.CONTINUE,
        ReasonCode.TEMP_UNREACHABLE_BACKOFF: Action.CONTINUE,
        ReasonCode.AVOIDING_SWITCH_CHANNEL: Action.SWITCH_CHANNEL,
        ReasonCode.SWITCHED_OFF_MOVE_OR_TRACE: Action.SWITCH_CONTACT_POINT,
        ReasonCode.INVALID_TRACE: Action.TRACE,
        ReasonCode.RECYCLED_SUPPRESS: Action.SWITCH_CONTACT_POINT,
        ReasonCode.THIRD_PARTY_RESTRICT: Action.SWITCH_CONTACT_POINT,
        ReasonCode.ADDRESS_ABSENT_CHANGE_TIME: Action.CONTINUE,
        ReasonCode.ADDRESS_MOVED_TRACE: Action.TRACE,
        ReasonCode.ADDRESS_FABRICATED_TRACE_FLAG: Action.TRACE,
        ReasonCode.ADDRESS_UNRESOLVED_REVIEW: Action.CONTINUE,
    }
    action = action_map.get(reason_code, Action.CONTINUE)

    if action not in guardrail_result.allowed_actions:
        if Action.CONTINUE in guardrail_result.allowed_actions:
            action = Action.CONTINUE
        elif Action.SWITCH_CONTACT_POINT in guardrail_result.allowed_actions:
            action = Action.SWITCH_CONTACT_POINT
        elif Action.SWITCH_CHANNEL in guardrail_result.allowed_actions:
            action = Action.SWITCH_CHANNEL
        else:
            # No allowed action (fully suppressed): park on switch_channel so
            # the account still has a next action, but never trace.
            action = Action.SWITCH_CHANNEL

    params = ActionParams()
    if action == Action.CONTINUE:
        params.next_attempt_after = datetime.now().astimezone()
    elif action == Action.SWITCH_CHANNEL:
        params.target_channel = "field"
    elif action == Action.SWITCH_CONTACT_POINT:
        params.next_attempt_after = datetime.now().astimezone()
    return action, params


__all__ = [
    "decide",
    "decide_full",
    "rank_trace",
    "evaluate_guardrails",
    "GuardrailReport",
    "GuardrailsEngine",
    "GuardrailDecision",
    "GuardrailResult",
    "map_reason_code",
    "decide_action",
    "DecisionReason",
    "to_contract_reason",
    "AccountContext",
    "AccountFlags",
    "ContactPointScore",
]
