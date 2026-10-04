"""Decision layer: guardrails, actions, VOI, exploration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import yaml

from src.rpc.contracts import (
    Action,
    ActionParams,
    Flags,
    OutputDecision,
    RankedContactPoint,
    ReasonCode,
    StatePosterior,
    TraceInfo,
    ContactPointType,
)


class GuardrailDecision(str, Enum):
    ALLOW = "allow"
    RESTRICT = "restrict"
    SUPPRESS = "suppress"


@dataclass
class GuardrailResult:
    decision: GuardrailDecision
    allowed_actions: set[Action]
    reason: str | None = None


class GuardrailsEngine:
    """Hard compliance rules - evaluated first, models can only narrow."""

    def __init__(self, config_path: str = "configs/guardrails.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["guardrails"]

    def evaluate(self, account_id: str, contact_points: list[RankedContactPoint], context: dict[str, Any]) -> GuardrailResult:
        """Evaluate all guardrails for an account."""
        allowed = {Action.CONTINUE, Action.SWITCH_CONTACT_POINT, Action.SWITCH_CHANNEL, Action.TRACE}

        # Check suppression list (highest priority)
        for cp in contact_points:
            if context.get("suppression", {}).get(cp.ref):
                return GuardrailResult(
                    decision=GuardrailDecision.SUPPRESS,
                    allowed_actions=set(),
                    reason=f"Contact point {cp.ref} suppressed",
                )

        # Contact hours
        if self.config["contact_hours"]["enabled"]:
            now = datetime.now(timezone.utc)
            hour = now.hour
            if hour < self.config["contact_hours"]["start_hour"] or hour >= self.config["contact_hours"]["end_hour"]:
                allowed.discard(Action.SWITCH_CHANNEL)
                allowed.discard(Action.TRACE)
                # Only continue with backoff allowed

        # Frequency caps
        if self.config["frequency_caps"]["enabled"]:
            attempts_today = context.get("attempts_today", 0)
            if attempts_today >= self.config["frequency_caps"]["max_attempts_per_day"]:
                allowed.discard(Action.CONTINUE)
                allowed.discard(Action.SWITCH_CHANNEL)

        # DND
        if self.config["dnd"]["enabled"] and context.get("dnd", False):
            allowed.discard(Action.SWITCH_CHANNEL)  # No voice/telecaller
            # SMS/WhatsApp still allowed

        # Consent
        if self.config["consent"]["enabled"] and not context.get("consent", True):
            return GuardrailResult(
                decision=GuardrailDecision.SUPPRESS,
                allowed_actions=set(),
                reason="No consent",
            )

        # Disputes
        if self.config["disputes"]["enabled"] and context.get("dispute", False):
            return GuardrailResult(
                decision=GuardrailDecision.SUPPRESS,
                allowed_actions=set(),
                reason="Active dispute",
            )

        # Deceased/Insolvent
        if self.config["deceased_insolvent"]["enabled"] and context.get("deceased_or_insolvent", False):
            return GuardrailResult(
                decision=GuardrailDecision.SUPPRESS,
                allowed_actions=set(),
                reason="Deceased or insolvent",
            )

        return GuardrailResult(
            decision=GuardrailDecision.ALLOW,
            allowed_actions=allowed,
        )


def map_reason_code(
    state_posterior: StatePosterior,
    p_rpc: float,
    guardrail_result: GuardrailResult,
) -> ReasonCode:
    """Map model outputs to reason code."""
    dominant = state_posterior.dominant_state()

    if dominant in (PhoneState.RECYCLED, PhoneState.INVALID):
        return ReasonCode.INVALID_TRACE if dominant == PhoneState.INVALID else ReasonCode.RECYCLED_SUPPRESS

    if dominant == PhoneState.AVOIDING:
        return ReasonCode.AVOIDING_SWITCH_CHANNEL

    if dominant == PhoneState.SWITCHED_OFF_LONG:
        return ReasonCode.SWITCHED_OFF_MOVE_OR_TRACE

    if dominant == PhoneState.TEMP_UNREACHABLE:
        return ReasonCode.TEMP_UNREACHABLE_BACKOFF

    if dominant == PhoneState.THIRD_PARTY:
        return ReasonCode.THIRD_PARTY_RESTRICT

    if dominant == PhoneState.VALID_REACHABLE:
        return ReasonCode.VALID_CONTINUE

    return ReasonCode.VALID_CONTINUE


def decide_action(
    reason_code: ReasonCode,
    guardrail_result: GuardrailResult,
    contact_points: list[RankedContactPoint],
) -> tuple[Action, ActionParams]:
    """Determine action from reason code, respecting guardrails."""
    action_map = {
        ReasonCode.VALID_CONTINUE: Action.CONTINUE,
        ReasonCode.TEMP_UNREACHABLE_BACKOFF: Action.CONTINUE,
        ReasonCode.AVOIDING_SWITCH_CHANNEL: Action.SWITCH_CHANNEL,
        ReasonCode.SWITCHED_OFF_MOVE_OR_TRACE: Action.SWITCH_CONTACT_POINT,
        ReasonCode.INVALID_TRACE: Action.TRACE,
        ReasonCode.RECYCLED_SUPPRESS: Action.TRACE,  # Will be suppressed by guardrails
        ReasonCode.THIRD_PARTY_RESTRICT: Action.SWITCH_CONTACT_POINT,
        ReasonCode.ADDRESS_ABSENT_CHANGE_TIME: Action.CONTINUE,
        ReasonCode.ADDRESS_MOVED_TRACE: Action.TRACE,
        ReasonCode.ADDRESS_FABRICATED_TRACE_FLAG: Action.TRACE,
        ReasonCode.ADDRESS_UNRESOLVED_REVIEW: Action.CONTINUE,
    }

    action = action_map.get(reason_code, Action.CONTINUE)

    # Guardrails can only narrow
    if action not in guardrail_result.allowed_actions:
        # Fallback to most conservative allowed action
        if Action.CONTINUE in guardrail_result.allowed_actions:
            action = Action.CONTINUE
        elif Action.SWITCH_CONTACT_POINT in guardrail_result.allowed_actions:
            action = Action.SWITCH_CONTACT_POINT
        else:
            action = Action.TRACE  # Last resort

    params = ActionParams()
    if action == Action.CONTINUE:
        params.next_attempt_after = datetime.now(timezone.utc)
        if contact_points:
            params.best_slot = contact_points[0].best_slot
    elif action == Action.SWITCH_CHANNEL:
        params.target_channel = "whatsapp"  # Default fallback
    elif action == Action.SWITCH_CONTACT_POINT:
        params.next_attempt_after = datetime.now(timezone.utc)

    return action, params


# Import PhoneState for mapping
from src.rpc.contracts import PhoneState