"""Stable reason-code dictionary for the decision layer.

The 11 ``VALID_*``/``ADDRESS_*``/state codes match the frozen
``contracts.ReasonCode`` values one-for-one and must stay stable across
model versions. The ``GUARDRAIL_*``, ``DEFERRED_*`` and ``LOW_VOI_*`` codes
are decision-layer extensions: they are authoritative in audit logs and in
:mod:`src.rpc.decision.actions` outputs, and are coerced to the closest
frozen contract code only at the serving boundary (see
:func:`to_contract_reason`). A contract change adding them to
``contracts.ReasonCode`` is recommended but not required (flagged for the
coordinator in docs/decision.md).
"""

from __future__ import annotations

from enum import Enum

from src.rpc.contracts import ReasonCode as ContractReasonCode


class DecisionReason(str, Enum):
    # --- frozen contract codes (stable, do not rename) ---
    VALID_CONTINUE = "VALID_CONTINUE"
    TEMP_UNREACHABLE_BACKOFF = "TEMP_UNREACHABLE_BACKOFF"
    AVOIDING_SWITCH_CHANNEL = "AVOIDING_SWITCH_CHANNEL"
    SWITCHED_OFF_MOVE_OR_TRACE = "SWITCHED_OFF_MOVE_OR_TRACE"
    INVALID_TRACE = "INVALID_TRACE"
    RECYCLED_SUPPRESS = "RECYCLED_SUPPRESS"
    THIRD_PARTY_RESTRICT = "THIRD_PARTY_RESTRICT"
    ADDRESS_ABSENT_CHANGE_TIME = "ADDRESS_ABSENT_CHANGE_TIME"
    ADDRESS_MOVED_TRACE = "ADDRESS_MOVED_TRACE"
    ADDRESS_FABRICATED_TRACE_FLAG = "ADDRESS_FABRICATED_TRACE_FLAG"
    ADDRESS_UNRESOLVED_REVIEW = "ADDRESS_UNRESOLVED_REVIEW"
    # --- decision-layer extensions (guardrail / VOI drivers) ---
    GUARDRAIL_NO_CONSENT = "GUARDRAIL_NO_CONSENT"
    GUARDRAIL_DISPUTE = "GUARDRAIL_DISPUTE"
    GUARDRAIL_DECEASED = "GUARDRAIL_DECEASED"
    GUARDRAIL_LEGAL_CASE = "GUARDRAIL_LEGAL_CASE"
    GUARDRAIL_SUPPRESSED = "GUARDRAIL_SUPPRESSED"
    GUARDRAIL_DND = "GUARDRAIL_DND"
    GUARDRAIL_CONTACT_HOURS = "GUARDRAIL_CONTACT_HOURS"
    GUARDRAIL_FREQ_CAP = "GUARDRAIL_FREQ_CAP"
    GUARDRAIL_TRACE_PENDING = "GUARDRAIL_TRACE_PENDING"
    DEFERRED_TRACE_WAIT = "DEFERRED_TRACE_WAIT"
    LOW_VOI_DEPRIORITISED = "LOW_VOI_DEPRIORITISED"


REASON_CODE_INFO: dict[str, dict[str, str]] = {
    "VALID_CONTINUE": {
        "driver": "dominant posterior valid_reachable; avoidance below threshold",
        "thresholds": "avoidance < decision.avoidance_threshold",
        "action": "continue",
    },
    "TEMP_UNREACHABLE_BACKOFF": {
        "driver": "dominant posterior temp_unreachable, or trace deferred to later",
        "thresholds": "backoff_days from guardrails.yaml decision section",
        "action": "continue (with backoff)",
    },
    "AVOIDING_SWITCH_CHANNEL": {
        "driver": "borrower-level avoidance high on callable lines",
        "thresholds": "max avoiding posterior >= decision.avoidance_threshold",
        "action": "switch_channel",
    },
    "SWITCHED_OFF_MOVE_OR_TRACE": {
        "driver": "dominant posterior switched_off_long",
        "thresholds": "healthy alternative exists -> move; else trace",
        "action": "switch_contact_point | trace",
    },
    "INVALID_TRACE": {
        "driver": "dominant posterior invalid (or no viable contact point)",
        "thresholds": "no viable phone and trace eligible",
        "action": "trace",
    },
    "RECYCLED_SUPPRESS": {
        "driver": "recycled_risk or recycled posterior mass above cost-ratio threshold",
        "thresholds": "risk >= 1/(1+cost_ratio); cost_ratio in costs.yaml (assumption)",
        "action": "suppression + move to another contact point/channel",
    },
    "THIRD_PARTY_RESTRICT": {
        "driver": "third_party posterior mass above threshold; use within Fair Practices Code only",
        "thresholds": "third_party mass >= decision.third_party_threshold",
        "action": "switch_contact_point (never discuss debt)",
    },
    "ADDRESS_ABSENT_CHANGE_TIME": {
        "driver": "address_state valid_absent (phase-2 stub)",
        "thresholds": "no viable phone; address usually unoccupied at visit time",
        "action": "continue (change visit time)",
    },
    "ADDRESS_MOVED_TRACE": {
        "driver": "address_state moved (phase-2 stub)",
        "thresholds": "borrower moved; trace new address",
        "action": "trace",
    },
    "ADDRESS_FABRICATED_TRACE_FLAG": {
        "driver": "address_state fabricated/incomplete at origination (phase-2 stub)",
        "thresholds": "trace + flags.origination_review = true",
        "action": "trace (+ origination_review)",
    },
    "ADDRESS_UNRESOLVED_REVIEW": {
        "driver": "address_state hard_to_find (phase-2 stub); never written off by us",
        "thresholds": "manual review; parked, not traced",
        "action": "continue (manual review)",
    },
    "GUARDRAIL_NO_CONSENT": {
        "driver": "flags.no_consent (DPDP); suppress all outbound, never trace",
        "thresholds": "hard block",
        "action": "switch_channel/field (parked; serving layer must not execute outreach)",
    },
    "GUARDRAIL_DISPUTE": {
        "driver": "flags.dispute; stop all contact until resolved, never trace",
        "thresholds": "hard block",
        "action": "switch_channel/field (parked)",
    },
    "GUARDRAIL_DECEASED": {
        "driver": "flags.deceased_or_insolvent; suppress all, never trace",
        "thresholds": "hard block",
        "action": "switch_channel/field (parked)",
    },
    "GUARDRAIL_LEGAL_CASE": {
        "driver": "flags.legal_case; suppress all until cleared, never trace",
        "thresholds": "hard block (conservative assumption)",
        "action": "switch_channel/field (parked)",
    },
    "GUARDRAIL_SUPPRESSED": {
        "driver": "every contact point suppressed or recycled-risk; never trace",
        "thresholds": "no viable contact point + trace ineligible",
        "action": "switch_channel/field (parked)",
    },
    "GUARDRAIL_DND": {
        "driver": "flags.dnd; voice/telecaller channels removed",
        "thresholds": "channel restriction only",
        "action": "driver action via allowed channel",
    },
    "GUARDRAIL_CONTACT_HOURS": {
        "driver": "ctx.now outside contact window in Asia/Kolkata",
        "thresholds": "window in guardrails.yaml; supports overnight windows",
        "action": "driver action minus new channel/trace",
    },
    "GUARDRAIL_FREQ_CAP": {
        "driver": "attempts_today/week at cap",
        "thresholds": "caps in guardrails.yaml",
        "action": "continue with backoff | switch_channel (no new phone attempts now)",
    },
    "GUARDRAIL_TRACE_PENDING": {
        "driver": "trace already queued (trace_pending); no double-queue",
        "thresholds": "trace removed from allowed actions",
        "action": "continue with backoff",
    },
    "DEFERRED_TRACE_WAIT": {
        "driver": "EV(waiting) exceeds tracing now (e.g. temp_unreachable likely to self-cure)",
        "thresholds": "deferral params in guardrails.yaml decision section",
        "action": "continue (trace later = continue with backoff)",
    },
    "LOW_VOI_DEPRIORITISED": {
        "driver": "trace VOI per rupee below costs.yaml voi.min_voi_per_rupee",
        "thresholds": "VOI gate; avoiding accounts get ~zero VOI",
        "action": "switch_channel/field (parked, not queued)",
    },
}

# Coercion to the frozen contract enum at the serving boundary. Guardrail
# outcomes that park an account map to THIRD_PARTY_RESTRICT, the frozen code
# whose semantics are "outreach restricted by compliance". The authoritative
# internal code is always kept in DecisionResult/audit logs.
_CONTRACT_FALLBACK = {
    "GUARDRAIL_NO_CONSENT": ContractReasonCode.THIRD_PARTY_RESTRICT,
    "GUARDRAIL_DISPUTE": ContractReasonCode.THIRD_PARTY_RESTRICT,
    "GUARDRAIL_DECEASED": ContractReasonCode.THIRD_PARTY_RESTRICT,
    "GUARDRAIL_LEGAL_CASE": ContractReasonCode.THIRD_PARTY_RESTRICT,
    "GUARDRAIL_SUPPRESSED": ContractReasonCode.THIRD_PARTY_RESTRICT,
    "GUARDRAIL_DND": ContractReasonCode.THIRD_PARTY_RESTRICT,
    "GUARDRAIL_CONTACT_HOURS": ContractReasonCode.THIRD_PARTY_RESTRICT,
    "GUARDRAIL_FREQ_CAP": ContractReasonCode.THIRD_PARTY_RESTRICT,
    "GUARDRAIL_TRACE_PENDING": ContractReasonCode.TEMP_UNREACHABLE_BACKOFF,
    "DEFERRED_TRACE_WAIT": ContractReasonCode.TEMP_UNREACHABLE_BACKOFF,
    "LOW_VOI_DEPRIORITISED": ContractReasonCode.THIRD_PARTY_RESTRICT,
}


def to_contract_reason(code: DecisionReason | str) -> ContractReasonCode:
    """Coerce an internal reason code to the frozen contract enum."""
    name = code.value if isinstance(code, DecisionReason) else str(code)
    try:
        return ContractReasonCode(name)
    except ValueError:
        return _CONTRACT_FALLBACK[name]
