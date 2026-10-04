"""Decision layer: guardrails, actions, reason codes, VOI, exploration."""

from src.rpc.decision.actions import DecisionResult, decide, decide_full
from src.rpc.decision.guardrails import GuardrailReport, evaluate_guardrails, in_contact_hours
from src.rpc.decision.reason_codes import DecisionReason, to_contract_reason
from src.rpc.decision.types import (
    AccountContext,
    AccountFlags,
    ContactPointScore,
    RankedTrace,
    TraceCandidate,
)
from src.rpc.decision.voi import compute_voi, rank_trace, recoverable_amount, should_defer

__all__ = [
    "decide",
    "decide_full",
    "DecisionResult",
    "evaluate_guardrails",
    "GuardrailReport",
    "in_contact_hours",
    "DecisionReason",
    "to_contract_reason",
    "AccountContext",
    "AccountFlags",
    "ContactPointScore",
    "TraceCandidate",
    "RankedTrace",
    "compute_voi",
    "rank_trace",
    "recoverable_amount",
    "should_defer",
]
