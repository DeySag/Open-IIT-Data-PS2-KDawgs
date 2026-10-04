# Pydantic contract schemas
# FROZEN after day 1 - change only with coordinator approval

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    DIAL_ATTEMPT = "dial_attempt"
    DISPOSITION = "disposition"
    BOT_TRANSCRIPT = "bot_transcript"
    FIELD_VISIT = "field_visit"
    CONTACT_POINT_UPDATE = "contact_point_update"
    PAYMENT = "payment"


class NetworkResponse(str, Enum):
    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    SWITCHED_OFF = "switched_off"
    NOT_REACHABLE = "not_reachable"
    DOES_NOT_EXIST = "does_not_exist"
    IMMEDIATE_HANGUP = "immediate_hangup"


class Disposition(str, Enum):
    RPC = "RPC"
    WRONG_NUMBER = "wrong_number"
    THIRD_PARTY = "third_party"
    SWITCHED_OFF = "switched_off"
    NOT_REACHABLE = "not_reachable"
    PROMISE_TO_PAY = "promise_to_pay"
    DISPUTE = "dispute"
    CALLBACK = "callback"


class ContactPointType(str, Enum):
    PHONE = "phone"
    ADDRESS = "address"


class PhoneState(str, Enum):
    VALID_REACHABLE = "valid_reachable"
    AVOIDING = "avoiding"
    TEMP_UNREACHABLE = "temp_unreachable"
    SWITCHED_OFF_LONG = "switched_off_long"
    RECYCLED = "recycled"
    THIRD_PARTY = "third_party"
    INVALID = "invalid"


class AddressState(str, Enum):
    VALID_OCCUPIED = "valid_occupied"
    VALID_ABSENT = "valid_absent"
    MOVED = "moved"
    HARD_TO_FIND = "hard_to_find"
    FABRICATED = "fabricated"
    INCOMPLETE = "incomplete"


class Action(str, Enum):
    CONTINUE = "continue"
    SWITCH_CONTACT_POINT = "switch_contact_point"
    SWITCH_CHANNEL = "switch_channel"
    TRACE = "trace"


class ReasonCode(str, Enum):
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


class DialAttemptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    network_response: NetworkResponse
    ring_seconds: float = Field(ge=0)
    dialled_number: str | None = None
    caller_id: str | None = None
    channel: Literal["voice_bot", "telecaller"] = "voice_bot"


class DispositionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposition: Disposition
    remarks: str | None = None
    agent_id: str | None = None
    channel: Literal["voice_bot", "telecaller"] = "voice_bot"


class BotTranscriptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str
    language: str = "hi-en"
    extracted_phrases: list[str] = Field(default_factory=list)
    channel: Literal["voice_bot"] = "voice_bot"


class FieldVisitPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["locked_premises", "nobody_of_that_name", "met_borrower", "met_third_party", "address_not_found"]
    gps_lat: float | None = None
    gps_lon: float | None = None
    dwell_seconds: int = Field(ge=0, default=0)
    visit_time: datetime
    agent_id: str | None = None
    remarks: str | None = None


class ContactPointUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["KYC", "later_update", "bureau", "borrower_on_call", "skip_trace"]
    contact_value: str
    contact_type: ContactPointType
    is_primary: bool = False


class PaymentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(gt=0)
    payment_mode: str
    received_at: datetime


class InputEvent(BaseModel):
    """Canonical input event envelope from CN systems."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    event_id: UUID
    event_type: EventType
    lender_id: str
    borrower_id: str
    account_id: str
    contact_point_ref: str  # Hash of normalised contact point
    occurred_at: datetime
    received_at: datetime
    payload: (
        DialAttemptPayload
        | DispositionPayload
        | BotTranscriptPayload
        | FieldVisitPayload
        | ContactPointUpdatePayload
        | PaymentPayload
    )


class StatePosterior(BaseModel):
    """Posterior distribution over contact point states."""

    model_config = ConfigDict(extra="forbid")

    valid_reachable: float = Field(ge=0, le=1)
    avoiding: float = Field(ge=0, le=1)
    temp_unreachable: float = Field(ge=0, le=1)
    switched_off_long: float = Field(ge=0, le=1)
    recycled: float = Field(ge=0, le=1)
    third_party: float = Field(ge=0, le=1)
    invalid: float = Field(ge=0, le=1)

    def dominant_state(self) -> PhoneState:
        probs = self.model_dump()
        return PhoneState(max(probs, key=probs.get))


class RankedContactPoint(BaseModel):
    """Contact point ranked by health for dialer."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    ref: str
    type: ContactPointType
    p_rpc: float = Field(ge=0, le=1)
    state_posterior: StatePosterior
    confidence: float = Field(ge=0, le=1)
    best_slot: str | None = None  # e.g., "weekday_10-11"


class ActionParams(BaseModel):
    """Parameters for the chosen action."""

    model_config = ConfigDict(extra="forbid")

    next_attempt_after: datetime | None = None
    best_slot: str | None = None
    target_channel: Literal["sms", "whatsapp", "voice_bot", "telecaller", "field"] | None = None
    backoff_days: int = 0


class TraceInfo(BaseModel):
    """Skip-trace information for trace queue."""

    model_config = ConfigDict(extra="forbid")

    voi_per_rupee: float = Field(ge=0)
    rank: int = Field(ge=0)
    est_cost: float = Field(ge=0)
    trace_method: Literal["digital", "physical", "bureau"] = "digital"
    recoverable_amount: float = Field(ge=0)


class Flags(BaseModel):
    """Additional flags for downstream consumers."""

    model_config = ConfigDict(extra="forbid")

    origination_review: bool = False


class OutputDecision(BaseModel):
    """Per-account decision output to CN consumers."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    account_id: str
    lender_id: str
    as_of: datetime
    valid_until: datetime
    model_version: str
    feature_snapshot_id: str
    action: Action
    action_params: ActionParams
    reason_code: ReasonCode
    ranked_contact_points: list[RankedContactPoint] = Field(min_length=1)
    trace: TraceInfo | None = None
    flags: Flags = Field(default_factory=Flags)


class SuppressionEntry(BaseModel):
    """Suppression list entry."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    contact_point_ref: str
    lender_id: str
    reason: Literal["recycled", "third_party"]
    evidence: list[UUID] = Field(min_length=1)
    added_at: datetime
    model_version: str
    removal_requires: Literal["cn_signoff"] = "cn_signoff"