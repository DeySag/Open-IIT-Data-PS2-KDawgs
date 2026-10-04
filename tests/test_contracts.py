"""Contract tests for Pydantic schemas."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.rpc.contracts import (
    InputEvent,
    OutputDecision,
    SuppressionEntry,
    EventType,
    NetworkResponse,
    Disposition,
    Action,
    ReasonCode,
    ContactPointType,
    PhoneState,
    DialAttemptPayload,
    DispositionPayload,
    RankedContactPoint,
    StatePosterior,
    ActionParams,
    TraceInfo,
    Flags,
)


def test_input_event_dial_attempt():
    event = InputEvent(
        event_id=uuid4(),
        event_type=EventType.DIAL_ATTEMPT,
        lender_id="LENDER_001",
        borrower_id="BORROWER_001",
        account_id="ACC_001",
        contact_point_ref="cp_hash_123",
        occurred_at=datetime.now(timezone.utc),
        received_at=datetime.now(timezone.utc),
        payload=DialAttemptPayload(
            network_response=NetworkResponse.ANSWERED,
            ring_seconds=5.0,
        ),
    )
    assert event.event_type == EventType.DIAL_ATTEMPT
    assert event.payload.network_response == NetworkResponse.ANSWERED


def test_input_event_dial_attempt_extra_field_rejected():
    with pytest.raises(Exception):
        InputEvent(
            event_id=uuid4(),
            event_type=EventType.DIAL_ATTEMPT,
            lender_id="LENDER_001",
            borrower_id="BORROWER_001",
            account_id="ACC_001",
            contact_point_ref="cp_hash_123",
            occurred_at=datetime.now(timezone.utc),
            received_at=datetime.now(timezone.utc),
            payload=DialAttemptPayload(
                network_response=NetworkResponse.ANSWERED,
                ring_seconds=5.0,
                extra_field="not_allowed",  # type: ignore
            ),
        )


def test_output_decision_serialization():
    decision = OutputDecision(
        account_id="ACC_001",
        lender_id="LENDER_001",
        as_of=datetime.now(timezone.utc),
        valid_until=datetime.now(timezone.utc),
        model_version="v0.1.0",
        feature_snapshot_id="fs_123",
        action=Action.CONTINUE,
        action_params=ActionParams(
            next_attempt_after=datetime.now(timezone.utc),
            best_slot="weekday_10-11",
        ),
        reason_code=ReasonCode.VALID_CONTINUE,
        ranked_contact_points=[
            RankedContactPoint(
                ref="cp_hash_123",
                type=ContactPointType.PHONE,
                p_rpc=0.8,
                state_posterior=StatePosterior(
                    valid_reachable=0.7,
                    avoiding=0.1,
                    temp_unreachable=0.1,
                    switched_off_long=0.05,
                    recycled=0.02,
                    third_party=0.02,
                    invalid=0.01,
                ),
                confidence=0.85,
            )
        ],
    )
    json_str = decision.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["action"] == "continue"
    assert parsed["reason_code"] == "VALID_CONTINUE"
    assert len(parsed["ranked_contact_points"]) == 1


def test_suppression_entry():
    suppression = SuppressionEntry(
        contact_point_ref="cp_hash_123",
        lender_id="LENDER_001",
        reason="recycled",
        evidence=[uuid4()],
        added_at=datetime.now(timezone.utc),
        model_version="v0.1.0",
    )
    assert suppression.reason == "recycled"
    assert suppression.removal_requires == "cn_signoff"


def test_state_posterior_dominant():
    posterior = StatePosterior(
        valid_reachable=0.6,
        avoiding=0.1,
        temp_unreachable=0.1,
        switched_off_long=0.05,
        recycled=0.05,
        third_party=0.05,
        invalid=0.05,
    )
    assert posterior.dominant_state() == PhoneState.VALID_REACHABLE


def test_reason_codes_complete():
    expected = {
        "VALID_CONTINUE",
        "TEMP_UNREACHABLE_BACKOFF",
        "AVOIDING_SWITCH_CHANNEL",
        "SWITCHED_OFF_MOVE_OR_TRACE",
        "INVALID_TRACE",
        "RECYCLED_SUPPRESS",
        "THIRD_PARTY_RESTRICT",
        "ADDRESS_ABSENT_CHANGE_TIME",
        "ADDRESS_MOVED_TRACE",
        "ADDRESS_FABRICATED_TRACE_FLAG",
        "ADDRESS_UNRESOLVED_REVIEW",
    }
    actual = {rc.value for rc in ReasonCode}
    assert actual == expected