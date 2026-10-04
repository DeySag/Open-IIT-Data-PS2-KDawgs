"""End-to-end smoke test: event in, decision out."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from src.rpc.contracts import (
    InputEvent,
    EventType,
    NetworkResponse,
    DialAttemptPayload,
    Action,
)
from src.rpc.serve.app import app


async def run_smoke_test():
    """Run end-to-end smoke test."""
    print("Running smoke test...")

    # Create a test event
    event = InputEvent(
        event_id=uuid4(),
        event_type=EventType.DIAL_ATTEMPT,
        lender_id="LENDER_001",
        borrower_id="BORROWER_001",
        account_id="ACC_001",
        contact_point_ref="cp_test_123",
        occurred_at=datetime.now(timezone.utc),
        received_at=datetime.now(timezone.utc),
        payload=DialAttemptPayload(
            network_response=NetworkResponse.ANSWERED,
            ring_seconds=5.0,
        ),
    )

    print(f"  Input event: {event.event_type} for {event.account_id}")

    # Test ingestion endpoint
    from fastapi.testclient import TestClient
    client = TestClient(app)

    # Test health
    response = client.get("/health")
    assert response.status_code == 200
    print(f"  Health check: {response.json()}")

    # Test event ingestion
    response = client.post("/events", json=[event.model_dump(mode="json")])
    assert response.status_code == 200
    print(f"  Event ingestion: {response.json()}")

    # Test scoring
    score_request = {
        "account_id": "ACC_001",
        "lender_id": "LENDER_001",
        "contact_points": [
            {"ref": "cp_test_123", "p_rpc": 0.8, "best_slot": "weekday_10-11"}
        ],
        "context": {"attempts_today": 1, "consent": True},
    }
    response = client.post("/score", json=score_request)
    assert response.status_code == 200
    decision = response.json()["decision"]
    print(f"  Decision: action={decision['action']}, reason={decision['reason_code']}")

    # Validate decision structure
    assert decision["action"] in [a.value for a in Action]
    assert decision["reason_code"] in [r.value for r in __import__("src.rpc.contracts", fromlist=["ReasonCode"]).ReasonCode]
    assert len(decision["ranked_contact_points"]) > 0
    assert "model_version" in decision
    assert "feature_snapshot_id" in decision

    print("  All assertions passed!")
    print("Smoke test PASSED")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())