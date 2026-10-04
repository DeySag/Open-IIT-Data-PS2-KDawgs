"""End-to-end smoke test: event in, decision out.

Posts a small batch of synthetic events, then exercises the
decision, dial list, trace queue and suppression endpoints.
Passes with the in-memory stubs and later with the real
modules without code changes.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

from src.rpc.contracts import (
    DialAttemptPayload,
    Disposition,
    DispositionPayload,
    EventType,
    InputEvent,
    NetworkResponse,
)
from src.rpc.serve.app import app


def _event(
    event_type: EventType,
    payload: Any,
    lender_id: str = "LENDER_001",
    borrower_id: str = "BORROWER_001",
    account_id: str = "ACC_001",
    contact_point_ref: str = "cp_smoke_1",
) -> InputEvent:
    now = datetime.now(timezone.utc)
    return InputEvent(
        event_id=uuid4(),
        event_type=event_type,
        lender_id=lender_id,
        borrower_id=borrower_id,
        account_id=account_id,
        contact_point_ref=contact_point_ref,
        occurred_at=now,
        received_at=now,
        payload=payload,
    )


def run_smoke_test() -> None:
    """Run the end-to-end smoke test."""
    print("Running smoke test...")
    client = TestClient(app)

    # 1. Health
    response = client.get("/v1/health")
    assert response.status_code == 200, response.text
    health = response.json()
    assert health["status"] == "ok"
    assert "model_version" in health or "version" in health
    print(f"  health: {health['status']} staleness={health['staleness_state']}")

    # 2. Post a small batch of synthetic events
    events = [
        _event(
            EventType.DIAL_ATTEMPT,
            DialAttemptPayload(
                network_response=NetworkResponse.ANSWERED,
                ring_seconds=5.0,
            ),
        ),
        _event(
            EventType.DISPOSITION,
            DispositionPayload(
                disposition=Disposition.RPC,
                remarks="borrower answered",
            ),
            contact_point_ref="cp_smoke_2",
        ),
    ]
    response = client.post(
        "/v1/events",
        json=[e.model_dump(mode="json") for e in events],
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert response.status_code == 200, response.text
    ingest = response.json()
    assert ingest["accepted"] == 2, ingest
    assert "valid_until" in ingest
    assert ingest["valid_until"] > ingest["generated_at"]
    print(f"  events: accepted={ingest['accepted']}")

    # 3. Get a decision (single account)
    response = client.post(
        "/v1/score",
        json={
            "account_id": "ACC_001",
            "lender_id": "LENDER_001",
        },
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert response.status_code == 200, response.text
    decision = response.json()["decision"]
    assert decision["action"] in {
        "continue",
        "switch_contact_point",
        "switch_channel",
        "trace",
    }
    assert decision["reason_code"]
    assert len(decision["ranked_contact_points"]) > 0
    assert decision["valid_until"] > decision["as_of"]
    assert decision["model_version"]
    assert decision["feature_snapshot_id"]
    print(f"  score: action={decision['action']} reason={decision['reason_code']}")

    # 4. Dial list
    response = client.get(
        "/v1/dial-lists", headers={"X-Lender-Id": "LENDER_001"}
    )
    assert response.status_code == 200, response.text
    dial = response.json()
    assert "accounts" in dial
    assert "valid_until" in dial
    print(f"  dial-lists: accounts={len(dial['accounts'])}")

    # 5. Trace queue
    response = client.get(
        "/v1/trace-queue",
        params={"budget": 100000},
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert response.status_code == 200, response.text
    trace = response.json()
    assert "entries" in trace
    print(f"  trace-queue: entries={len(trace['entries'])}")

    # 6. Suppression list
    response = client.get(
        "/v1/suppression", headers={"X-Lender-Id": "LENDER_001"}
    )
    assert response.status_code == 200, response.text
    supp = response.json()
    assert "version" in supp
    assert "entries" in supp
    print(f"  suppression: version={supp['version']} entries={len(supp['entries'])}")

    print("Smoke test PASSED")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except AssertionError as exc:
        print(f"Smoke test FAILED: {exc}")
        sys.exit(1)
