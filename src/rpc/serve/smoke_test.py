"""End-to-end smoke test: event in, decision out.

Posts a small batch of test events, then exercises the
decision, dial list, trace queue and suppression endpoints.
Passes with the in-memory stubs and later with the real
modules without code changes.

``run_real_chain_smoke_test`` goes further: it fits the real state tracker
on the posted events and serves decisions through the real ``decide_full``
layer, proving the true pipeline (not just the stubs) runs end to end.
All data below are inline test fixtures for this test.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient

from src.rpc.contracts import (
    ContactPointType,
    DialAttemptPayload,
    Disposition,
    DispositionPayload,
    EventType,
    InputEvent,
    NetworkResponse,
    RankedContactPoint,
    StatePosterior,
)
from src.rpc.serve.app import app, create_app


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

    # 2. Post a small batch of test events
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


class StateTrackerServeScorer:
    """Serve ``Scorer`` protocol over a fitted state tracker (smoke-only).

    Fits ``StateTracker`` on the given canonical event frame once, then
    serves its real posteriors as ``RankedContactPoint`` objects. This is
    what proves the served chain is real: the posteriors carry model
    semantics (answered + RPC evidence concentrates ``valid_reachable``
    mass), which a hash stub cannot guarantee.
    """

    def __init__(self, events: pd.DataFrame) -> None:
        from src.rpc.models.state_tracker.model import (
            StateTracker,
            StateTrackerScorer,
        )

        self._scorer = StateTrackerScorer(tracker=StateTracker().fit(events))

    def score(
        self,
        as_of: datetime,
        contact_point_refs: list[str] | None = None,
    ) -> list[RankedContactPoint]:
        frame = self._scorer.score_df(as_of, list(contact_point_refs or []))
        out: list[RankedContactPoint] = []
        for _, row in frame.iterrows():
            posterior = dict(row["state_posterior"])
            out.append(
                RankedContactPoint(
                    ref=str(row["contact_point_ref"]),
                    type=ContactPointType.PHONE,
                    p_rpc=float(row["p_rpc"]),
                    state_posterior=StatePosterior(**posterior),
                    confidence=float(row["confidence"]),
                )
            )
        return out


def _frame_events(events: list[InputEvent]) -> pd.DataFrame:
    """Frame canonical events for model fitting (payload kept as dicts)."""
    rows = []
    for event in events:
        dumped = event.model_dump(mode="json")
        dumped["payload"] = dict(dumped["payload"])
        rows.append(dumped)
    return pd.DataFrame(rows)


def run_real_chain_smoke_test() -> None:
    """Smoke the true pipeline: real tracker posteriors + real decide_full."""
    print("Running real-chain smoke test...")
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
        ),
    ]
    real_app = create_app(scorer=StateTrackerServeScorer(_frame_events(events)))
    client = TestClient(real_app)

    response = client.post(
        "/v1/events",
        json=[e.model_dump(mode="json") for e in events],
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["accepted"] == 2, response.text

    response = client.post(
        "/v1/score",
        json={"account_id": "ACC_001", "lender_id": "LENDER_001"},
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
    ranked = decision["ranked_contact_points"]
    assert len(ranked) > 0
    # Real-model semantics: answered + RPC evidence must concentrate
    # valid_reachable mass. A hash stub cannot guarantee this.
    for point in ranked:
        total = sum(point["state_posterior"].values())
        assert abs(total - 1.0) < 1e-6, point
    dominant = max(
        ranked[0]["state_posterior"], key=ranked[0]["state_posterior"].get
    )
    assert dominant == "valid_reachable", ranked[0]["state_posterior"]
    assert decision["valid_until"] > decision["as_of"]
    print(
        f"  real-chain: action={decision['action']} "
        f"reason={decision['reason_code']} dominant={dominant}"
    )
    print("Real-chain smoke test PASSED")


if __name__ == "__main__":
    try:
        run_smoke_test()
        run_real_chain_smoke_test()
    except AssertionError as exc:
        print(f"Smoke test FAILED: {exc}")
        sys.exit(1)
