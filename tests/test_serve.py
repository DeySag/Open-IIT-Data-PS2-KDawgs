"""Tests for the serving layer (src/rpc/serve).

Covers:
  - schema-valid output for every action (including trace), no 500s
  - 4xx with clear errors for invalid input
  - tenancy isolation via X-Lender-Id
  - compliance fast path (all rules, latency, idempotency, removal)
  - stale-score fallback (stale flag, confidence decay, no trace)
  - valid_until > as_of, versions/ids on every response
  - end-to-end smoke (event in -> decision/dial-list/trace/suppression)
  - dial list never empty without a stated fallback action

All tests run against the in-memory stubs and are written so they
pass unchanged once the real EventStore / Scorer / Decider land.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.rpc.contracts import (
    Action,
    BotTranscriptPayload,
    ContactPointType,
    DialAttemptPayload,
    Disposition,
    DispositionPayload,
    EventType,
    FieldVisitPayload,
    InputEvent,
    NetworkResponse,
    OutputDecision,
    PhoneState,
    RankedContactPoint,
    StatePosterior,
    SuppressionEntry,
)
from src.rpc.serve.app import create_app
from src.rpc.serve.config import ServeConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_posterior(dominant: str, value: float = 0.80) -> dict:
    """Build a state posterior with a clear dominant state.

    Background recycled mass stays below the decision layer's cost-ratio
    cutoff (1/(1+100) ~= 0.0099) so healthy lines are not suppressed; this
    matches the calibrated background in decision/stubs.py.
    """
    posterior = {
        "valid_reachable": 0.02,
        "avoiding": 0.02,
        "temp_unreachable": 0.02,
        "switched_off_long": 0.02,
        "recycled": 0.005,
        "third_party": 0.02,
        "invalid": 0.02,
    }
    posterior[dominant] = value
    total = sum(posterior.values())
    return {k: round(v / total, 4) for k, v in posterior.items()}


class FixedScorer:
    """Scorer stub returning fixed posteriors per contact point ref."""

    def __init__(
        self,
        ref_to_posterior: dict[str, dict],
        model_version: str = "v0.1.0-test",
    ) -> None:
        self.ref_to_posterior = ref_to_posterior
        self.model_version = model_version

    def score(
        self,
        as_of: datetime,
        contact_point_refs: list[str] | None = None,
    ) -> list[RankedContactPoint]:
        refs = contact_point_refs or list(self.ref_to_posterior.keys())
        scores: list[RankedContactPoint] = []
        for ref in refs:
            posterior = self.ref_to_posterior.get(
                ref, make_posterior("valid_reachable")
            )
            scores.append(
                RankedContactPoint(
                    ref=ref,
                    type=ContactPointType.PHONE,
                    p_rpc=round(posterior["valid_reachable"], 4),
                    state_posterior=StatePosterior(**posterior),
                    confidence=0.80,
                    best_slot="weekday_10-11",
                )
            )
        return scores


class FailingScorer:
    """Scorer stub that always raises (simulates scorer outage)."""

    def score(
        self,
        as_of: datetime,
        contact_point_refs: list[str] | None = None,
    ) -> list[RankedContactPoint]:
        raise RuntimeError("scorer unavailable")


def make_event(
    event_type: EventType,
    payload,
    lender_id: str = "LENDER_001",
    borrower_id: str = "BORROWER_001",
    account_id: str = "ACC_001",
    contact_point_ref: str = "cp_test_1",
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


def make_dial_event(
    ref: str = "cp_test_1",
    account_id: str = "ACC_001",
    lender_id: str = "LENDER_001",
    response: NetworkResponse = NetworkResponse.ANSWERED,
) -> InputEvent:
    return make_event(
        EventType.DIAL_ATTEMPT,
        DialAttemptPayload(network_response=response, ring_seconds=5.0),
        lender_id=lender_id,
        account_id=account_id,
        contact_point_ref=ref,
    )


def permissive(app) -> None:
    """Disable time/frequency guardrails for deterministic action tests."""
    app.state.guardrails.config["contact_hours"]["enabled"] = False
    app.state.guardrails.config["frequency_caps"]["enabled"] = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Default app with in-memory stubs."""
    return create_app()


@pytest.fixture
def client(app):
    permissive(app)
    return TestClient(app)


@pytest.fixture
def trace_app():
    """App whose scorer marks cp_trace as invalid (=> trace action)."""
    scorer = FixedScorer({"cp_trace": make_posterior("invalid")})
    app = create_app(scorer=scorer)
    permissive(app)
    return app


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "data_freshness" in body
    assert "staleness_state" in body
    assert "suppression_version" in body


# ---------------------------------------------------------------------------
# Events ingestion
# ---------------------------------------------------------------------------

def test_events_ingest_counts(client):
    events = [
        make_dial_event(ref="cp_a", account_id="ACC_1"),
        make_dial_event(ref="cp_b", account_id="ACC_1"),
    ]
    response = client.post(
        "/v1/events", json=[e.model_dump(mode="json") for e in events]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] == 2
    assert body["duplicate"] == 0
    assert body["rejected"] == 0
    assert "model_version" in body
    assert "feature_snapshot_id" in body
    assert "generated_at" in body
    assert "valid_until" in body


def test_events_rejects_invalid_payload(client):
    bad = {
        "event_id": str(uuid4()),
        "event_type": "dial_attempt",
        "lender_id": "LENDER_001",
        "borrower_id": "B1",
        "account_id": "ACC_1",
        "contact_point_ref": "cp_x",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "payload": {"network_response": "not_a_real_response"},
    }
    response = client.post("/v1/events", json=[bad])
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Score: every action, no 500s, trace bug fix
# ---------------------------------------------------------------------------

def test_score_every_action_no_500(client):
    """Every action is producible and returns schema-valid output."""
    scorer = FixedScorer(
        {
            "cp_valid": make_posterior("valid_reachable"),
            "cp_avoiding": make_posterior("avoiding"),
            "cp_switched": make_posterior("switched_off_long"),
            "cp_invalid": make_posterior("invalid"),
        }
    )
    app = create_app(scorer=scorer)
    permissive(app)
    c = TestClient(app)

    cases = [
        # (account, ref, expected action, expected reason code). Expectations
        # follow the real decision layer: a single dead line with no healthy
        # alternative traces (cf. test_all_phones_dead_traces_with_reason_code
        # in test_decision.py), it does not "switch" to a nonexistent line.
        ("ACC_VALID", "cp_valid", "continue", "VALID_CONTINUE"),
        ("ACC_AVOID", "cp_avoiding", "switch_channel", "AVOIDING_SWITCH_CHANNEL"),
        ("ACC_SWITCH", "cp_switched", "trace", "SWITCHED_OFF_MOVE_OR_TRACE"),
        ("ACC_INVALID", "cp_invalid", "trace", "INVALID_TRACE"),
    ]
    for account_id, ref, expected_action, _ in cases:
        c.post(
            "/v1/events",
            json=[
                make_dial_event(
                    ref=ref, account_id=account_id
                ).model_dump(mode="json")
            ],
        )

    for account_id, ref, expected_action, expected_reason in cases:
        response = c.post(
            "/v1/score",
            json={"account_id": account_id, "lender_id": "LENDER_001"},
            headers={"X-Lender-Id": "LENDER_001"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        decision = OutputDecision(**body["decision"])
        assert decision.action == expected_action, (
            f"{account_id}: expected {expected_action}, got {decision.action}"
        )
        assert decision.reason_code == expected_reason, (
            f"{account_id}: expected {expected_reason}, got {decision.reason_code}"
        )
        assert len(decision.ranked_contact_points) >= 1
        assert decision.valid_until > decision.as_of
        if expected_action == "trace":
            assert decision.trace is not None
            assert decision.trace.recoverable_amount is not None
            assert decision.trace.recoverable_amount >= 0


def test_score_trace_includes_recoverable_amount(trace_app):
    """Regression: /v1/score must not 500 on trace; TraceInfo must
    carry recoverable_amount."""
    c = TestClient(trace_app)
    c.post(
        "/v1/events",
        json=[
            make_dial_event(
                ref="cp_trace", account_id="ACC_TRACE"
            ).model_dump(mode="json")
        ],
    )
    response = c.post(
        "/v1/score",
        json={"account_id": "ACC_TRACE", "lender_id": "LENDER_001"},
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert response.status_code == 200, response.text
    decision = OutputDecision(**response.json()["decision"])
    assert decision.action == Action.TRACE
    assert decision.trace is not None
    assert decision.trace.recoverable_amount is not None
    assert decision.trace.recoverable_amount >= 0


def test_score_invalid_input_returns_4xx(client):
    response = client.post("/v1/score", json={"lender_id": "LENDER_001"})
    assert response.status_code == 400
    assert "account_id" in response.json()["detail"]

    response = client.post("/v1/score", json={"account_id": "ACC_1"})
    assert response.status_code == 400
    assert "lender_id" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Decisions (batch, paginated)
# ---------------------------------------------------------------------------

def test_decisions_batch_paginated(client):
    for i in range(5):
        client.post(
            "/v1/events",
            json=[
                make_dial_event(
                    ref=f"cp_{i}", account_id=f"ACC_{i}"
                ).model_dump(mode="json")
            ],
        )
    response = client.get(
        "/v1/decisions",
        params={"page": 1, "page_size": 2},
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 2
    assert body["total"] == 5
    assert len(body["decisions"]) == 2
    for decision in body["decisions"]:
        OutputDecision(**decision)


# ---------------------------------------------------------------------------
# Dial lists
# ---------------------------------------------------------------------------

def test_dial_lists_ranked_with_exclude_flags(client):
    client.post(
        "/v1/events",
        json=[
            make_dial_event(
                ref="cp_dl_1", account_id="ACC_DL"
            ).model_dump(mode="json")
        ],
    )
    response = client.get(
        "/v1/dial-lists", headers={"X-Lender-Id": "LENDER_001"}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["accounts"]) == 1
    account = body["accounts"][0]
    assert account["account_id"] == "ACC_DL"
    assert len(account["contact_points"]) >= 1
    cp = account["contact_points"][0]
    assert "p_rpc" in cp
    assert "best_slot" in cp
    assert "confidence" in cp
    assert "excluded" in cp
    assert cp["excluded"] is False


def test_dial_lists_all_suppressed_has_fallback_action(client):
    """Dial list is never empty without a stated fallback action."""
    client.post(
        "/v1/events",
        json=[
            make_dial_event(
                ref="cp_sup", account_id="ACC_SUP"
            ).model_dump(mode="json")
        ],
    )
    # Suppress the only contact point directly
    client.app.state.suppression.add(
        contact_point_ref="cp_sup",
        lender_id="LENDER_001",
        reason="recycled",
        evidence=[uuid4()],
    )
    response = client.get(
        "/v1/dial-lists", headers={"X-Lender-Id": "LENDER_001"}
    )
    assert response.status_code == 200
    account = response.json()["accounts"][0]
    assert all(cp["excluded"] for cp in account["contact_points"])
    assert account["fallback_action"] is not None
    assert account["fallback_reason"] is not None


# ---------------------------------------------------------------------------
# Trace queue
# ---------------------------------------------------------------------------

def test_trace_queue_ranked_and_cut_at_budget():
    scorer = FixedScorer(
        {
            "cp_t1": make_posterior("invalid"),
            "cp_t2": make_posterior("invalid"),
            "cp_t3": make_posterior("invalid"),
        }
    )
    config = ServeConfig(
        recoverable_amount_default=500000.0,
        collection_cost_fraction=0.01,
        compliance_cost_fraction=0.01,
        trace_cost=500.0,
    )
    app = create_app(scorer=scorer, config=config)
    permissive(app)
    c = TestClient(app)
    for i, ref in enumerate(["cp_t1", "cp_t2", "cp_t3"]):
        c.post(
            "/v1/events",
            json=[
                make_dial_event(
                    ref=ref, account_id=f"ACC_T{i}"
                ).model_dump(mode="json")
            ],
        )
    response = c.get(
        "/v1/trace-queue",
        params={"budget": 1000.0},
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["entries"]) >= 1
    assert body["total_cost"] <= 1000.0
    for entry in body["entries"]:
        assert "recoverable_amount" in entry
        assert "p_find" in entry
        assert "est_cost" in entry
        assert "voi_per_rupee" in entry
        assert "rank" in entry


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

def test_suppression_list_and_diff(client):
    client.post(
        "/v1/events",
        json=[
            make_event(
                EventType.DISPOSITION,
                DispositionPayload(disposition=Disposition.WRONG_NUMBER),
                contact_point_ref="cp_sup1",
            ).model_dump(mode="json")
        ],
    )
    full = client.get(
        "/v1/suppression", headers={"X-Lender-Id": "LENDER_001"}
    )
    assert full.status_code == 200
    version = full.json()["version"]
    assert version >= 1
    assert len(full.json()["entries"]) >= 1
    SuppressionEntry(**full.json()["entries"][0])

    # Diff since the current version should be empty
    diff = client.get(
        "/v1/suppression",
        params={"since": version},
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert diff.status_code == 200
    assert diff.json()["entries"] == []


def test_suppression_removal_only_via_request(client):
    """There is no direct delete; removal goes through a pending
    request requiring CN sign-off and the entry stays in force."""
    client.post(
        "/v1/events",
        json=[
            make_event(
                EventType.DISPOSITION,
                DispositionPayload(disposition=Disposition.WRONG_NUMBER),
                contact_point_ref="cp_rm",
            ).model_dump(mode="json")
        ],
    )
    # Entry is in force
    entries = client.get(
        "/v1/suppression", headers={"X-Lender-Id": "LENDER_001"}
    ).json()["entries"]
    assert any(e["contact_point_ref"] == "cp_rm" for e in entries)

    # No DELETE endpoint exists
    assert not any(
        r.path == "/v1/suppression" and "DELETE" in r.methods
        for r in client.app.routes
    )

    # Removal request is pending and does not remove the entry
    response = client.post(
        "/v1/suppression/removal-requests",
        json={
            "contact_point_ref": "cp_rm",
            "lender_id": "LENDER_001",
            "reason": "recycled",
            "requester": "ops@example.com",
        },
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_cn_signoff"
    assert body["removal_requires"] == "cn_signoff"

    # Entry still in force after the request
    entries_after = client.get(
        "/v1/suppression", headers={"X-Lender-Id": "LENDER_001"}
    ).json()["entries"]
    assert any(e["contact_point_ref"] == "cp_rm" for e in entries_after)


def test_suppression_removal_requires_lender(client):
    response = client.post(
        "/v1/suppression/removal-requests",
        json={
            "contact_point_ref": "cp_rm",
            "reason": "recycled",
            "requester": "ops@example.com",
        },
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Visit candidates
# ---------------------------------------------------------------------------

def test_visit_candidates_address_health(client):
    now = datetime.now(timezone.utc)
    events = [
        make_event(
            EventType.FIELD_VISIT,
            FieldVisitPayload(
                outcome="met_borrower", visit_time=now
            ),
            contact_point_ref="cp_addr_1",
        ),
        make_event(
            EventType.FIELD_VISIT,
            FieldVisitPayload(
                outcome="nobody_of_that_name", visit_time=now
            ),
            contact_point_ref="cp_addr_2",
        ),
        make_event(
            EventType.FIELD_VISIT,
            FieldVisitPayload(
                outcome="address_not_found", visit_time=now
            ),
            contact_point_ref="cp_addr_3",
        ),
    ]
    client.post(
        "/v1/events", json=[e.model_dump(mode="json") for e in events]
    )
    response = client.get(
        "/v1/visit-candidates", headers={"X-Lender-Id": "LENDER_001"}
    )
    assert response.status_code == 200
    candidates = response.json()["candidates"]
    by_ref = {c["contact_point_ref"]: c for c in candidates}
    assert by_ref["cp_addr_1"]["address_health"] == "occupied"
    assert by_ref["cp_addr_2"]["address_health"] == "absent"
    assert by_ref["cp_addr_3"]["address_health"] == "moved"
    # PS3 dropped: location confidence is always null
    for candidate in candidates:
        assert candidate["location_confidence"] is None
        assert "recommendation" in candidate


# ---------------------------------------------------------------------------
# Tenancy isolation
# ---------------------------------------------------------------------------

def test_tenancy_isolation(client):
    # Lender A and Lender B each have their own account + contact point
    client.post(
        "/v1/events",
        json=[
            make_dial_event(
                ref="cp_A", account_id="ACC_A", lender_id="LENDER_A"
            ).model_dump(mode="json"),
            make_dial_event(
                ref="cp_B", account_id="ACC_B", lender_id="LENDER_B"
            ).model_dump(mode="json"),
        ],
    )

    # Scoped to lender A: never see lender B's data
    response = client.get(
        "/v1/dial-lists", headers={"X-Lender-Id": "LENDER_A"}
    )
    assert response.status_code == 200
    accounts = response.json()["accounts"]
    account_ids = {a["account_id"] for a in accounts}
    assert "ACC_A" in account_ids
    assert "ACC_B" not in account_ids
    for account in accounts:
        for cp in account["contact_points"]:
            assert cp["ref"] != "cp_B"

    # Decisions scoped to A
    response = client.get(
        "/v1/decisions", headers={"X-Lender-Id": "LENDER_A"}
    )
    decisions = response.json()["decisions"]
    assert all(d["lender_id"] == "LENDER_A" for d in decisions)
    assert all(d["account_id"] != "ACC_B" for d in decisions)

    # Suppression scoped to A
    client.app.state.suppression.add(
        contact_point_ref="cp_B",
        lender_id="LENDER_B",
        reason="recycled",
        evidence=[uuid4()],
    )
    response = client.get(
        "/v1/suppression", headers={"X-Lender-Id": "LENDER_A"}
    )
    assert all(
        e["lender_id"] == "LENDER_A" for e in response.json()["entries"]
    )


# ---------------------------------------------------------------------------
# Fast path
# ---------------------------------------------------------------------------

def test_fast_path_wrong_number(client):
    response = client.post(
        "/v1/events",
        json=[
            make_event(
                EventType.DISPOSITION,
                DispositionPayload(disposition=Disposition.WRONG_NUMBER),
                contact_point_ref="cp_wn",
            ).model_dump(mode="json")
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert any(
        fp["rule"] == "wrong_number_disposition"
        and fp["reason"] == "recycled"
        for fp in body["fast_path"]
    )
    entries = client.app.state.suppression.list_entries("LENDER_001")
    assert any(
        e.contact_point_ref == "cp_wn" and e.reason == "recycled"
        for e in entries
    )


def test_fast_path_third_party(client):
    response = client.post(
        "/v1/events",
        json=[
            make_event(
                EventType.DISPOSITION,
                DispositionPayload(disposition=Disposition.THIRD_PARTY),
                contact_point_ref="cp_tp",
            ).model_dump(mode="json")
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert any(
        fp["rule"] == "third_party_disposition"
        and fp["reason"] == "third_party"
        for fp in body["fast_path"]
    )
    entries = client.app.state.suppression.list_entries("LENDER_001")
    assert any(
        e.contact_point_ref == "cp_tp" and e.reason == "third_party"
        for e in entries
    )


def test_fast_path_transcript_cue(client):
    response = client.post(
        "/v1/events",
        json=[
            make_event(
                EventType.BOT_TRANSCRIPT,
                BotTranscriptPayload(
                    transcript="Hello, who is this? I don't know you."
                ),
                contact_point_ref="cp_cue",
            ).model_dump(mode="json")
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert any(
        fp["rule"].startswith("transcript_cue:")
        and fp["reason"] == "recycled"
        for fp in body["fast_path"]
    )
    entries = client.app.state.suppression.list_entries("LENDER_001")
    assert any(
        e.contact_point_ref == "cp_cue" and e.reason == "recycled"
        for e in entries
    )


def test_fast_path_recycled_risk_threshold():
    """recycled_risk from the Scorer above threshold triggers suppression."""
    scorer = FixedScorer({"cp_rr": make_posterior("recycled", 0.90)})
    app = create_app(scorer=scorer)
    permissive(app)
    c = TestClient(app)
    response = c.post(
        "/v1/events",
        json=[
            make_dial_event(
                ref="cp_rr", account_id="ACC_RR"
            ).model_dump(mode="json")
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert any(
        fp["rule"] == "recycled_risk_threshold"
        and fp["reason"] == "recycled"
        for fp in body["fast_path"]
    )
    entries = app.state.suppression.list_entries("LENDER_001")
    assert any(
        e.contact_point_ref == "cp_rr" and e.reason == "recycled"
        for e in entries
    )


def test_fast_path_latency_recorded_and_bounded(client):
    response = client.post(
        "/v1/events",
        json=[
            make_event(
                EventType.DISPOSITION,
                DispositionPayload(disposition=Disposition.WRONG_NUMBER),
                contact_point_ref="cp_lat",
            ).model_dump(mode="json")
        ],
    )
    assert response.status_code == 200
    latencies = response.json()["fast_path_latencies_ms"]
    assert len(latencies) >= 1
    bound = client.app.state.config.fast_path_max_latency_ms
    for latency in latencies:
        assert latency >= 0
        assert latency < bound, f"fast path latency {latency}ms >= {bound}ms"


def test_fast_path_replay_does_not_duplicate(client):
    event = make_event(
        EventType.DISPOSITION,
        DispositionPayload(disposition=Disposition.WRONG_NUMBER),
        contact_point_ref="cp_dup",
    )
    payload = event.model_dump(mode="json")

    first = client.post("/v1/events", json=[payload])
    assert first.json()["accepted"] == 1
    version_after_first = client.app.state.suppression.version

    # Replay the same event (same event_id)
    second = client.post("/v1/events", json=[payload])
    assert second.json()["duplicate"] == 1
    version_after_second = client.app.state.suppression.version

    # No new suppression entry was created
    assert version_after_second == version_after_first
    entries = client.app.state.suppression.list_entries("LENDER_001")
    matching = [e for e in entries if e.contact_point_ref == "cp_dup"]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# Stale-score fallback
# ---------------------------------------------------------------------------

def test_stale_fallback_flags_decay_and_no_trace(trace_app):
    c = TestClient(trace_app)
    # Populate accounts + score cache with a fresh (trace) decision
    c.post(
        "/v1/events",
        json=[
            make_dial_event(
                ref="cp_trace", account_id="ACC_TRACE"
            ).model_dump(mode="json")
        ],
    )
    fresh = c.post(
        "/v1/score",
        json={"account_id": "ACC_TRACE", "lender_id": "LENDER_001"},
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert fresh.json()["decision"]["action"] == "trace"
    assert fresh.json()["stale"] is False

    # Simulate a scorer outage and an aged cache
    c.app.state.scorer = FailingScorer()
    old_time = datetime.now(timezone.utc) - timedelta(hours=48)
    old_scores = FixedScorer(
        {"cp_trace": make_posterior("invalid")}
    ).score(old_time, ["cp_trace"])
    c.app.state.score_cache.put("LENDER_001:ACC_TRACE", old_time, old_scores)

    # Score: stale, confidence decayed, no trace recommendation
    stale = c.post(
        "/v1/score",
        json={"account_id": "ACC_TRACE", "lender_id": "LENDER_001"},
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert stale.status_code == 200
    body = stale.json()
    assert body["stale"] is True
    decision = OutputDecision(**body["decision"])
    assert decision.action != Action.TRACE
    assert decision.trace is None
    # Confidence decayed below the original 0.80
    assert decision.ranked_contact_points[0].confidence < 0.80

    # Dial list: stale flag set, suppression still enforced
    c.app.state.suppression.add(
        contact_point_ref="cp_trace",
        lender_id="LENDER_001",
        reason="recycled",
        evidence=[uuid4()],
    )
    dial = c.get(
        "/v1/dial-lists", headers={"X-Lender-Id": "LENDER_001"}
    )
    assert dial.json()["stale"] is True
    account = dial.json()["accounts"][0]
    assert account["contact_points"][0]["excluded"] is True
    assert account["contact_points"][0]["exclude_reason"] == "suppressed"


def test_stale_fallback_never_blocks_consumer(trace_app):
    """Even with no cache and a failing scorer, the consumer gets a
    response (never a 500)."""
    c = TestClient(trace_app)
    c.post(
        "/v1/events",
        json=[
            make_dial_event(
                ref="cp_trace", account_id="ACC_TRACE"
            ).model_dump(mode="json")
        ],
    )
    c.app.state.scorer = FailingScorer()
    c.app.state.score_cache.clear()
    response = c.post(
        "/v1/score",
        json={"account_id": "ACC_TRACE", "lender_id": "LENDER_001"},
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert response.status_code == 200
    assert response.json()["stale"] is True


# ---------------------------------------------------------------------------
# valid_until, versions, ids
# ---------------------------------------------------------------------------

def test_valid_until_greater_than_as_of(client):
    client.post(
        "/v1/events",
        json=[
            make_dial_event(
                ref="cp_vu", account_id="ACC_VU"
            ).model_dump(mode="json")
        ],
    )
    endpoints = [
        client.get("/v1/health"),
        client.get("/v1/dial-lists", headers={"X-Lender-Id": "LENDER_001"}),
        client.get("/v1/decisions", headers={"X-Lender-Id": "LENDER_001"}),
        client.get("/v1/trace-queue", headers={"X-Lender-Id": "LENDER_001"}),
        client.get("/v1/suppression", headers={"X-Lender-Id": "LENDER_001"}),
        client.get("/v1/visit-candidates", headers={"X-Lender-Id": "LENDER_001"}),
        client.post(
            "/v1/score",
            json={"account_id": "ACC_VU", "lender_id": "LENDER_001"},
            headers={"X-Lender-Id": "LENDER_001"},
        ),
    ]
    for response in endpoints:
        assert response.status_code == 200
        body = response.json()
        assert "valid_until" in body
        assert "generated_at" in body
        assert body["valid_until"] > body["generated_at"]

    # Decision-level as_of < valid_until
    decision = endpoints[6].json()["decision"]
    assert decision["valid_until"] > decision["as_of"]


def test_versions_and_ids_on_every_response(client):
    client.post(
        "/v1/events",
        json=[
            make_dial_event(
                ref="cp_vid", account_id="ACC_VID"
            ).model_dump(mode="json")
        ],
    )
    responses = [
        client.get("/v1/health"),
        client.get("/v1/dial-lists", headers={"X-Lender-Id": "LENDER_001"}),
        client.get("/v1/decisions", headers={"X-Lender-Id": "LENDER_001"}),
        client.get("/v1/trace-queue", headers={"X-Lender-Id": "LENDER_001"}),
        client.get("/v1/suppression", headers={"X-Lender-Id": "LENDER_001"}),
        client.get("/v1/visit-candidates", headers={"X-Lender-Id": "LENDER_001"}),
        client.post(
            "/v1/score",
            json={"account_id": "ACC_VID", "lender_id": "LENDER_001"},
            headers={"X-Lender-Id": "LENDER_001"},
        ),
    ]
    for response in responses:
        body = response.json()
        if response.request.url.path == "/v1/health":
            assert "version" in body
        else:
            assert "model_version" in body
            assert "feature_snapshot_id" in body


# ---------------------------------------------------------------------------
# API key (placeholder)
# ---------------------------------------------------------------------------

def test_api_key_check():
    config = ServeConfig(api_key="secret-key")
    app = create_app(config=config)
    permissive(app)
    c = TestClient(app)
    no_key = c.get("/v1/health")
    assert no_key.status_code == 401
    with_key = c.get("/v1/health", headers={"X-API-Key": "secret-key"})
    assert with_key.status_code == 200


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------

def test_end_to_end_smoke():
    """Post a small batch of test events, then get a decision,
    dial list, trace queue entry and suppression entry. Passes with
    the stubs and later with the real modules without code changes."""
    app = create_app()
    permissive(app)
    c = TestClient(app)

    # 1. Post a small batch of test events
    events = [
        make_dial_event(ref="cp_e2e_1", account_id="ACC_E2E"),
        make_event(
            EventType.DISPOSITION,
            DispositionPayload(disposition=Disposition.WRONG_NUMBER),
            contact_point_ref="cp_e2e_2",
            account_id="ACC_E2E",
        ),
    ]
    response = c.post(
        "/v1/events", json=[e.model_dump(mode="json") for e in events]
    )
    assert response.status_code == 200
    assert response.json()["accepted"] == 2

    # 2. Get a decision
    decision_resp = c.post(
        "/v1/score",
        json={"account_id": "ACC_E2E", "lender_id": "LENDER_001"},
        headers={"X-Lender-Id": "LENDER_001"},
    )
    assert decision_resp.status_code == 200
    decision = OutputDecision(**decision_resp.json()["decision"])
    assert decision.action in {
        Action.CONTINUE,
        Action.SWITCH_CONTACT_POINT,
        Action.SWITCH_CHANNEL,
        Action.TRACE,
    }

    # 3. Get a dial list
    dial_resp = c.get("/v1/dial-lists", headers={"X-Lender-Id": "LENDER_001"})
    assert dial_resp.status_code == 200
    assert len(dial_resp.json()["accounts"]) >= 1

    # 4. Get a trace queue (may be empty if no trace candidates)
    trace_resp = c.get(
        "/v1/trace-queue", headers={"X-Lender-Id": "LENDER_001"}
    )
    assert trace_resp.status_code == 200
    assert "entries" in trace_resp.json()

    # 5. Get a suppression entry (from the wrong_number fast path)
    supp_resp = c.get("/v1/suppression", headers={"X-Lender-Id": "LENDER_001"})
    assert supp_resp.status_code == 200
    entries = supp_resp.json()["entries"]
    assert any(e["reason"] == "recycled" for e in entries)


# ---------------------------------------------------------------------------
# Dominant state mapping sanity
# ---------------------------------------------------------------------------

def test_posterior_dominant_state():
    posterior = StatePosterior(**make_posterior("avoiding"))
    assert posterior.dominant_state() == PhoneState.AVOIDING
    posterior = StatePosterior(**make_posterior("invalid"))
    assert posterior.dominant_state() == PhoneState.INVALID
