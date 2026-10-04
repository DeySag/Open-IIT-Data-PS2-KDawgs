"""FastAPI serving layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rpc.contracts import InputEvent, OutputDecision, Action, ActionParams, ReasonCode, RankedContactPoint, StatePosterior, TraceInfo, Flags, ContactPointType
from src.rpc.decision.engine import GuardrailsEngine, decide_action, map_reason_code


app = FastAPI(title="PS2 RPC Service", version="0.1.0")

guardrails = GuardrailsEngine()


class ScoreRequest(BaseModel):
    account_id: str
    lender_id: str
    contact_points: list[dict[str, Any]]  # Simplified for v0
    context: dict[str, Any] = {}


class ScoreResponse(BaseModel):
    decision: OutputDecision


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    """Score an account and return decision."""
    # Convert contact points to RankedContactPoint (stub)
    ranked_cps = []
    for cp in request.contact_points:
        ranked_cps.append(RankedContactPoint(
            ref=cp.get("ref", "cp_1"),
            type=ContactPointType.PHONE,
            p_rpc=cp.get("p_rpc", 0.5),
            state_posterior=StatePosterior(
                valid_reachable=0.5,
                avoiding=0.1,
                temp_unreachable=0.1,
                switched_off_long=0.1,
                recycled=0.05,
                third_party=0.05,
                invalid=0.1,
            ),
            confidence=0.7,
            best_slot=cp.get("best_slot"),
        ))

    # Evaluate guardrails
    guardrail_result = guardrails.evaluate(request.account_id, ranked_cps, request.context)

    # Map to reason code (using first contact point's posterior)
    reason_code = map_reason_code(ranked_cps[0].state_posterior, ranked_cps[0].p_rpc, guardrail_result)

    # Determine action
    action, params = decide_action(reason_code, guardrail_result, ranked_cps)

    decision = OutputDecision(
        account_id=request.account_id,
        lender_id=request.lender_id,
        as_of=datetime.now(timezone.utc),
        valid_until=datetime.now(timezone.utc),
        model_version="v0.1.0",
        feature_snapshot_id="fs_dev",
        action=action,
        action_params=params,
        reason_code=reason_code,
        ranked_contact_points=ranked_cps,
        trace=TraceInfo(voi_per_rupee=1.5, rank=1, est_cost=500.0) if action == Action.TRACE else None,
        flags=Flags(),
    )

    return ScoreResponse(decision=decision)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/events")
async def ingest_events(events: list[InputEvent]) -> dict[str, int]:
    """Ingest events (stub - writes to event store)."""
    # In real implementation: deduplicate, write to event store, trigger feature recomputation
    return {"accepted": len(events)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)