"""Interfaces (Protocols) and in-memory stubs for EventStore, Scorer, Decider.

Real implementations will replace these stubs when available from other workstreams.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import pandas as pd

from src.rpc.contracts import (
    Action,
    ContactPointType,
    Flags,
    InputEvent,
    OutputDecision,
    RankedContactPoint,
    StatePosterior,
    TraceInfo,
)

logger = logging.getLogger(__name__)


class EventStore(Protocol):
    """Event store interface for ingestion and reading."""

    def ingest(self, batch: list[InputEvent], source: str) -> dict[str, int]:
        """Ingest a batch of events. Returns counts: accepted, duplicate, rejected, dirty_marked."""
        ...

    def read_events(
        self,
        lender_id: str | None = None,
        borrower_id: str | None = None,
        account_id: str | None = None,
        contact_point_ref: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[InputEvent]:
        """Read events with optional filters."""
        ...


class Scorer(Protocol):
    """Scorer interface for contact point health scoring."""

    def score(
        self,
        as_of: datetime,
        contact_point_refs: list[str] | None = None,
    ) -> list[RankedContactPoint]:
        """Score contact points as of a given timestamp."""
        ...


class Decider(Protocol):
    """Decider interface for action decisions and trace queue ranking."""

    def decide(
        self,
        ctx: dict[str, Any],
        scores: list[RankedContactPoint],
    ) -> OutputDecision:
        """Decide action for an account given context and scores."""
        ...

    def rank_trace(
        self,
        candidates: list[dict[str, Any]],
        budget: float,
    ) -> list[dict[str, Any]]:
        """Rank trace candidates by VOI per rupee within budget."""
        ...


# ==================== In-Memory Stubs ====================

class InMemoryEventStore:
    """In-memory stub for EventStore."""

    def __init__(self) -> None:
        self._events: list[InputEvent] = []
        self._event_ids: set[UUID] = set()

    def ingest(self, batch: list[InputEvent], source: str) -> dict[str, int]:
        accepted = 0
        duplicate = 0
        rejected = 0
        dirty_marked = 0

        for event in batch:
            if event.event_id in self._event_ids:
                duplicate += 1
                continue

            # Basic validation
            if not event.contact_point_ref or not event.lender_id:
                rejected += 1
                continue

            self._events.append(event)
            self._event_ids.add(event.event_id)
            accepted += 1

        return {
            "accepted": accepted,
            "duplicate": duplicate,
            "rejected": rejected,
            "dirty_marked": dirty_marked,
        }

    def read_events(
        self,
        lender_id: str | None = None,
        borrower_id: str | None = None,
        account_id: str | None = None,
        contact_point_ref: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[InputEvent]:
        result = []
        for event in self._events:
            if lender_id and event.lender_id != lender_id:
                continue
            if borrower_id and event.borrower_id != borrower_id:
                continue
            if account_id and event.account_id != account_id:
                continue
            if contact_point_ref and event.contact_point_ref != contact_point_ref:
                continue
            if event_type:
                ev_type = (
                    event.event_type.value
                    if hasattr(event.event_type, "value")
                    else str(event.event_type)
                )
                if ev_type != event_type:
                    continue
            if since and event.occurred_at < since:
                continue
            if until and event.occurred_at > until:
                continue
            result.append(event)
            if len(result) >= limit:
                break
        return result


class InMemoryScorer:
    """In-memory stub for Scorer.

    Produces deterministic pseudo-scores derived from the contact point ref hash
    so that rankings are stable and reproducible across calls.
    """

    def __init__(self, model_version: str = "v0.1.0-stub") -> None:
        self.model_version = model_version
        self._scores_cache: dict[str, list[RankedContactPoint]] = {}

    def score(
        self,
        as_of: datetime,
        contact_point_refs: list[str] | None = None,
    ) -> list[RankedContactPoint]:
        cache_key = f"{as_of.isoformat()}:{','.join(sorted(contact_point_refs or []))}"
        if cache_key in self._scores_cache:
            return self._scores_cache[cache_key]

        if contact_point_refs:
            scores = [self._dummy_score(ref, as_of) for ref in contact_point_refs]
        else:
            scores = [self._dummy_score(f"cp_{i}", as_of) for i in range(3)]

        self._scores_cache[cache_key] = scores
        return scores

    def _dummy_score(self, ref: str, as_of: datetime) -> RankedContactPoint:
        import hashlib

        digest = hashlib.sha256(ref.encode()).digest()
        h0 = digest[0] / 255.0
        h1 = digest[1] / 255.0
        h2 = digest[2] / 255.0
        h3 = digest[3] / 255.0

        valid_reachable = 0.20 + 0.55 * h0
        avoiding = 0.05 + 0.20 * h1
        temp_unreachable = 0.05 + 0.15 * h2
        switched_off_long = 0.02 + 0.10 * (1.0 - h0)
        recycled = 0.01 + 0.08 * h1
        third_party = 0.01 + 0.05 * h2
        invalid = 0.01 + 0.10 * h3

        total = (
            valid_reachable
            + avoiding
            + temp_unreachable
            + switched_off_long
            + recycled
            + third_party
            + invalid
        )
        p_rpc = min(1.0, valid_reachable + 0.5 * temp_unreachable)

        return RankedContactPoint(
            ref=ref,
            type=ContactPointType.PHONE,
            p_rpc=round(p_rpc, 4),
            state_posterior=StatePosterior(
                valid_reachable=round(valid_reachable / total, 4),
                avoiding=round(avoiding / total, 4),
                temp_unreachable=round(temp_unreachable / total, 4),
                switched_off_long=round(switched_off_long / total, 4),
                recycled=round(recycled / total, 4),
                third_party=round(third_party / total, 4),
                invalid=round(invalid / total, 4),
            ),
            confidence=round(0.55 + 0.40 * h3, 4),
            best_slot="weekday_10-11" if h0 < 0.5 else "weekday_16-17",
        )


class InMemoryDecider:
    """In-memory stub for Decider."""

    def __init__(self, model_version: str = "v0.1.0-stub") -> None:
        self.model_version = model_version

    def decide(
        self,
        ctx: dict[str, Any],
        scores: list[RankedContactPoint],
    ) -> OutputDecision:
        from src.rpc.decision.engine import GuardrailsEngine, decide_action, map_reason_code

        guardrails = GuardrailsEngine()
        guardrail_result = guardrails.evaluate(ctx.get("account_id", ""), scores, ctx)

        if not scores:
            raise ValueError("No scores provided for decision")

        reason_code = map_reason_code(scores[0].state_posterior, scores[0].p_rpc, guardrail_result)
        action, params = decide_action(reason_code, guardrail_result, scores)

        as_of = ctx.get("as_of", datetime.now())
        feature_snapshot_id = ctx.get("feature_snapshot_id", "fs_dev")

        # Build trace info if action is TRACE
        trace = None
        if action == Action.TRACE:
            recoverable_amount = ctx.get("recoverable_amount", 50000.0)
            trace = TraceInfo(
                voi_per_rupee=1.5,
                rank=1,
                est_cost=500.0,
                recoverable_amount=recoverable_amount,
            )

        return OutputDecision(
            account_id=ctx.get("account_id", "unknown"),
            lender_id=ctx.get("lender_id", "unknown"),
            as_of=as_of,
            valid_until=as_of,
            model_version=self.model_version,
            feature_snapshot_id=feature_snapshot_id,
            action=action,
            action_params=params,
            reason_code=reason_code,
            ranked_contact_points=scores,
            trace=trace,
            flags=Flags(),
        )

    def rank_trace(
        self,
        candidates: list[dict[str, Any]],
        budget: float,
    ) -> list[dict[str, Any]]:
        # Simple ranking by VOI per rupee
        ranked = sorted(candidates, key=lambda x: x.get("voi_per_rupee", 0), reverse=True)
        total_cost = 0.0
        result = []
        for c in ranked:
            if total_cost + c.get("est_cost", 0) <= budget:
                total_cost += c.get("est_cost", 0)
                result.append(c)
        return result


# ==================== Factory Functions ====================

def get_event_store(config: Any | None = None) -> EventStore:
    """Get an event store: real DuckDB store when configured, else in-memory.

    The real store is selected by ``config.event_store_db`` (a DuckDB file
    path). Anything else (including no config) keeps the previous default so
    existing callers and tests are unaffected.
    """
    db_path = getattr(config, "event_store_db", None)
    if db_path:
        return DuckDBEventStoreAdapter(db_path=db_path)
    return InMemoryEventStore()


def get_scorer() -> Scorer:
    """Get Scorer instance (stub for now)."""
    return InMemoryScorer()


def get_decider() -> Decider:
    """Get Decider instance (stub for now)."""
    return InMemoryDecider()


class DuckDBEventStoreAdapter:
    """Serve-protocol adapter over the real DuckDB event store.

    Intake batches are already-canonical ``InputEvent`` objects, so they go
    through the identity ``api`` field mapping (validation, dedupe on
    ``event_id`` keeping earliest ``received_at``, dead-letter quarantine and
    dirty marking all apply). Reads convert stored rows back to
    ``InputEvent``. Only counts are logged, never event contents.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        from src.rpc.ingest.store import IngestConfig

        self._config = IngestConfig(db_path=str(db_path) if db_path else "")

    @staticmethod
    def _frame_event(event: InputEvent) -> dict[str, Any]:
        dumped = event.model_dump(mode="json")
        payload = dumped.pop("payload")
        dumped["payload"] = json.dumps(payload, sort_keys=True)
        return dumped

    def ingest(self, batch: list[InputEvent], source: str) -> dict[str, int]:
        from src.rpc.ingest import ingest as real_ingest

        if not batch:
            return {"accepted": 0, "duplicate": 0, "rejected": 0, "dirty_marked": 0}
        frame = pd.DataFrame([self._frame_event(event) for event in batch])
        # Events are canonical by construction (validated by FastAPI), so the
        # identity "api" mapping applies regardless of the caller's label.
        result = real_ingest(frame, "api", config=self._config)
        logger.info(
            "store adapter ingest rows=%d accepted=%d duplicate=%d rejected=%d "
            "dirty_marked=%d",
            len(batch),
            result["accepted"],
            result["duplicate"],
            result["rejected"],
            result["dirty_marked"],
        )
        return result

    def read_events(
        self,
        lender_id: str | None = None,
        borrower_id: str | None = None,
        account_id: str | None = None,
        contact_point_ref: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[InputEvent]:
        from src.rpc.ingest import read_events as real_read_events

        frame = real_read_events(
            received_before=until,
            event_types=[event_type] if event_type else None,
            lender_id=lender_id,
            contact_point_refs=[contact_point_ref] if contact_point_ref else None,
            db_path=self._config.db_path,
        )
        if borrower_id is not None:
            frame = frame[frame["borrower_id"] == borrower_id]
        if account_id is not None:
            frame = frame[frame["account_id"] == account_id]
        if since is not None:
            frame = frame[pd.to_datetime(frame["occurred_at"], utc=True) >= since]
        frame = frame.sort_values("occurred_at", kind="stable").head(limit)
        return [self._row_to_event(row) for _, row in frame.iterrows()]

    @staticmethod
    def _row_to_event(row: pd.Series) -> InputEvent:
        occurred = row["occurred_at"]
        received = row["received_at"]
        return InputEvent(
            event_id=UUID(str(row["event_id"])),
            event_type=str(row["event_type"]),
            lender_id=str(row["lender_id"]),
            borrower_id=str(row["borrower_id"]),
            account_id=str(row["account_id"]),
            contact_point_ref=str(row["contact_point_ref"]),
            occurred_at=occurred.to_pydatetime()
            if hasattr(occurred, "to_pydatetime")
            else occurred,
            received_at=received.to_pydatetime()
            if hasattr(received, "to_pydatetime")
            else received,
            payload=json.loads(str(row["payload"])),
        )
