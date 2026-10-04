"""Compliance fast path: synchronous recycled / third-party signal detection.

Runs on every accepted event inside POST /v1/events and updates the
suppression list before the response returns. Rules are config-driven and
each triggered rule is logged as evidence. The same evidence never creates
duplicate suppression entries (idempotent).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from src.rpc.contracts import Disposition, InputEvent
from src.rpc.serve.interfaces import Scorer

logger = logging.getLogger(__name__)


def _event_type_value(event_type: Any) -> str:
    """Return the string value of an event type (enum or str)."""
    return event_type.value if hasattr(event_type, "value") else str(event_type)


def _disposition_value(disposition: Any) -> str:
    """Return the string value of a disposition (enum or str)."""
    return disposition.value if hasattr(disposition, "value") else str(disposition)


# Cues inside a bot transcript that indicate the number was recycled to a
# new subscriber (the person answering does not know the borrower).
RECYCLED_TRANSCRIPT_CUES: tuple[str, ...] = (
    "who is this",
    "who's this",
    "who is speaking",
    "wrong number",
    "name mismatch",
    "language mismatch",
    "don't know",
    "do not know",
    "not interested",
)


@dataclass
class FastPathResult:
    """Outcome of the fast path for a single event."""

    contact_point_ref: str
    reason: str  # "recycled" | "third_party"
    evidence: list[UUID]
    rule: str
    latency_ms: float


class RecycledSignalDetector:
    """Detects recycled / third-party signals synchronously on ingestion."""

    def __init__(self, scorer: Scorer | None, config: Any) -> None:
        self.scorer = scorer
        self.config = config

    def detect(self, event: InputEvent) -> list[FastPathResult]:
        """Run all rules against one event, recording per-event latency."""
        start = time.perf_counter()
        results: list[FastPathResult] = []
        try:
            results = self._detect(event)
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            for result in results:
                result.latency_ms = latency_ms
        return results

    def _detect(self, event: InputEvent) -> list[FastPathResult]:
        results: list[FastPathResult] = []
        ref = event.contact_point_ref
        event_type = _event_type_value(event.event_type)

        # Rule 1: wrong_number disposition -> recycled
        if event_type == "disposition":
            disposition = getattr(event.payload, "disposition", None)
            disposition_value = _disposition_value(disposition)
            if disposition_value == Disposition.WRONG_NUMBER.value:
                results.append(
                    FastPathResult(
                        contact_point_ref=ref,
                        reason="recycled",
                        evidence=[event.event_id],
                        rule="wrong_number_disposition",
                        latency_ms=0.0,
                    )
                )
                logger.info(
                    "fast_path rule=wrong_number_disposition cp=%s",
                    ref,
                )

            # Rule 2: third_party disposition -> third_party
            if disposition_value == Disposition.THIRD_PARTY.value:
                results.append(
                    FastPathResult(
                        contact_point_ref=ref,
                        reason="third_party",
                        evidence=[event.event_id],
                        rule="third_party_disposition",
                        latency_ms=0.0,
                    )
                )
                logger.info(
                    "fast_path rule=third_party_disposition cp=%s",
                    ref,
                )

        # Rule 3: recycled cues in a bot transcript -> recycled
        if event_type == "bot_transcript":
            transcript = str(getattr(event.payload, "transcript", "")).lower()
            phrases = [
                str(p).lower()
                for p in getattr(event.payload, "extracted_phrases", [])
            ]
            cue = self._match_cue(transcript, phrases)
            if cue is not None:
                results.append(
                    FastPathResult(
                        contact_point_ref=ref,
                        reason="recycled",
                        evidence=[event.event_id],
                        rule=f"transcript_cue:{cue}",
                        latency_ms=0.0,
                    )
                )
                logger.info(
                    "fast_path rule=transcript_cue cue=%s cp=%s",
                    cue,
                    ref,
                )

        # Rule 4: recycled_risk from the Scorer above threshold (when available)
        if self.scorer is not None:
            try:
                scores = self.scorer.score(
                    datetime.now(timezone.utc), [ref]
                )
                for score in scores:
                    if score.ref != ref:
                        continue
                    if (
                        score.state_posterior.recycled
                        >= self.config.recycled_risk_threshold
                    ):
                        results.append(
                            FastPathResult(
                                contact_point_ref=ref,
                                reason="recycled",
                                evidence=[event.event_id],
                                rule="recycled_risk_threshold",
                                latency_ms=0.0,
                            )
                        )
                        logger.info(
                            "fast_path rule=recycled_risk_threshold cp=%s",
                            ref,
                        )
            except Exception:
                logger.warning("fast_path scorer unavailable for cp=%s", ref)

        return results

    @staticmethod
    def _match_cue(
        transcript: str, phrases: list[str]
    ) -> str | None:
        for cue in RECYCLED_TRANSCRIPT_CUES:
            if cue in transcript:
                return cue
            if any(cue in phrase for phrase in phrases):
                return cue
        return None
