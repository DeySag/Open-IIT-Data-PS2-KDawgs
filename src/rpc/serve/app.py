"""FastAPI serving layer for the PS2 RPC service.

Endpoints (all under /v1):
  POST /events                 batch intake + compliance fast path
  GET  /dial-lists             per-account ranked contact points
  GET  /decisions              batch decisions (paginated)
  POST /score                  single-account decision
  GET  /trace-queue            accounts ranked by VOI per rupee
  GET  /suppression            suppression list (full or ?since=version diff)
  POST /suppression/removal-requests  create a pending removal request
  GET  /visit-candidates       address health stub from field_visit events
  GET  /health                 data freshness and staleness state

Every response carries model_version, feature_snapshot_id, generated_at
and valid_until. Lender scoping is via the X-Lender-Id header. A
placeholder X-API-Key check is enforced when configured.

All collaborators are held on ``app.state`` so tests (and later the
real modules) can be swapped in without code changes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.rpc.contracts import (
    Action,
    ContactPointType,
    Flags,
    InputEvent,
    OutputDecision,
    RankedContactPoint,
    StatePosterior,
    SuppressionEntry,
    TraceInfo,
)
from src.rpc.decision.engine import (
    GuardrailsEngine,
)
from src.rpc.serve.config import ServeConfig
from src.rpc.serve.fast_path import RecycledSignalDetector
from src.rpc.serve.interfaces import (
    Decider,
    EventStore,
    InMemoryDecider,
    InMemoryScorer,
    Scorer,
    decide_from_ranked,
    get_event_store,
)
from src.rpc.serve.suppression import SuppressionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ResponseEnvelope(BaseModel):
    """Metadata carried on every response."""

    model_version: str
    feature_snapshot_id: str
    generated_at: datetime
    valid_until: datetime
    stale: bool = False


class FastPathResultModel(BaseModel):
    contact_point_ref: str
    reason: str
    evidence: list[UUID]
    rule: str
    latency_ms: float


class EventsResponse(ResponseEnvelope):
    accepted: int
    duplicate: int
    rejected: int
    dirty_marked: int
    fast_path: list[FastPathResultModel] = Field(default_factory=list)
    fast_path_latencies_ms: list[float] = Field(default_factory=list)


class DialListContactPoint(BaseModel):
    ref: str
    type: str
    p_rpc: float
    best_slot: str | None = None
    confidence: float
    excluded: bool = False
    exclude_reason: str | None = None
    state_posterior: StatePosterior


class DialListAccount(BaseModel):
    account_id: str
    contact_points: list[DialListContactPoint]
    fallback_action: str | None = None
    fallback_reason: str | None = None


class DialListResponse(ResponseEnvelope):
    lender_id: str | None = None
    date: datetime
    accounts: list[DialListAccount]


class DecisionsResponse(ResponseEnvelope):
    decisions: list[OutputDecision]
    page: int
    page_size: int
    total: int


class ScoreResponse(ResponseEnvelope):
    decision: OutputDecision


class TraceQueueEntry(BaseModel):
    account_id: str
    lender_id: str
    voi_per_rupee: float
    recoverable_amount: float
    p_find: float
    est_cost: float
    rank: int


class TraceQueueResponse(ResponseEnvelope):
    lender_id: str | None = None
    budget: float
    total_cost: float
    entries: list[TraceQueueEntry]


class SuppressionListResponse(ResponseEnvelope):
    version: int
    entries: list[SuppressionEntry]


class RemovalRequestBody(BaseModel):
    contact_point_ref: str
    lender_id: str | None = None
    reason: str
    requester: str
    evidence: list[UUID] = Field(default_factory=list)
    notes: str | None = None


class RemovalRequestResponse(ResponseEnvelope):
    request_id: str
    status: str
    removal_requires: str


class VisitCandidate(BaseModel):
    account_id: str
    lender_id: str
    contact_point_ref: str
    address_health: str  # occupied | absent | moved | unresolved
    best_visit_window: str | None = None
    recommendation: str
    origination_review: bool = False
    location_confidence: float | None = None  # PS3 dropped


class VisitCandidatesResponse(ResponseEnvelope):
    candidates: list[VisitCandidate]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_version: str
    feature_snapshot_id: str
    generated_at: datetime
    valid_until: datetime
    data_freshness: dict[str, Any]
    last_batch_time: datetime | None = None
    staleness_state: str
    suppression_version: int


# ---------------------------------------------------------------------------
# Score cache for stale fallback
# ---------------------------------------------------------------------------

class ScoreCache:
    """Per-account score cache with timestamps for staleness detection."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[datetime, list[RankedContactPoint]]] = {}

    def get(
        self, key: str
    ) -> tuple[datetime, list[RankedContactPoint]] | None:
        return self._cache.get(key)

    def put(
        self, key: str, as_of: datetime, scores: list[RankedContactPoint]
    ) -> None:
        self._cache[key] = (as_of, scores)

    def items(
        self,
    ) -> list[tuple[str, tuple[datetime, list[RankedContactPoint]]]]:
        return list(self._cache.items())

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    event_store: EventStore | None = None,
    scorer: Scorer | None = None,
    decider: Decider | None = None,
    config: ServeConfig | None = None,
) -> FastAPI:
    """Build the FastAPI app with injectable dependencies."""

    cfg = config or ServeConfig.load()
    store: EventStore = event_store or get_event_store(cfg)
    scoring: Scorer = scorer or InMemoryScorer(cfg.model_version)
    deciding: Decider = decider or InMemoryDecider(cfg.model_version)
    suppression = SuppressionStore(cfg.model_version)
    detector = RecycledSignalDetector(scoring, cfg)
    guardrails = GuardrailsEngine()
    score_cache = ScoreCache()

    app = FastAPI(title="PS2 RPC Service", version=cfg.model_version)
    app.state.config = cfg
    app.state.event_store = store
    app.state.scorer = scoring
    app.state.decider = deciding
    app.state.suppression = suppression
    app.state.detector = detector
    app.state.guardrails = guardrails
    app.state.score_cache = score_cache
    app.state.last_batch_time = None
    app.state.fast_path_latencies = []

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if cfg.api_key is not None and x_api_key != cfg.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def resolve_lender(
        x_lender_id: str | None = Header(default=None),
    ) -> str | None:
        return x_lender_id

    def envelope() -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "model_version": cfg.model_version,
            "feature_snapshot_id": cfg.feature_snapshot_id,
            "generated_at": now,
            "valid_until": now + timedelta(hours=cfg.validity_hours),
        }

    # ------------------------------------------------------------------
    # Helpers (reference app.state so dependencies are swappable)
    # ------------------------------------------------------------------

    def _account_borrower(
        lender_id: str | None,
        account_id: str,
    ) -> str | None:
        """Look up the borrower owning an account from stored events."""
        try:
            events = app.state.event_store.read_events(
                lender_id=lender_id, account_id=account_id, limit=1000
            )
        except Exception:
            return None
        borrowers = [event.borrower_id for event in events if event.borrower_id]
        if not borrowers:
            return None
        return max(set(borrowers), key=borrowers.count)

    def _account_contact_points(
        lender_id: str | None,
    ) -> dict[str, list[str]]:
        """Derive account -> contact_point_refs from the event store."""
        events = app.state.event_store.read_events(
            lender_id=lender_id, limit=100000
        )
        mapping: dict[str, list[str]] = {}
        for event in events:
            refs = mapping.setdefault(event.account_id, [])
            if event.contact_point_ref not in refs:
                refs.append(event.contact_point_ref)
        return mapping

    def _decay_confidence(
        scores: list[RankedContactPoint], age_hours: float
    ) -> list[RankedContactPoint]:
        max_age = cfg.max_score_age_hours
        decay = max(cfg.confidence_decay_floor, 1.0 - age_hours / max_age)
        decayed: list[RankedContactPoint] = []
        for score in scores:
            decayed.append(
                score.model_copy(
                    update={"confidence": round(score.confidence * decay, 4)}
                )
            )
        return decayed

    def _get_account_scores(
        lender_id: str | None,
        account_id: str,
        contact_point_refs: list[str],
        as_of: datetime,
    ) -> tuple[list[RankedContactPoint], bool, float]:
        """Return (scores, stale, age_hours) with stale fallback.

        Serves cached scores with decayed confidence when the scorer
        fails or the cache is past its max age. Never raises.
        """
        cache_key = f"{lender_id}:{account_id}"
        cached = app.state.score_cache.get(cache_key)

        # Fresh cache hit: no need to recompute
        if cached is not None:
            cached_as_of, cached_scores = cached
            age_hours = max(0.0, (as_of - cached_as_of).total_seconds() / 3600.0)
            if age_hours < cfg.max_score_age_hours:
                return cached_scores, False, age_hours

        # Cache stale or missing: try to recompute
        try:
            scores = app.state.scorer.score(as_of, contact_point_refs)
            app.state.score_cache.put(cache_key, as_of, scores)
            return scores, False, 0.0
        except Exception:
            logger.warning(
                "scorer failed for account=%s, serving stale", account_id
            )

        if cached is not None:
            cached_as_of, cached_scores = cached
            age_hours = max(
                0.0, (as_of - cached_as_of).total_seconds() / 3600.0
            )
            decayed = _decay_confidence(cached_scores, age_hours)
            return decayed, True, age_hours

        # No cache at all: empty scores, stale
        return [], True, float(cfg.max_score_age_hours)

    def _apply_stale_guardrails(
        decision: OutputDecision, stale: bool
    ) -> OutputDecision:
        """Never recommend trace from stale data."""
        if not stale:
            return decision
        if decision.action == Action.TRACE:
            if len(decision.ranked_contact_points) > 1:
                new_action = Action.SWITCH_CONTACT_POINT
            else:
                new_action = Action.CONTINUE
            decision = decision.model_copy(
                update={"action": new_action, "trace": None}
            )
        return decision

    def _build_decision(
        lender_id: str | None,
        account_id: str,
        contact_point_refs: list[str],
        as_of: datetime,
        context: dict[str, Any] | None = None,
    ) -> tuple[OutputDecision, bool]:
        """Compute a decision for one account with stale fallback."""
        scores, stale, _age = _get_account_scores(
            lender_id, account_id, contact_point_refs, as_of
        )

        ctx: dict[str, Any] = {
            "account_id": account_id,
            "lender_id": lender_id or "unknown",
            "as_of": as_of,
            "feature_snapshot_id": cfg.feature_snapshot_id,
            "recoverable_amount": cfg.recoverable_amount_default,
            "suppression": {
                ref: True
                for ref in contact_point_refs
                if app.state.suppression.is_suppressed(ref, lender_id or "")
            },
        }
        if context:
            ctx.update(context)

        if not scores:
            # No viable contact point: route to trace (fresh) or review (stale)
            empty_posterior = StatePosterior(
                valid_reachable=0.0,
                avoiding=0.0,
                temp_unreachable=0.0,
                switched_off_long=0.0,
                recycled=0.0,
                third_party=0.0,
                invalid=1.0,
            )
            placeholder = RankedContactPoint(
                ref=contact_point_refs[0] if contact_point_refs else "unknown",
                type=ContactPointType.PHONE,
                p_rpc=0.0,
                state_posterior=empty_posterior,
                confidence=0.0,
            )
            scores = [placeholder]

        borrower_id = (context or {}).get("borrower_id") or _account_borrower(
            lender_id, account_id
        )
        # Simulation-only fallback: serving rarely knows the borrower id, but
        # the decision layer needs one for trace candidates. Prefer the
        # stored owner; otherwise derive a stable placeholder from the account.
        ctx["borrower_id"] = borrower_id or f"BORR_FOR_{account_id}"

        # Single construction point: the real decide_full decision layer
        # (guardrails -> exclusions -> action -> VOI gate), not the legacy
        # shim. It runs under the app's configured guardrails (so runtime
        # toggles such as permissive test mode apply), and suppressions it
        # emits are persisted immediately.
        result = decide_from_ranked(
            ctx, scores, guard_cfg=app.state.guardrails.config
        )
        for entry in result.suppressions:
            app.state.suppression.add(
                entry.contact_point_ref,
                entry.lender_id,
                entry.reason,  # type: ignore[arg-type]
                list(entry.evidence),
            )
        decision = _apply_stale_guardrails(result.decision, stale)
        return decision, stale

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.post("/v1/events", response_model=EventsResponse)
    async def ingest_events(
        events: list[InputEvent],
        request: Request,
        _key: None = Depends(require_api_key),
    ) -> EventsResponse:
        """Batch event intake. Triggers the compliance fast path on each
        accepted event before the response returns."""
        counts = app.state.event_store.ingest(events, source="api")

        fast_path_results: list[FastPathResultModel] = []
        latencies: list[float] = []
        for event in events:
            results = app.state.detector.detect(event)
            for result in results:
                entry = app.state.suppression.add(
                    contact_point_ref=result.contact_point_ref,
                    lender_id=event.lender_id,
                    reason=result.reason,
                    evidence=result.evidence,
                )
                if entry is not None:
                    logger.info(
                        "fast_path suppression added cp=%s reason=%s",
                        result.contact_point_ref,
                        result.reason,
                    )
            for result in results:
                fast_path_results.append(
                    FastPathResultModel(
                        contact_point_ref=result.contact_point_ref,
                        reason=result.reason,
                        evidence=result.evidence,
                        rule=result.rule,
                        latency_ms=result.latency_ms,
                    )
                )
                latencies.append(result.latency_ms)

        app.state.fast_path_latencies.extend(latencies)
        app.state.last_batch_time = datetime.now(timezone.utc)

        env = envelope()
        return EventsResponse(
            **env,
            accepted=counts["accepted"],
            duplicate=counts["duplicate"],
            rejected=counts["rejected"],
            dirty_marked=counts["dirty_marked"],
            fast_path=fast_path_results,
            fast_path_latencies_ms=latencies,
        )

    @app.get("/v1/dial-lists", response_model=DialListResponse)
    async def dial_lists(
        lender_id: str | None = Query(default=None),
        date: datetime | None = Query(default=None),
        _key: None = Depends(require_api_key),
        header_lender: str | None = Depends(resolve_lender),
    ) -> DialListResponse:
        """Per-account contact points ranked by health with exclude flags."""
        scope = header_lender or lender_id
        as_of = date or datetime.now(timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        accounts = _account_contact_points(scope)
        any_stale = False
        account_results: list[DialListAccount] = []

        for account_id, refs in accounts.items():
            scores, stale, _age = _get_account_scores(
                scope, account_id, refs, as_of
            )
            any_stale = any_stale or stale

            contact_points: list[DialListContactPoint] = []
            for score in scores:
                is_suppressed = app.state.suppression.is_suppressed(
                    score.ref, scope or ""
                )
                dead = (
                    score.state_posterior.invalid
                    + score.state_posterior.recycled
                ) >= cfg.dead_contact_threshold
                excluded = is_suppressed or dead
                exclude_reason = None
                if is_suppressed:
                    exclude_reason = "suppressed"
                elif dead:
                    exclude_reason = "dead_beyond_threshold"

                cp_type = score.type
                cp_type_value = (
                    cp_type.value if hasattr(cp_type, "value") else str(cp_type)
                )
                contact_points.append(
                    DialListContactPoint(
                        ref=score.ref,
                        type=cp_type_value,
                        p_rpc=score.p_rpc,
                        best_slot=score.best_slot,
                        confidence=score.confidence,
                        excluded=excluded,
                        exclude_reason=exclude_reason,
                        state_posterior=score.state_posterior,
                    )
                )

            fallback_action = None
            fallback_reason = None
            if contact_points and all(cp.excluded for cp in contact_points):
                if stale:
                    fallback_action = "manual_review"
                    fallback_reason = "all_contact_points_excluded_stale"
                else:
                    fallback_action = "trace"
                    fallback_reason = "all_contact_points_excluded"

            account_results.append(
                DialListAccount(
                    account_id=account_id,
                    contact_points=contact_points,
                    fallback_action=fallback_action,
                    fallback_reason=fallback_reason,
                )
            )

        env = envelope()
        return DialListResponse(
            **env,
            lender_id=scope,
            date=as_of,
            accounts=account_results,
            stale=any_stale,
        )

    @app.get("/v1/decisions", response_model=DecisionsResponse)
    async def decisions(
        lender_id: str | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=500),
        _key: None = Depends(require_api_key),
        header_lender: str | None = Depends(resolve_lender),
    ) -> DecisionsResponse:
        """Batch decisions, paginated."""
        scope = header_lender or lender_id
        as_of = datetime.now(timezone.utc)
        accounts = _account_contact_points(scope)

        all_decisions: list[OutputDecision] = []
        any_stale = False
        for account_id, refs in accounts.items():
            decision, stale = _build_decision(scope, account_id, refs, as_of)
            any_stale = any_stale or stale
            all_decisions.append(decision)

        total = len(all_decisions)
        start = (page - 1) * page_size
        page_decisions = all_decisions[start : start + page_size]

        env = envelope()
        return DecisionsResponse(
            **env,
            decisions=page_decisions,
            page=page,
            page_size=page_size,
            total=total,
            stale=any_stale,
        )

    @app.post("/v1/score", response_model=ScoreResponse)
    async def score(
        body: dict[str, Any],
        _key: None = Depends(require_api_key),
        header_lender: str | None = Depends(resolve_lender),
    ) -> ScoreResponse:
        """Score a single account and return its decision."""
        account_id = body.get("account_id")
        lender_id = header_lender or body.get("lender_id")
        if not account_id:
            raise HTTPException(status_code=400, detail="account_id required")
        if not lender_id:
            raise HTTPException(status_code=400, detail="lender_id required")

        as_of = body.get("as_of")
        if isinstance(as_of, str):
            as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        as_of = as_of or datetime.now(timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        context = body.get("context") or {}
        accounts = _account_contact_points(lender_id)
        refs = accounts.get(account_id, [])

        decision, stale = _build_decision(
            lender_id, account_id, refs, as_of, context
        )

        env = envelope()
        return ScoreResponse(**env, decision=decision, stale=stale)

    @app.get("/v1/trace-queue", response_model=TraceQueueResponse)
    async def trace_queue(
        lender_id: str | None = Query(default=None),
        budget: float = Query(default=None),
        _key: None = Depends(require_api_key),
        header_lender: str | None = Depends(resolve_lender),
    ) -> TraceQueueResponse:
        """Accounts ranked by VOI per rupee, cut at the trace budget."""
        scope = header_lender or lender_id
        trace_budget = budget if budget is not None else cfg.default_trace_budget
        as_of = datetime.now(timezone.utc)
        accounts = _account_contact_points(scope)

        candidates: list[dict[str, Any]] = []
        for account_id, refs in accounts.items():
            decision, _stale = _build_decision(scope, account_id, refs, as_of)
            if decision.action != Action.TRACE:
                continue

            recoverable_amount = cfg.recoverable_amount_default
            p_find = cfg.p_find
            p_reached = cfg.recovery_if_reached
            p_not_reached = cfg.recovery_if_not_reached
            trace_cost = cfg.trace_cost

            incremental = (p_reached - p_not_reached) * recoverable_amount
            collection_cost = cfg.collection_cost_fraction * recoverable_amount
            compliance_cost = cfg.compliance_cost_fraction * recoverable_amount
            voi = (
                p_find * incremental
                - trace_cost
                - collection_cost
                - compliance_cost
            )
            voi_per_rupee = voi / trace_cost if trace_cost > 0 else 0.0

            candidates.append(
                {
                    "account_id": account_id,
                    "lender_id": scope or "unknown",
                    "voi_per_rupee": round(voi_per_rupee, 4),
                    "recoverable_amount": recoverable_amount,
                    "p_find": p_find,
                    "est_cost": trace_cost,
                }
            )

        candidates.sort(key=lambda c: c["voi_per_rupee"], reverse=True)
        total_cost = 0.0
        entries: list[TraceQueueEntry] = []
        rank = 0
        for candidate in candidates:
            if total_cost + candidate["est_cost"] > trace_budget:
                continue
            rank += 1
            total_cost += candidate["est_cost"]
            entries.append(TraceQueueEntry(rank=rank, **candidate))

        env = envelope()
        return TraceQueueResponse(
            **env,
            lender_id=scope,
            budget=trace_budget,
            total_cost=round(total_cost, 2),
            entries=entries,
        )

    @app.get("/v1/suppression", response_model=SuppressionListResponse)
    async def suppression_list(
        lender_id: str | None = Query(default=None),
        since: int = Query(default=0, ge=0),
        _key: None = Depends(require_api_key),
        header_lender: str | None = Depends(resolve_lender),
    ) -> SuppressionListResponse:
        """Full suppression list, or a diff since a given version."""
        scope = header_lender or lender_id
        entries = app.state.suppression.list_entries(
            lender_id=scope, since_version=since
        )
        env = envelope()
        return SuppressionListResponse(
            **env, version=app.state.suppression.version, entries=entries
        )

    @app.post(
        "/v1/suppression/removal-requests",
        response_model=RemovalRequestResponse,
    )
    async def suppression_removal_request(
        body: RemovalRequestBody,
        _key: None = Depends(require_api_key),
        header_lender: str | None = Depends(resolve_lender),
    ) -> RemovalRequestResponse:
        """Create a pending removal request requiring CN sign-off.

        There is no direct delete; the entry stays in force until CN
        approves the request out-of-band.
        """
        scope = header_lender or body.lender_id
        if not scope:
            raise HTTPException(status_code=400, detail="lender_id required")

        request = app.state.suppression.request_removal(
            contact_point_ref=body.contact_point_ref,
            lender_id=scope,
            reason=body.reason,
            requester=body.requester,
            evidence=body.evidence,
            notes=body.notes,
        )
        env = envelope()
        return RemovalRequestResponse(
            **env,
            request_id=request["request_id"],
            status=request["status"],
            removal_requires=request["removal_requires"],
        )

    @app.get("/v1/visit-candidates", response_model=VisitCandidatesResponse)
    async def visit_candidates(
        lender_id: str | None = Query(default=None),
        _key: None = Depends(require_api_key),
        header_lender: str | None = Depends(resolve_lender),
    ) -> VisitCandidatesResponse:
        """Address health stub derived from field_visit events.

        Location confidence is always null (PS3 is dropped from scope).
        """
        scope = header_lender or lender_id
        events = app.state.event_store.read_events(
            lender_id=scope, event_type="field_visit", limit=100000
        )

        # Group field visits by account + contact point
        grouped: dict[tuple[str, str], list[InputEvent]] = {}
        for event in events:
            key = (event.account_id, event.contact_point_ref)
            grouped.setdefault(key, []).append(event)

        candidates: list[VisitCandidate] = []
        for (account_id, ref), visits in grouped.items():
            health, recommendation, origination_review = _address_health(visits)
            window = _best_visit_window(visits)
            candidates.append(
                VisitCandidate(
                    account_id=account_id,
                    lender_id=scope or "unknown",
                    contact_point_ref=ref,
                    address_health=health,
                    best_visit_window=window,
                    recommendation=recommendation,
                    origination_review=origination_review,
                    location_confidence=None,
                )
            )

        env = envelope()
        return VisitCandidatesResponse(**env, candidates=candidates)

    @app.get("/v1/health", response_model=HealthResponse)
    async def health(
        _key: None = Depends(require_api_key),
    ) -> HealthResponse:
        """Data freshness, last batch time and staleness state."""
        now = datetime.now(timezone.utc)
        last_batch = app.state.last_batch_time
        last_score_time = None
        score_age_hours: float | None = None
        for _key, (as_of, _scores) in app.state.score_cache.items():
            last_score_time = as_of
            score_age_hours = (now - as_of).total_seconds() / 3600.0
            break

        stale = score_age_hours is not None and (
            score_age_hours >= cfg.max_score_age_hours
        )
        staleness_state = "stale" if stale else "fresh"

        return HealthResponse(
            status="ok",
            version=cfg.model_version,
            model_version=cfg.model_version,
            feature_snapshot_id=cfg.feature_snapshot_id,
            generated_at=now,
            valid_until=now + timedelta(hours=cfg.validity_hours),
            data_freshness={
                "last_score_time": last_score_time,
                "score_age_hours": score_age_hours,
                "stale": stale,
                "max_score_age_hours": cfg.max_score_age_hours,
            },
            last_batch_time=last_batch,
            staleness_state=staleness_state,
            suppression_version=app.state.suppression.version,
        )

    return app


# ---------------------------------------------------------------------------
# Address health helpers (phase 2 stub)
# ---------------------------------------------------------------------------

def _address_health(visits: list[InputEvent]) -> tuple[str, str, bool]:
    """Map field_visit outcomes to address health + recommendation.

    Returns (health, recommendation, origination_review).
    """
    outcomes: list[str] = []
    for visit in visits:
        outcome = getattr(visit.payload, "outcome", None)
        if outcome:
            outcomes.append(outcome)

    if not outcomes:
        return "unresolved", "manual_review", True

    latest = outcomes[-1]
    if latest == "met_borrower":
        return "occupied", "schedule_field_visit", False
    if latest == "met_third_party":
        return "occupied", "schedule_field_visit", False
    if latest == "nobody_of_that_name":
        return "absent", "change_visit_time", False
    if latest == "address_not_found":
        return "moved", "trace", False
    if latest == "locked_premises":
        return "unresolved", "manual_review", True

    return "unresolved", "manual_review", True


def _best_visit_window(visits: list[InputEvent]) -> str | None:
    """Derive the best visit window from historical visit times."""
    hours: list[int] = []
    for visit in visits:
        visit_time = getattr(visit.payload, "visit_time", None)
        if visit_time is not None:
            hours.append(visit_time.hour)
    if not hours:
        return None
    avg_hour = sum(hours) / len(hours)
    start = int(avg_hour)
    end = (start + 1) % 24
    return f"hour_{start:02d}-{end:02d}"


# Module-level app for uvicorn / smoke test
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
