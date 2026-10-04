"""Decision-layer dataclasses (inputs/outputs of the decision layer).

These are intentionally plain dataclasses (not pydantic): the frozen
``src.rpc.contracts`` models remain the serving boundary. Converters in
:mod:`src.rpc.decision.actions` map these to the contract models.

``ContactPointScore`` mirrors ``src/rpc/models/types.py`` (which does not
exist yet on main, so this local definition is authoritative until
workstream B/C lands it; keep field names identical).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

PHONE_STATES: tuple[str, ...] = (
    "valid_reachable",
    "avoiding",
    "temp_unreachable",
    "switched_off_long",
    "recycled",
    "third_party",
    "invalid",
)

DEAD_STATES: frozenset[str] = frozenset({"switched_off_long", "invalid"})
CALLABLE_STATES: frozenset[str] = frozenset({"valid_reachable", "temp_unreachable"})


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, v))


def normalise_posterior(posterior: dict[str, float] | None) -> dict[str, float]:
    """Return a clean 7-key posterior: unknown keys dropped, missing keys
    filled with 0.0, negatives clamped, renormalised when mass > 0.

    Lenient by design: the decision layer must never crash on a malformed
    model output; it must still emit exactly one action with a reason code.
    """
    clean = {k: 0.0 for k in PHONE_STATES}
    for k, v in (posterior or {}).items():
        if k in clean:
            clean[k] = _clamp01(v)
    total = sum(clean.values())
    if total > 0:
        clean = {k: v / total for k, v in clean.items()}
    return clean


@dataclass
class ContactPointScore:
    """Per-contact-point model output consumed by the decision layer."""

    contact_point_ref: str
    type: str  # "phone" | "address"
    as_of: datetime
    state_posterior: dict[str, float]  # 7 phone-state keys
    p_rpc: float
    recycled_risk: float
    confidence: float
    evidence_ids: tuple[UUID, ...] = ()
    address_state: str | None = None  # address contact points only (phase-2 stub)

    def __post_init__(self) -> None:
        self.state_posterior = normalise_posterior(self.state_posterior)
        self.p_rpc = _clamp01(self.p_rpc)
        self.recycled_risk = _clamp01(self.recycled_risk)
        self.confidence = _clamp01(self.confidence)

    @property
    def dominant_state(self) -> str:
        return max(self.state_posterior, key=self.state_posterior.get)  # type: ignore[arg-type]

    @property
    def p_dead(self) -> float:
        """P(contact is invalid or long-dead). Excludes recycled mass: a
        recycled line says nothing about whether the borrower is findable,
        and recycled-only accounts must never trace (suppression instead)."""
        return self.state_posterior["invalid"] + self.state_posterior["switched_off_long"]

    @property
    def third_party_mass(self) -> float:
        return self.state_posterior["third_party"]


@dataclass
class AccountFlags:
    dispute: bool = False
    no_consent: bool = False
    deceased_or_insolvent: bool = False
    dnd: bool = False
    legal_case: bool = False


@dataclass
class AccountContext:
    account_id: str
    lender_id: str
    borrower_id: str
    dpd_bucket: str  # X | 1-30 | 31-60 | 61-90 | 90+
    product: str
    secured: bool
    outstanding: float
    now: datetime  # timezone-aware; guardrails interpret it in Asia/Kolkata
    flags: AccountFlags = field(default_factory=AccountFlags)
    suppressed_refs: set[str] = field(default_factory=set)
    attempts_today: int = 0
    attempts_week: int = 0
    trace_pending: bool = False
    whatsapp_opt_in: bool = False
    address_state: str | None = None  # phase-2 stub: valid_occupied | valid_absent | moved | hard_to_find | fabricated | incomplete


@dataclass
class TraceCandidate:
    """One account considered for the skip-trace queue."""

    account_id: str
    lender_id: str
    borrower_id: str
    dpd_bucket: str
    product: str
    secured: bool
    outstanding: float
    p_dead: float  # P(invalid or dead); avoiding mass must NOT inflate this
    trace_method: str = "digital"  # digital | physical | bureau
    trace_cost: float | None = None  # override; else costs.yaml per-method cost
    eligible: bool = True  # False when guardrails suppress or trace_pending
    deferral_ev_wait: float = 0.0  # expected value of waiting one cycle (Rs)


@dataclass
class RankedTrace:
    account_id: str
    lender_id: str
    trace_method: str
    voi: float
    voi_per_rupee: float
    cost: float
    recoverable_amount: float
    within_budget: bool
    rank: int  # 1-based among selected; 0 when not selected
    deferred: bool = False
