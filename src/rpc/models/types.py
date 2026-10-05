"""Shared model types (FROZEN interface).

Other workstreams import from here. Do not change field names, key sets, or
semantics without notifying the coordinator — downstream consumers (decision
layer, serving, eval harness) depend on this exact shape.

model estimates: all probabilities produced by models using these types are
estimates that require refit on issued data; never present them as measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# Exact output state keys. Every ContactPointScore.state_posterior must contain
# precisely these keys, summing to 1.
STATE_KEYS: tuple[str, ...] = (
    "valid_reachable",
    "avoiding",
    "temp_unreachable",
    "switched_off_long",
    "recycled",
    "third_party",
    "invalid",
)

ContactPointType = Literal["phone", "address"]


@dataclass
class ContactPointScore:
    """Per-contact-point health score.

    Attributes:
        contact_point_ref: hash of the normalised contact point (A2).
        type: "phone" or "address".
        as_of: point-in-time scoring timestamp; only events with
            received_at <= as_of may influence the score.
        state_posterior: dict with exactly the 7 STATE_KEYS, summing to 1.
            `avoiding` = P(line valid AND borrower avoiding);
            `valid_reachable` = P(line valid AND borrower not avoiding).
        p_rpc: probability the next dial at a typical slot is answered by the
            borrower (right-party contact).
        recycled_risk: P(line recycled to a new subscriber).
        confidence: 0-1; falls as evidence ages (entropy + recency decay).
    """

    contact_point_ref: str
    type: str  # "phone" | "address"
    as_of: datetime
    state_posterior: dict[str, float]
    p_rpc: float
    recycled_risk: float
    confidence: float

    def __post_init__(self) -> None:
        if self.type not in ("phone", "address"):
            raise ValueError(f"type must be 'phone' or 'address', got {self.type!r}")
        keys = set(self.state_posterior.keys())
        if keys != set(STATE_KEYS):
            raise ValueError(f"state_posterior keys must be exactly {list(STATE_KEYS)}, got {sorted(keys)}")
        vals = list(self.state_posterior.values())
        if any(v != v or v < 0.0 for v in vals):  # NaN check via v != v
            raise ValueError("state_posterior values must be finite and non-negative")
        total = sum(vals)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"state_posterior must sum to 1, got {total}")
        for name, v in (("p_rpc", self.p_rpc), ("recycled_risk", self.recycled_risk), ("confidence", self.confidence)):
            if not (0.0 <= v <= 1.0) or v != v:
                raise ValueError(f"{name} must be in [0, 1], got {v}")


# Backwards-compatible alias for the eval-harness DataFrame adapter.
SCORE_COLUMNS: tuple[str, ...] = (
    "contact_point_ref",
    "p_rpc",
    "recycled_risk",
    "confidence",
    "state_posterior",
    *[f"sp_{k}" for k in STATE_KEYS],
)
