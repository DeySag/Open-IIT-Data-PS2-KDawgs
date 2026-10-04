"""Stub score generator: representative ContactPointScore sets per state.

Lets the decision layer run end-to-end without the state-tracker / slot /
recycled models (workstreams B/C). All values are illustrative
(simulation-only) and carry stub evidence ids.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from src.rpc.decision.types import AccountContext, AccountFlags, ContactPointScore, TraceCandidate

_POSTERIORS: dict[str, dict[str, float]] = {
    "valid_reachable": {"valid_reachable": 0.70, "avoiding": 0.08, "temp_unreachable": 0.08,
                        "switched_off_long": 0.05, "recycled": 0.02, "third_party": 0.03, "invalid": 0.04},
    "avoiding": {"valid_reachable": 0.10, "avoiding": 0.65, "temp_unreachable": 0.08,
                 "switched_off_long": 0.07, "recycled": 0.02, "third_party": 0.05, "invalid": 0.03},
    "temp_unreachable": {"valid_reachable": 0.12, "avoiding": 0.06, "temp_unreachable": 0.60,
                         "switched_off_long": 0.12, "recycled": 0.02, "third_party": 0.03, "invalid": 0.05},
    "switched_off_long": {"valid_reachable": 0.04, "avoiding": 0.04, "temp_unreachable": 0.08,
                          "switched_off_long": 0.65, "recycled": 0.08, "third_party": 0.03, "invalid": 0.08},
    "recycled": {"valid_reachable": 0.02, "avoiding": 0.02, "temp_unreachable": 0.04,
                 "switched_off_long": 0.08, "recycled": 0.70, "third_party": 0.06, "invalid": 0.08},
    "third_party": {"valid_reachable": 0.08, "avoiding": 0.05, "temp_unreachable": 0.05,
                    "switched_off_long": 0.05, "recycled": 0.03, "third_party": 0.65, "invalid": 0.09},
    "invalid": {"valid_reachable": 0.01, "avoiding": 0.01, "temp_unreachable": 0.03,
                "switched_off_long": 0.08, "recycled": 0.07, "third_party": 0.05, "invalid": 0.75},
}

_P_RPC: dict[str, float] = {
    "valid_reachable": 0.75, "avoiding": 0.15, "temp_unreachable": 0.30,
    "switched_off_long": 0.08, "recycled": 0.05, "third_party": 0.10, "invalid": 0.02,
}

_RECYCLED_RISK: dict[str, float] = {
    # Calibrated P(recycled) from the cost-sensitive classifier. Non-recycled
    # states carry only background risk below the 1/(1+100) cost-ratio cutoff;
    # only a genuinely recycled line trips suppression.
    "valid_reachable": 0.003, "avoiding": 0.004, "temp_unreachable": 0.006,
    "switched_off_long": 0.008, "recycled": 0.80, "third_party": 0.004, "invalid": 0.006,
}

_CONFIDENCE: dict[str, float] = {
    "valid_reachable": 0.85, "avoiding": 0.70, "temp_unreachable": 0.65,
    "switched_off_long": 0.70, "recycled": 0.75, "third_party": 0.70, "invalid": 0.80,
}


def stub_score(
    ref: str,
    state: str,
    ctype: str = "phone",
    now: datetime | None = None,
    with_evidence: bool = True,
) -> ContactPointScore:
    """One representative score for ``state`` (a 7 phone-state key)."""
    ts = now or datetime.now(timezone.utc)
    return ContactPointScore(
        contact_point_ref=ref,
        type=ctype,
        as_of=ts,
        state_posterior=dict(_POSTERIORS[state]),
        p_rpc=_P_RPC[state],
        recycled_risk=_RECYCLED_RISK[state],
        confidence=_CONFIDENCE[state],
        evidence_ids=(uuid5(NAMESPACE_URL, f"stub-evidence:{ref}"),) if with_evidence else (),
    )


def stub_scores(
    states: list[str],
    prefix: str = "cp",
    now: datetime | None = None,
) -> list[ContactPointScore]:
    """A representative score set, one contact point per state in ``states``."""
    return [stub_score(f"{prefix}_{i}_{s}", s, now=now) for i, s in enumerate(states)]


def stub_context(
    now: datetime | None = None,
    **overrides: object,
) -> AccountContext:
    """A healthy baseline account context; override any field by keyword."""
    base: dict[str, object] = {
        "account_id": "ACC_STUB_001",
        "lender_id": "LENDER_001",
        "borrower_id": "BORR_STUB_001",
        "dpd_bucket": "31-60",
        "product": "unsecured_retail",
        "secured": False,
        "outstanding": 50000.0,
        "now": now or datetime.now(timezone.utc),
        "flags": AccountFlags(),
        "suppressed_refs": set(),
        "attempts_today": 0,
        "attempts_week": 0,
        "trace_pending": False,
        "whatsapp_opt_in": True,
        "address_state": None,
    }
    base.update(overrides)
    return AccountContext(**base)  # type: ignore[arg-type]


def stub_trace_candidate(
    account_id: str = "ACC_STUB_001",
    p_dead: float = 0.8,
    **overrides: object,
) -> TraceCandidate:
    base: dict[str, object] = {
        "account_id": account_id,
        "lender_id": "LENDER_001",
        "borrower_id": "BORR_STUB_001",
        "dpd_bucket": "31-60",
        "product": "unsecured_retail",
        "secured": False,
        "outstanding": 50000.0,
        "p_dead": p_dead,
        "trace_method": "digital",
        "trace_cost": None,
        "eligible": True,
        "deferral_ev_wait": 0.0,
    }
    base.update(overrides)
    return TraceCandidate(**base)  # type: ignore[arg-type]
