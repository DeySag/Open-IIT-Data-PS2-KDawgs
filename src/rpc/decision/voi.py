"""Skip-trace VOI (expected-value calculation, not a learned model).

VOI = P(find valid contact | trace)
      x [P(recovery | reached) - P(recovery | not reached)]
      x recoverable_amount
      - trace_cost - expected_collection_cost - compliance_and_goodwill_cost

All monetary inputs come from ``configs/costs.yaml``; every number there is
an assumption for CN/SMEs to confirm (simulation-only).

Key semantics (per System Prompt):
- Value is incremental (self-cure accounts gain nothing from a trace).
- recoverable_amount is principal + lawful interest net of penal-charge
  restrictions with a settlement haircut, discounted -- not headline outstanding.
- Condition on P(invalid or dead), never on P(no contact): avoiding
  borrowers get near-zero VOI because tracing finds nothing new.
- Ranking is greedy by VOI-per-rupee under the portfolio budget (a heuristic
  for the 0/1 knapsack, documented below -- not exact).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.rpc.decision.types import RankedTrace, TraceCandidate

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COSTS_PATH = REPO_ROOT / "configs" / "costs.yaml"


def load_costs_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_COSTS_PATH) as f:
        return yaml.safe_load(f)["costs"]


def _bucket_key(bucket: str) -> str:
    return str(bucket)


def recoverable_amount(
    outstanding: float,
    dpd_bucket: str,
    secured: bool,
    costs: dict,
) -> float:
    """Recoverable principal + lawful interest net of penal-charge caps and
    settlement haircut, discounted to present value. Units: INR."""
    rec = costs.get("recoverable", {})
    bucket_mult = float(rec.get("dpd_multipliers", {}).get(_bucket_key(dpd_bucket), 0.5))
    sec_mult = float(rec.get("secured_multiplier", 1.0)) if secured else float(rec.get("unsecured_multiplier", 0.6))
    haircut = float(costs.get("settlement", {}).get("haircut", 0.2))
    rate = float(rec.get("discount_rate", 0.12))
    lag_years = float(costs.get("settlement", {}).get("collection_lag_years", 0.5))
    gross = max(0.0, float(outstanding)) * bucket_mult * sec_mult * (1.0 - haircut)
    return gross / ((1.0 + rate) ** lag_years)


def recovery_gain(dpd_bucket: str, product: str, costs: dict) -> float:
    """Incremental recovery probability: P(recovery|reached) - P(recovery|not reached)."""
    curves = costs.get("recovery_curves", {})
    buckets = curves.get("by_bucket", {})
    b = buckets.get(_bucket_key(dpd_bucket), {"reached": 0.25, "not_reached": 0.02})
    prod_mult = float(curves.get("product_multiplier", {}).get(product, 1.0))
    gain = (float(b.get("reached", 0.25)) - float(b.get("not_reached", 0.02))) * prod_mult
    return max(0.0, gain)


def trace_cost_for(method: str, costs: dict, override: float | None = None) -> float:
    if override is not None:
        return max(0.0, float(override))
    return float(costs.get("skip_trace", {}).get(method, {}).get("cost_per_trace", 100.0))


def collection_and_goodwill_cost(costs: dict) -> tuple[float, float]:
    voi_cfg = costs.get("voi_costs", {})
    return (
        float(voi_cfg.get("expected_collection_cost", 60.0)),
        float(voi_cfg.get("expected_goodwill_cost_per_trace", 25.0)),
    )


def trace_success_rate(method: str, costs: dict) -> float:
    return float(costs.get("skip_trace", {}).get(method, {}).get("success_rate", 0.25))


def compute_voi(
    p_dead: float,
    dpd_bucket: str,
    product: str,
    secured: bool,
    outstanding: float,
    trace_method: str,
    costs: dict,
    trace_cost_override: float | None = None,
) -> dict[str, float]:
    """Full VOI decomposition for one account/method. ``p_dead`` must be
    P(invalid or dead) -- never P(no contact) -- so avoiding accounts get
    near-zero VOI."""
    p_dead = min(1.0, max(0.0, float(p_dead)))
    p_find = trace_success_rate(trace_method, costs) * p_dead
    gain = recovery_gain(dpd_bucket, product, costs)
    rec_amt = recoverable_amount(outstanding, dpd_bucket, secured, costs)
    t_cost = trace_cost_for(trace_method, costs, trace_cost_override)
    coll_cost, gw_cost = collection_and_goodwill_cost(costs)
    voi = p_find * gain * rec_amt - t_cost - coll_cost - gw_cost
    voi_per_rupee = (voi / t_cost) if t_cost > 0 else (float("inf") if voi > 0 else 0.0)
    return {
        "p_find": p_find,
        "recovery_gain": gain,
        "recoverable_amount": rec_amt,
        "trace_cost": t_cost,
        "collection_cost": coll_cost,
        "goodwill_cost": gw_cost,
        "voi": voi,
        "voi_per_rupee": voi_per_rupee,
    }


def should_defer(voi_now: float, ev_wait: float, costs: dict) -> bool:
    """Defer ('trace later' = continue with backoff) when the expected value
    of waiting one cycle exceeds tracing now. Both inputs in INR."""
    rate = float(costs.get("recoverable", {}).get("discount_rate", 0.12))
    defer_days = float(costs.get("deferral", {}).get("deferral_days", 7))
    discounted_wait = float(ev_wait) / ((1.0 + rate) ** (defer_days / 365.0))
    return discounted_wait > float(voi_now)


@dataclass
class _ScoredCandidate:
    candidate: TraceCandidate
    voi: float
    voi_per_rupee: float
    cost: float
    recoverable: float


def rank_trace(
    candidates: list[TraceCandidate],
    budget: float,
    costs: dict | None = None,
) -> list[RankedTrace]:
    """Rank skip-trace candidates by VOI per rupee under a portfolio budget.

    - Ineligible candidates (guardrail-suppressed, trace_pending) and
      below-threshold VOI per rupee are never selected.
    - Selection is greedy by ratio: take in descending voi_per_rupee order
      while the cumulative cost fits. EXACTNESS NOTE: greedy-by-ratio is the
      exact optimum for the *fractional* knapsack but only a heuristic for
      this 0/1 (whole-trace) knapsack; a DP would be exact but is out of
      scope for day-1 scale. The gap is bounded and documented in
      docs/decision.md.
    - Returns every candidate ordered by voi_per_rupee (selected first),
      with ``within_budget`` and 1-based ``rank`` for selected ones.
    """
    cfg = costs or load_costs_config()
    voi_cfg = cfg.get("voi", {})
    min_ratio = float(voi_cfg.get("min_voi_per_rupee", 0.5))

    scored: list[_ScoredCandidate] = []
    for c in candidates:
        if not c.eligible:
            scored.append(_ScoredCandidate(c, voi=float("-inf"), voi_per_rupee=float("-inf"),
                                           cost=trace_cost_for(c.trace_method, cfg, c.trace_cost),
                                           recoverable=recoverable_amount(
                                               c.outstanding, c.dpd_bucket, c.secured, cfg)))
            continue
        d = compute_voi(c.p_dead, c.dpd_bucket, c.product, c.secured,
                        c.outstanding, c.trace_method, cfg, c.trace_cost)
        scored.append(_ScoredCandidate(c, d["voi"], d["voi_per_rupee"], d["trace_cost"], d["recoverable_amount"]))

    scored.sort(key=lambda s: s.voi_per_rupee, reverse=True)

    selected: set[str] = set()
    spent = 0.0
    for s in scored:
        if not s.candidate.eligible:
            continue
        if s.voi_per_rupee < min_ratio or s.voi <= 0:
            continue
        if spent + s.cost <= float(budget):
            selected.add(s.candidate.account_id)
            spent += s.cost

    out: list[RankedTrace] = []
    rank = 0
    for s in scored:
        is_in = s.candidate.account_id in selected
        if is_in:
            rank += 1
        out.append(RankedTrace(
            account_id=s.candidate.account_id,
            lender_id=s.candidate.lender_id,
            trace_method=s.candidate.trace_method,
            voi=s.voi,
            voi_per_rupee=s.voi_per_rupee,
            cost=s.cost,
            recoverable_amount=s.recoverable,
            within_budget=is_in,
            rank=rank if is_in else 0,
            deferred=False,
        ))
    return out
