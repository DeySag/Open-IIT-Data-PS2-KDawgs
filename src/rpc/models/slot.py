"""Time-slot RPC model (workbook model 5, P1).

What it is:
    Per-attempt RPC broken down by time-of-day slot, shrunk toward the
    segment/global rate and expressed as a *multiplier* on the state
    tracker's ``p_rpc`` head (whose ``slot_multiplier_default`` is 1.0 —
    see ``configs/state_tracker.yaml``). Dialled attempts only; undialled
    links are never negatives (censoring rule, audit §4).

Label (per attempt, dialled only):
    ``is_rpc = answered AND disposition IN sanctioned rpc set``
    (defaults match ``configs/eval.yaml``; strict variant drops hung_up /
    refused). ``rpc_*`` on a non-answered network is quarantined, never
    a positive.

Point-in-time requirements:
    - ``fit`` drops every row with ``occurred_at > as_of`` (assumes
      ``received_at == occurred_at`` flagged assumption until ask-CN #1
      lands; late-event logic is unit-tested synthetically).
    - Slot bins + timezone come from ``configs/features.yaml``
      (``features.slots``, ``features.timezone``); naive timestamps are
      interpreted in that zone. A 5:30 shift moves every slot feature,
      so the zone is config, never a constant.
    - Banned/hidden columns (``verified_status``, ``true_state``,
      ``ground_truth`` and friends) are never read — predictions are
      identical whether or not they are present.

Units: rates are probabilities in [0, 1]; multipliers are unitless,
clipped to ``[clip_lo, clip_hi]`` from ``configs/slot.yaml``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Columns that must never influence the model, even if present.
BANNED_COLUMNS = frozenset({
    "ground_truth",
    "true_state",
    "true_avoidance",
    "true_borrower_avoidance",
    "shared_reason",
    "policy_action",
    "verified_status",
    "verified_holdout",
})

ANSWERED = ("answered",)

# Sanctioned canonical set; mirrors configs/eval.yaml labels.rpc_dispositions.
DEFAULT_RPC_DISPOSITIONS = ("RPC", "promise_to_pay", "callback", "dispute")
# Raw rpc_* variants accepted defensively (pipelines that pass raw through).
RAW_RPC = (
    "rpc_ptp",
    "rpc_call_back",
    "rpc_hung_up",
    "rpc_refused",
    "rpc_hardship",
    "rpc_dispute",
    "rpc_promise",
)
STRICT_EXCLUDE_CANONICAL = frozenset({"hung_up", "refused"})
STRICT_EXCLUDE_RAW = frozenset({"rpc_hung_up", "rpc_refused"})
OFF_HOURS = "off_hours"


def load_slot_params(
    slot_cfg: str | Path | None = None,
    features_cfg: str | Path | None = None,
) -> dict[str, Any]:
    """Load slot hyperparameters + bins/timezone (config over constants).

    Returns keys: smoothing_alpha, min_samples, clip_lo, clip_hi,
    segment_col, seed, slots (name -> [lo, hi)), timezone.
    """
    repo = Path(__file__).resolve().parents[3]
    scfg = Path(slot_cfg) if slot_cfg else repo / "configs" / "slot.yaml"
    fcfg = Path(features_cfg) if features_cfg else repo / "configs" / "features.yaml"
    with scfg.open(encoding="utf-8") as fh:
        s = yaml.safe_load(fh) or {}
    with fcfg.open(encoding="utf-8") as fh:
        f = (yaml.safe_load(fh) or {}).get("features", {})
    return {
        "smoothing_alpha": float(s.get("smoothing_alpha", 10.0)),
        "min_samples": int(s.get("min_samples", 30)),
        "clip_lo": float(s.get("clip_lo", 0.5)),
        "clip_hi": float(s.get("clip_hi", 2.0)),
        "segment_col": str(s.get("segment_col", "lender_id")),
        "seed": int(s.get("seed", 42)),
        "slots": dict(
            f.get("slots", {"morning": [8, 12], "afternoon": [12, 16], "evening": [16, 19]})
        ),
        "timezone": str(f.get("timezone", "Asia/Kolkata")),
    }


def to_slot(
    ts: pd.Timestamp | datetime | str,
    slots: dict[str, list[int]],
    timezone: str,
) -> str:
    """Bin one timestamp into a slot name (local time) or ``off_hours``.

    Naive timestamps are interpreted in ``timezone`` (flagged IST
    assumption); aware timestamps are converted.
    """
    t = pd.to_datetime(ts, utc=True)
    t = t.tz_localize("UTC").tz_convert(timezone) if t.tzinfo is None else t.tz_convert(timezone)
    hour = int(t.hour)
    for name, (lo, hi) in slots.items():
        if int(lo) <= hour < int(hi):
            return name
    return OFF_HOURS


def attempt_is_rpc(
    network_response: object,
    disposition: object,
    rpc_dispositions: tuple[str, ...] = DEFAULT_RPC_DISPOSITIONS,
    strict: bool = False,
) -> bool:
    """Per-attempt RPC: answered AND sanctioned disposition, never OR."""
    if str(network_response) not in ANSWERED:
        return False
    disp = str(disposition)
    allowed = set(rpc_dispositions) | set(RAW_RPC)
    if strict:
        allowed = allowed - STRICT_EXCLUDE_CANONICAL - STRICT_EXCLUDE_RAW
    return disp in allowed


class SlotRPCModel:
    """Smoothed per-(segment, slot) RPC rates + multipliers. Deterministic."""

    name = "slot_rpc"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or load_slot_params()
        self._rates: dict[tuple[str, str], float] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._global_rate = 0.0
        self._global_n = 0
        self.fitted_as_of: pd.Timestamp | None = None

    def fit(
        self,
        attempts: pd.DataFrame,
        as_of: datetime | pd.Timestamp,
        ts_col: str = "occurred_at",
        nr_col: str = "network_response",
        disp_col: str = "disposition",
    ) -> SlotRPCModel:
        """Fit on dialled attempts with ``occurred_at <= as_of`` (PIT).

        ``attempts`` needs timestamp + network_response + disposition columns
        (``attempt_ts`` accepted as an alias) and optionally the segment
        column. Banned columns are ignored. Rows after ``as_of`` are dropped.
        """
        p = self.params
        slots = p["slots"]
        tz = p["timezone"]
        seg_col = p["segment_col"]
        df = attempts.copy()
        if ts_col not in df.columns and "attempt_ts" in df.columns:
            ts_col = "attempt_ts"
        for banned in BANNED_COLUMNS:
            if banned in df.columns:
                df = df.drop(columns=[banned])
        df["_ts"] = pd.to_datetime(df[ts_col], utc=True)
        as_of_ts = pd.to_datetime(as_of, utc=True)
        df = df[df["_ts"] <= as_of_ts].copy()
        self.fitted_as_of = as_of_ts
        if df.empty:
            self._global_rate, self._global_n = 0.0, 0
            self._rates, self._counts = {}, {}
            return self
        df["_slot"] = [to_slot(t, slots, tz) for t in df["_ts"]]
        df["_rpc"] = [
            attempt_is_rpc(nr, d) for nr, d in zip(df[nr_col], df[disp_col])
        ]
        seg = df[seg_col].astype(str) if seg_col in df.columns else "all"
        df["_seg"] = seg if isinstance(seg, str) else seg.to_numpy()
        alpha = float(p["smoothing_alpha"])
        self._global_rate = float(df["_rpc"].mean())
        self._global_n = len(df)
        self._rates, self._counts = {}, {}
        for (sg, sl), g in df.groupby(["_seg", "_slot"]):
            k = float(g["_rpc"].sum())
            n = len(g)
            # Shrink toward the global rate; thin cells fall back later.
            self._rates[(str(sg), str(sl))] = (k + alpha * self._global_rate) / (n + alpha)
            self._counts[(str(sg), str(sl))] = n
        return self

    def multiplier(self, slot: str, segment: str = "all") -> float:
        """Unitless multiplier for one (segment, slot); 1.0 when thin/unseen."""
        p = self.params
        n = self._counts.get((str(segment), str(slot)))
        if n is None or n < int(p["min_samples"]):
            # Segment fallback: same slot pooled across segments is NOT
            # estimated — fall back to neutral rather than invent signal.
            return 1.0
        if self._global_rate <= 0:
            return 1.0
        m = self._rates[(str(segment), str(slot))] / self._global_rate
        return float(min(max(m, p["clip_lo"]), p["clip_hi"]))

    def multipliers(self, segment: str = "all") -> dict[str, float]:
        """All configured slots (+ off_hours) for one segment."""
        out = {}
        for name in [*list(self.params["slots"]), OFF_HOURS]:
            out[name] = self.multiplier(name, segment)
        return out

    def best_slot(self, segment: str = "all") -> str | None:
        """Slot with the highest multiplier, or None when all fall back."""
        ms = self.multipliers(segment)
        if all(m == 1.0 for m in ms.values()):
            return None
        return max(ms, key=lambda k: ms[k])

    def apply(self, p_rpc: float, slot: str, segment: str = "all") -> float:
        """Scale a base ``p_rpc`` for a slot; output stays in [0, 1]."""
        return float(min(max(float(p_rpc) * self.multiplier(slot, segment), 0.0), 1.0))

    def to_dict(self) -> dict[str, Any]:
        """Audit-log snapshot (params + fitted rates/counts)."""
        return {
            "params": {k: v for k, v in self.params.items()},
            "global_rate": self._global_rate,
            "global_n": self._global_n,
            "fitted_as_of": str(self.fitted_as_of),
            "rates": {f"{sg}|{sl}": r for (sg, sl), r in self._rates.items()},
            "counts": {f"{sg}|{sl}": n for (sg, sl), n in self._counts.items()},
        }


def attach_best_slot(
    scorer: Any,
    model: SlotRPCModel,
    refs: list[str],
    segment: str = "all",
) -> Any:
    """Attach each ref's best-slot multiplier to a tracker/scorer.

    The multiplier answers "how much better is this contact at its best
    time of day". Refs in thin/unseen cells attach 1.0 (neutral).
    Returns the scorer for chaining.
    """
    best = model.best_slot(segment)
    mults = (
        {r: model.multiplier(best, segment) for r in refs}
        if best is not None else {}
    )
    scorer.attach_slot_multipliers(mults)
    return scorer
