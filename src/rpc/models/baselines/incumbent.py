"""Baseline (a): incumbent attempt-count rule.

Re-implements the incumbent policy as configured in configs/eval.yaml:
rank primary first, move on after k consecutive failures, trace after a fixed
number of total attempts. Produces both a score (for ranking) and an action
(one of continue / switch_contact_point / trace).

No learning, no hidden state. Uses only PIT attempt counters supplied by eval.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd

ACTION_CONTINUE = "continue"
ACTION_SWITCH = "switch_contact_point"
ACTION_TRACE = "trace"


class IncumbentRuleScorer:
    """Attempt-count rule. ``fit`` is a no-op (rule has no parameters to learn)."""

    name = "incumbent"

    def __init__(
        self,
        k_consecutive_failures: int = 3,
        max_attempts_trace: int = 6,
        primary_first: bool = True,
        base_score: float = 0.5,
    ):
        self.k = k_consecutive_failures
        self.max_attempts = max_attempts_trace
        self.primary_first = primary_first
        self.base_score = base_score
        self._features: pd.DataFrame | None = None

    def fit(self, features: pd.DataFrame, labels: pd.Series | None = None) -> "IncumbentRuleScorer":
        self._features = features.copy()
        return self

    def attach_features(self, features: pd.DataFrame) -> "IncumbentRuleScorer":
        """Eval supplies the PIT feature frame for the scoring origin."""
        self._features = features.copy()
        return self

    def _frame(self, refs: Sequence[str]) -> pd.DataFrame:
        feats = self._features
        base = pd.DataFrame({"contact_point_ref": list(refs)})
        if feats is None or feats.empty:
            base["consec_failures"] = 0.0
            base["n_attempts"] = 0.0
            base["is_primary"] = 0.0
            return base
        f = feats[feats["contact_point_ref"].isin(set(refs))]
        return base.merge(f, on="contact_point_ref", how="left").fillna(
            {"consec_failures": 0.0, "n_attempts": 0.0, "is_primary": 0.0}
        )

    def score(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        fr = self._frame(contact_point_refs)
        # Score decays with consecutive failures; primary gets a small bump.
        p = self.base_score / (1.0 + fr["consec_failures"].astype(float))
        if self.primary_first:
            p = p + 0.05 * fr["is_primary"].astype(float)
        return pd.DataFrame(
            {
                "contact_point_ref": fr["contact_point_ref"],
                "p_rpc": p.clip(0.01, 0.99),
                "confidence": 0.5,
            }
        )

    def decide(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        fr = self._frame(contact_point_refs)
        streak = fr["consec_failures"].astype(float)
        total = fr["n_attempts"].astype(float)
        action = pd.Series(ACTION_CONTINUE, index=fr.index)
        action[(streak >= self.k) & (total < self.max_attempts)] = ACTION_SWITCH
        action[total >= self.max_attempts] = ACTION_TRACE
        return pd.DataFrame({"contact_point_ref": fr["contact_point_ref"], "action": action})
