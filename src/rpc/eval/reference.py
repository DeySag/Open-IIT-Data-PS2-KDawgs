"""Reference scorers for harness diagnostics (eval-only, simulation-only).

- OracleScorer reads GROUND TRUTH and must get near-perfect scores; it calibrates
  the harness, never a model. Lives here (not in baselines/) so the leakage guard
  (baselines never read ground truth) stays trivially true.
- RandomScorer must get ~0.5 AUC.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd


class OracleScorer:
    """Perfect foresight from ground truth (harness diagnostic only)."""

    name = "oracle"

    def __init__(self, ground_truth: pd.DataFrame, reachable_states: Sequence[str] = ("valid_reachable",)):
        self._gt = ground_truth.set_index("contact_point_ref")["true_state"].to_dict()
        self._reach = set(reachable_states)

    def score(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        rows = [
            {
                "contact_point_ref": r,
                "p_rpc": 1.0 if self._gt.get(r) in self._reach else 0.0,
                "confidence": 1.0,
            }
            for r in contact_point_refs
        ]
        return pd.DataFrame(rows)


class RandomScorer:
    """Uniform random scores (harness diagnostic only)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def score(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "contact_point_ref": list(contact_point_refs),
                "p_rpc": self._rng.uniform(0, 1, len(list(contact_point_refs))),
                "confidence": 0.5,
            }
        )
