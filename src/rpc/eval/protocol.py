"""Scorer protocol: the single plug-in interface for baselines and models.

Any model (baseline or the state tracker built by another workstream) plugs in
through this protocol. Eval calls ``score``; training is the scorer's own
concern (baselines expose ``fit``; see src/rpc/models/baselines/).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

# Optional state-posterior columns a scorer may return.
STATE_COLUMNS = (
    "valid_reachable",
    "avoiding",
    "temp_unreachable",
    "switched_off_long",
    "recycled",
    "third_party",
    "invalid",
)


@runtime_checkable
class Scorer(Protocol):
    """Scores contact points as of a timestamp. Point-in-time only."""

    name: str

    def score(
        self, as_of: datetime, contact_point_refs: Sequence[str]
    ) -> pd.DataFrame:
        """Return one row per ref with columns:

        - contact_point_ref (str)
        - p_rpc (float in [0, 1])
        - optional: any of STATE_COLUMNS (posterior probs), recycled_risk, confidence.
        """
        ...
