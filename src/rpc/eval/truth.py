"""Hidden-ground-truth join for evaluation only (simulation-only).

This module exists so model evaluation can compare observed labels against
the simulator's hidden state. It must never be imported from
``src/rpc/features/`` (asserted in tests).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ground_truth(data_dir: str | Path) -> pd.DataFrame | None:
    """Load hidden ground truth for evaluation only.

    Returns None when the table is absent (e.g. simulator v0 emits none).
    """
    path = Path(data_dir) / "ground_truth.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def join_truth(
    labels: pd.DataFrame, truth: pd.DataFrame, on: list[str] | None = None
) -> pd.DataFrame:
    """Join observed labels to hidden ground truth for eval-side analysis."""
    on = on or ["lender_id", "borrower_id", "contact_point_ref"]
    shared = [c for c in on if c in truth.columns]
    if not shared:
        raise ValueError("Ground-truth table shares no join keys with labels")
    return labels.merge(truth, on=shared, how="left", suffixes=("", "_truth"))
