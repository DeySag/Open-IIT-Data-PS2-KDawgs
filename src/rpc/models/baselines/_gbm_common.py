"""Shared LightGBM helpers for the GBM baselines (no state tracking)."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUM_COLS = [
    "n_attempts",
    "n_answered",
    "answer_rate",
    "n_recent_7d",
    "n_recent_30d",
    "consec_failures",
    "days_since_last_attempt",
    "days_since_last_answer",
    "avg_ring_seconds",
    "n_switched_off",
    "n_not_reachable",
    "n_does_not_exist",
    "n_no_answer",
    "n_immediate_hangup",
    "is_primary",
]


def to_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in NUM_COLS if c in features.columns]
    X = features[cols].copy()
    for c in X.columns:
        # The real feature layer emits nullable boolean columns (e.g.
        # is_primary); coerce to float so NaN handling below is dtype-safe.
        if pd.api.types.is_bool_dtype(X[c].dtype):
            X[c] = X[c].astype("float64")
    # Recency NaN = "never happened": far past. Sentinel keeps it learnable.
    for c in ("days_since_last_attempt", "days_since_last_answer"):
        if c in X.columns:
            X[c] = X[c].fillna(9999.0)
    X = X.fillna(0.0).to_numpy(dtype=float)
    return X, cols


def make_lgbm(params: dict) -> object:
    """LightGBM if installed, else sklearn GradientBoosting fallback (same interface)."""
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            min_child_samples=params.get("min_child_samples", 20),
            random_state=params.get("seed", 42),
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=params.get("seed", 42))
