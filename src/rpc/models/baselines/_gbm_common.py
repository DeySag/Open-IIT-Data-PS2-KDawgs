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

# Key/metadata columns that are never model inputs (kept here so to_matrix
# stays dependency-free instead of importing the feature spec).
_NON_FEATURE_COLUMNS = frozenset([
    "lender_id",
    "borrower_id",
    "account_id",
    "contact_point_ref",
    "as_of",
    "feature_snapshot_id",
    "event_watermark",
])


def to_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Select model columns: legacy mini columns when present, else every
    numeric/boolean registry column (keys, metadata and strings excluded).

    The legacy list covers the eval mini-features contract; the generic
    fallback lets the scorer train on the full PIT registry output.
    """
    legacy = [c for c in NUM_COLS if c in features.columns]
    has_registry_cols = any(
        c not in _NON_FEATURE_COLUMNS and c not in NUM_COLS for c in features.columns
    )
    if legacy and not has_registry_cols:
        cols = legacy
    else:
        cols = [
            c for c in features.columns
            if c not in _NON_FEATURE_COLUMNS
            and (
                pd.api.types.is_any_real_numeric_dtype(features[c].dtype)
                or pd.api.types.is_bool_dtype(features[c].dtype)
            )
        ]
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
