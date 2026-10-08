"""Baseline models - must exist and be beaten by our models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit


class AttemptCountBaseline:
    """Rule-based: contact if attempts < threshold, else trace."""

    def __init__(self, max_attempts: int = 10):
        self.max_attempts = max_attempts

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return P(RPC) based on attempt count."""
        attempts = features.get("attempt_count_total", pd.Series(0, index=features.index))
        # Simple decay: high prob when few attempts, low when many
        prob = np.clip(1.0 - attempts / self.max_attempts, 0.05, 0.95)
        return prob


class AccountLevelGBM:
    """Account-level GBM contactability score (baseline b)."""

    def __init__(self, **lgb_params):
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "random_state": 42,
            **lgb_params,
        }
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None):
        self.feature_names = list(X.columns)
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(X_val, label=y_val))

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X[self.feature_names])


class ContactPointGBM:
    """Per-contact-point GBM without state tracking (baseline c)."""

    def __init__(self, **lgb_params):
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "random_state": 42,
            **lgb_params,
        }
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None):
        self.feature_names = list(X.columns)
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(X_val, label=y_val))

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X[self.feature_names])


def time_aware_split(df: pd.DataFrame, time_col: str, n_splits: int = 5, test_size: float = 0.2):
    """Generate time-aware train/test splits (purged)."""
    df = df.sort_values(time_col)
    n = len(df)
    test_n = int(n * test_size)
    train_n = n - test_n

    # Purge gap (10% of test size)
    gap = int(test_n * 0.1)

    splits = []
    for i in range(n_splits):
        start = i * (train_n // n_splits)
        end = start + train_n // n_splits
        test_start = end + gap
        test_end = test_start + test_n // n_splits

        if test_end > n:
            break

        train_idx = np.arange(start, end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits


def save_model(model: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)