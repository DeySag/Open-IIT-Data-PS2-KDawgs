"""Selection-bias awareness: inverse-propensity weighting via the policy log.

Metrics are ALWAYS reported on dialled-only data (flagged in the report).
As an optional second view, when ``data/policy_log.parquet`` exists, we fit a
simple propensity model P(dialled | features) and re-weight dialled outcomes by
1/p to approximate the all-contact-point population. Eval-only reader of the
policy log; models must never touch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.rpc.eval import metrics as M


def fit_propensity(
    features: pd.DataFrame, dialled: np.ndarray | pd.Series, seed: int = 42
) -> LogisticRegression:
    X = features.select_dtypes(include=[np.number]).fillna(0.0)
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X.to_numpy(), np.asarray(dialled).astype(int))
    return clf


def propensity_weights(
    clf: LogisticRegression,
    features: pd.DataFrame,
    min_prob: float = 0.05,
    max_prob: float = 0.95,
) -> np.ndarray:
    X = features.select_dtypes(include=[np.number]).fillna(0.0)
    p = np.clip(clf.predict_proba(X.to_numpy())[:, 1], min_prob, max_prob)
    return 1.0 / p


def weighted_brier(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p, w = y[m], p[m], w[m]
    if len(y) == 0:
        return float("nan")
    return float((w * (y - np.clip(p, 0, 1)) ** 2).sum() / w.sum())


def ipw_view(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Second-view metrics under IPW weights (dialled-only point estimates + IPW)."""
    out: dict[str, float] = {
        "auc_dialled": M.roc_auc(y, p),
        "brier_dialled": M.brier(y, p),
        "brier_ipw": weighted_brier(y, p, w),
    }
    return out
