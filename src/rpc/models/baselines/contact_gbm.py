"""Baseline (c): per-contact-point GBM without state tracking (simulation-only).

Plain LightGBM on PIT per-contact-point features. Strong on recent-answer
patterns; has no notion of latent states, evidence decay, or borrower-level
sharing -- that is what the state tracker must beat.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from src.rpc.models.baselines._gbm_common import make_lgbm, to_matrix


class ContactGBMScorer:
    name = "contact_gbm"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self._model: object = make_lgbm(self.params)
        self._base_rate = 0.5
        self._features: pd.DataFrame | None = None

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "ContactGBMScorer":
        X, _ = to_matrix(features)
        y = np.asarray(labels, dtype=float)
        m = ~np.isnan(y)
        self._base_rate = float(y[m].mean()) if m.any() else 0.5
        if m.sum() >= 2 and np.unique(y[m]).size >= 2:
            self._model.fit(X[m], y[m])  # type: ignore[union-attr]
        else:
            self._model = None  # type: ignore[assignment]
        return self

    def attach_features(self, features: pd.DataFrame) -> "ContactGBMScorer":
        self._features = features.copy()
        return self

    def score(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        refs = list(contact_point_refs)
        if self._model is None or self._features is None:
            return pd.DataFrame(
                {"contact_point_ref": refs, "p_rpc": self._base_rate, "confidence": 0.5}
            )
        fr = pd.DataFrame({"contact_point_ref": refs}).merge(
            self._features, on="contact_point_ref", how="left"
        )
        # No blanket fillna here: to_matrix() handles NaN per column
        # (recency NaN -> far-past sentinel, rest -> 0), and a blanket
        # fillna(0.0) raises on the real layer's nullable boolean columns.
        X, _ = to_matrix(fr)
        try:
            p = self._model.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        except Exception:
            p = np.full(len(refs), self._base_rate)
        return pd.DataFrame(
            {"contact_point_ref": refs, "p_rpc": np.clip(p, 0.01, 0.99), "confidence": 0.6}
        )
