"""Baseline (b): account-level GBM contactability score (simulation-only).

Aggregates per-contact-point PIT features to account level, fits one
LightGBM, and assigns every contact point its account's score. No state
tracking, no cross-line latent -- the point is to show what pooled
account history alone buys.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from src.rpc.models.baselines._gbm_common import make_lgbm, to_matrix


class AccountGBMScorer:
    name = "account_gbm"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self._model: object = make_lgbm(self.params)
        self._account_score: dict[str, float] = {}
        self._global_mean = 0.5
        self._cp_account: dict[str, str] = {}

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        account_ids: pd.Series,
    ) -> "AccountGBMScorer":
        fr = features.copy()
        fr["_y"] = np.asarray(labels, dtype=float)
        fr["_acc"] = np.asarray(account_ids)
        self._cp_account = dict(zip(fr["contact_point_ref"], fr["_acc"]))
        X_cp, _ = to_matrix(fr)
        fr_num = fr.drop(columns=["contact_point_ref", "_acc"])
        agg = fr_num.groupby(fr["_acc"]).mean(numeric_only=True)
        y_acc = fr.groupby("_acc")["_y"].max()  # account contacted if any line was
        self._global_mean = float(y_acc.mean()) if len(y_acc) else 0.5
        Xa = agg.drop(columns=["_y"], errors="ignore").fillna(0.0).to_numpy(dtype=float)
        if len(agg) >= 2 and y_acc.nunique() >= 2:
            self._model.fit(Xa, y_acc.to_numpy())  # type: ignore[union-attr]
        else:  # degenerate: fall back to account mean rate
            self._model = None  # type: ignore[assignment]
        self._account_score = {}
        if self._model is not None:
            try:
                ps = self._model.predict_proba(Xa)[:, 1]  # type: ignore[union-attr]
                self._account_score = dict(zip(agg.index, ps))
            except Exception:
                self._model = None
        return self

    def attach_context(
        self, account_ids: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> "AccountGBMScorer":
        """Eval supplies cp -> account mapping for the scoring origin."""
        self._cp_account = dict(
            zip(account_ids["contact_point_ref"], account_ids["account_id"])
        )
        return self

    def score(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        refs = list(contact_point_refs)
        p = [
            float(self._account_score.get(self._cp_account.get(r, ""), self._global_mean))
            for r in refs
        ]
        return pd.DataFrame(
            {"contact_point_ref": refs, "p_rpc": np.clip(p, 0.01, 0.99), "confidence": 0.6}
        )
