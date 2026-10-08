"""Baseline (b): account-level GBM contactability score.

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
        self._agg_columns: list[str] = []
        self._score_features: pd.DataFrame | None = None

    def set_params(self, params: dict) -> "AccountGBMScorer":
        """Replace hyperparameters (eval uses this for validation selection).

        Rebuilds the unfitted estimator; no data is touched.
        """
        self.params = dict(params)
        self._model = make_lgbm(self.params)
        return self

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
        # Drop eval-merged label columns before account aggregation: their
        # per-account means are the target by another name (same leak class
        # as _gbm_common._LABEL_COLUMNS, applied here because this path
        # aggregates instead of calling to_matrix).
        label_cols = [c for c in ("rpc_next_7d", "censored", "n_dials_window",
                                  "was_dialled_next_7d", "_y")
                      if c in fr.columns]
        fr_num = fr.drop(columns=["contact_point_ref", "_acc", *label_cols])
        agg = fr_num.groupby(fr["_acc"]).mean(numeric_only=True)
        y_acc = fr.groupby("_acc")["_y"].max()  # account contacted if any line was
        self._global_mean = float(y_acc.mean()) if len(y_acc) else 0.5
        Xa = agg.drop(columns=["_y"], errors="ignore").fillna(0.0).to_numpy(dtype=float)
        self._agg_columns = list(agg.drop(columns=["_y"], errors="ignore").columns)
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
        """Eval supplies cp -> account mapping (plus the scoring-origin PIT
        feature frame so unseen accounts can be scored by the fitted model)."""
        self._cp_account = dict(
            zip(account_ids["contact_point_ref"], account_ids["account_id"])
        )
        self._score_features = features.copy() if features is not None else None
        return self

    def _predict_unseen(self, accounts: list[str]) -> dict[str, float]:
        """Score accounts absent from training by applying the fitted LightGBM
        to their aggregated scoring-origin features (same aggregation as fit).

        Seen accounts keep their stored scores; only unseen ones flow here.
        """
        out: dict[str, float] = {}
        feats = self._score_features
        if self._model is None or feats is None or feats.empty or not self._agg_columns:
            return out
        fr = feats[feats["contact_point_ref"].isin(
            [r for r, a in self._cp_account.items() if a in set(accounts)]
        )].copy()
        if fr.empty:
            return out
        fr["_acc"] = fr["contact_point_ref"].map(self._cp_account)
        num = pd.DataFrame(index=fr.index)
        for c in self._agg_columns:
            num[c] = pd.to_numeric(fr[c], errors="coerce") if c in fr.columns else np.nan
        try:
            Xa = num.groupby(fr["_acc"]).mean(numeric_only=True).reindex(
                columns=self._agg_columns).fillna(0.0).to_numpy(dtype=float)
            accs = num.groupby(fr["_acc"]).mean(numeric_only=True).index.tolist()
            ps = self._model.predict_proba(Xa)[:, 1]  # type: ignore[union-attr]
            out = {a: float(v) for a, v in zip(accs, ps)}
        except Exception:
            out = {}
        return out

    def score(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        refs = list(contact_point_refs)
        unseen = sorted({self._cp_account.get(r, "") for r in refs} - set(self._account_score))
        fresh = self._predict_unseen([a for a in unseen if a and a != "UNK"])
        p = [
            float(self._account_score.get(
                self._cp_account.get(r, ""),
                fresh.get(self._cp_account.get(r, ""), self._global_mean)))
            for r in refs
        ]
        return pd.DataFrame(
            {"contact_point_ref": refs, "p_rpc": np.clip(p, 0.01, 0.99), "confidence": 0.6}
        )
