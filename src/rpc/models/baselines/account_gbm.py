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

    @staticmethod
    def _aggregate(fr: pd.DataFrame, acc: pd.Series) -> pd.DataFrame:
        """Account-grain features: per-column mean + max plus contact count.

        Mean carries the account's typical line; max carries its best line
        (the dialer's actual choice set); n_contacts carries choice-set size.
        Single shared helper so fit and score-time paths always agree.
        """
        num = fr.drop(columns=["_y"], errors="ignore").select_dtypes(include="number").copy()
        num["_acc"] = acc.to_numpy()
        grp = num.groupby("_acc")
        mean = grp.mean(numeric_only=True).add_suffix("_mean")
        mx = grp.max(numeric_only=True).add_suffix("_max")
        out = mean.join(mx)
        out["n_contacts"] = grp.size()
        return out.drop(columns=["_acc_mean", "_acc_max"], errors="ignore")

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
        agg = self._aggregate(fr_num, fr["_acc"])
        y_acc = fr.groupby("_acc")["_y"].max()  # account contacted if any line was
        self._global_mean = float(y_acc.mean()) if len(y_acc) else 0.5
        Xa = agg.fillna(0.0).to_numpy(dtype=float)
        self._agg_columns = list(agg.columns)
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
        raw = fr.drop(columns=["contact_point_ref", "_acc"], errors="ignore")
        try:
            agg = self._aggregate(raw, fr["_acc"]).reindex(columns=self._agg_columns)
            Xa = agg.fillna(0.0).to_numpy(dtype=float)
            ps = self._model.predict_proba(Xa)[:, 1]  # type: ignore[union-attr]
            out = {a: float(v) for a, v in zip(agg.index.tolist(), ps)}
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
