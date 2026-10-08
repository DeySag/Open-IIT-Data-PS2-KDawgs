"""Trace-conditional recovery model (workbook model 9, P1).

What it is:
    P(payment within 30d after ``trace_date`` | pre-trace history) for traced
    accounts. A GBM over caller-supplied PIT dial-history features, fit on
    uncensored traces only. Its output feeds VOI as the trace-conditional
    leg; the incremental gain still subtracts the self-cure baseline
    (``scale_recovery_gain`` in ``src/rpc/decision/voi.py``).

What it is NOT (honest limits):
    - NOT causal uplift: all 766 issued traces share one trigger rule
      (``15_consecutive_failed_contacts``) with no no-trace control arm, so
      P(pay | trace) - P(pay | no trace) is unidentifiable here. This model
      estimates the first term only. A randomized trace holdout is the
      prerequisite for true uplift (logged as future work, not built).
    - Trace ``result`` (``new_phone_found`` et al.) is post-treatment and
      must NEVER be a feature — it is known only after paying the cost.
      Observed split on issued extracts: found-phone 0.203 / no-info 0.168
      / new-address 0.037 (n=27, thin) at 30d.

Labels (``trace_outcome_labels``):
    ``paid`` = any payment with ``trace_date < payment_ts <= trace_date +
    window``; ``censored`` = ``trace_date + window`` past the last payment
    timestamp (label unknown, excluded from fit — 5.9% at 30d, 50% at 60d,
    which is why 60d is unusable). Payments run to 2026-07-24.

Point-in-time requirements:
    - Features must be pre-trace (``<= trace_date``); post-trace payments,
      trace results, and new IDs are outcomes, never inputs.
    - Banned/hidden columns are never read. Quarantined snapshot numerics
      stay out unless the caller explicitly supplies them (as-of
      unconfirmed — owner: CN ask #7).

Units: probabilities in [0, 1]; ``cost_inr`` in rupees (context only,
never a predictor of willingness to pay — kept out of the matrix).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.rpc.models.baselines._gbm_common import make_lgbm

# Columns that must never be features, even if present.
BANNED_COLUMNS = frozenset({
    "ground_truth",
    "true_state",
    "true_avoidance",
    "true_borrower_avoidance",
    "shared_reason",
    "policy_action",
    "verified_status",
    "verified_holdout",
    "paid",
    "censored",
    "trace_id",
    "account_id",
    "trace_date",
    "result",
    "new_contact_point_id",
    "cost_inr",
    "trigger_rule",
})


def load_uplift_params(cfg: str | Path | None = None) -> dict[str, Any]:
    """Load label window + GBM hyperparameters (config over constants)."""
    repo = Path(__file__).resolve().parents[3]
    path = Path(cfg) if cfg else repo / "configs" / "uplift.yaml"
    with path.open(encoding="utf-8") as fh:
        s = yaml.safe_load(fh) or {}
    gbm = s.get("gbm", {})
    return {
        "window_days": int(s.get("window_days", 30)),
        "gbm": {
            "n_estimators": int(gbm.get("n_estimators", 100)),
            "learning_rate": float(gbm.get("learning_rate", 0.05)),
            "num_leaves": int(gbm.get("num_leaves", 31)),
            "min_child_samples": int(gbm.get("min_child_samples", 20)),
        },
        "seed": int(s.get("seed", 42)),
    }


def trace_outcome_labels(
    traces: pd.DataFrame,
    payments: pd.DataFrame,
    window_days: int = 30,
) -> pd.DataFrame:
    """Label each trace: payment in (trace_date, trace_date + window].

    Returns ``trace_id`` + ``account_id`` + ``paid`` (1/0, NaN when
    censored) + ``censored`` (trace window exceeds the payment feed).
    """
    tr = traces.copy()
    pay = payments.copy()
    tr["trace_date"] = pd.to_datetime(tr["trace_date"], utc=True)
    pay["payment_ts"] = pd.to_datetime(pay["payment_ts"], utc=True)
    window = pd.Timedelta(days=window_days)
    last_pay = pay["payment_ts"].max()
    pay_by_acc = pay.groupby("account_id")["payment_ts"].apply(list).to_dict()
    paid, censored = [], []
    for row in tr.itertuples():
        t0 = row.trace_date
        hit = any(
            pd.Timedelta(0) < ts - t0 <= window
            for ts in pay_by_acc.get(row.account_id, [])
        )
        cens = bool(t0 + window > last_pay)
        paid.append(float("nan") if cens else float(hit))
        censored.append(cens)
    return pd.DataFrame({
        "trace_id": tr["trace_id"].astype(str).to_numpy(),
        "account_id": tr["account_id"].astype(str).to_numpy(),
        "paid": paid,
        "censored": censored,
    })


_MIN_FIT_ROWS = 2
_MIN_FIT_CLASSES = 2


class TraceOutcomeModel:
    """GBM for P(pay in 30d | traced, pre-trace history). Deterministic."""

    name = "trace_outcome"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or load_uplift_params()
        gbm_params = dict(self.params["gbm"])
        gbm_params["seed"] = self.params["seed"]
        self._model: Any = make_lgbm(gbm_params)
        self._base_rate = 0.0
        self._columns: list[str] = []

    def _matrix(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cols = [
            c for c in features.columns
            if c not in BANNED_COLUMNS
            and (
                pd.api.types.is_any_real_numeric_dtype(features[c].dtype)
                or pd.api.types.is_bool_dtype(features[c].dtype)
            )
        ]
        mat = features[cols].copy()
        for c in cols:
            if pd.api.types.is_bool_dtype(mat[c].dtype):
                mat[c] = mat[c].astype("float64")
        return mat.fillna(0.0).to_numpy(dtype=float), cols

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> TraceOutcomeModel:
        """Fit on uncensored traces (censored rows excluded, never negative)."""
        lab = labels.set_index(labels["trace_id"].astype(str))
        feat = features.copy()
        feat["_key"] = features["trace_id"].astype(str)
        keep = [k for k in feat["_key"] if k in lab.index and not bool(lab.loc[k, "censored"])]
        feat = feat[feat["_key"].isin(keep)].reset_index(drop=True)
        y = np.array([float(lab.loc[k, "paid"]) for k in feat["_key"]])
        mat, self._columns = self._matrix(feat)
        self._base_rate = float(y.mean()) if len(y) else 0.0
        if len(y) >= _MIN_FIT_ROWS and np.unique(y).size >= _MIN_FIT_CLASSES:
            self._model.fit(mat, y)
        else:
            self._model = None
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score traces: ``p_recover_30d`` in [0.01, 0.99] + base-rate fallback."""
        mat, _ = self._matrix(features)
        if self._model is None or not self._columns:
            p = np.full(len(features), self._base_rate)
        else:
            try:
                p = self._model.predict_proba(mat)[:, 1]
            except Exception:
                p = np.full(len(features), self._base_rate)
        return pd.DataFrame({
            "trace_id": features["trace_id"].astype(str).to_numpy(),
            "p_recover_30d": np.clip(p, 0.01, 0.99),
        })

    def to_dict(self) -> dict[str, Any]:
        """Audit-log snapshot (params + base rate + matrix columns)."""
        return {
            "params": self.params,
            "base_rate": self._base_rate,
            "columns": list(self._columns),
        }
