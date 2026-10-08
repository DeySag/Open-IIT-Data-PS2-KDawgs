"""Third-party risk scorer (workbook model 7, P1) — risk only, never a decision.

What it is:
    A Beta-Binomial risk score for review prioritisation. Source priors
    (employer/reference phones answer third-party far more often than KYC
    lines on the issued extracts) are updated with the link's own
    pre-``as_of`` third-party history. Output is a risk score plus an audit
    trail (prior, counts, posterior); the suppression threshold lives in the
    decision layer's cost-ratio rule, not here.

What it is NOT (boundaries):
    - Never an auto-decision: no threshold, no ``decide``/``predict_action``
      method, no recommended cutoff. Debt is never disclosed to a third
      party — risk resolves toward suppression in the decision layer.
    - Not a validity classifier: low risk is not proof the line reaches the
      borrower; use the state tracker's posteriors for that.

Labels and gold:
    - Fit labels are weak: any ``third_party_contact`` / ``third_party_ptp``
      disposition on the link inside the caller-supplied fit window.
      True third-party status is UNKNOWN (audit §5).
    - Eval-only gold: ``third_party_number`` (74) vs ``borrower_number``
      (127) from ``verified_contact_points.csv`` — never features, never
      fit rows (even membership carries selection info).

Point-in-time requirements:
    - Fit outcomes and score-time histories must derive from evidence
      strictly ``<= as_of``; the caller guarantees the window, this module
      never looks at timestamps beyond filtering (score takes counts).
    - ``source`` is link metadata (known when the link enters via
      ``added_date``); only links with ``added_date <= as_of`` may be
      scored. Banned/hidden columns are never read.

Units: all rates/probabilities in [0, 1]; counts are ints; risk unitless
in [0, 1].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Columns that must never influence the model, even if present.
BANNED_COLUMNS = frozenset({
    "ground_truth",
    "true_state",
    "true_avoidance",
    "true_borrower_avoidance",
    "shared_reason",
    "policy_action",
    "verified_status",
    "verified_holdout",
})

TP_DISPOSITIONS = ("third_party_contact", "third_party_ptp")
GOLD_POSITIVE = "third_party_number"
GOLD_NEGATIVE = "borrower_number"


def load_third_party_params(cfg: str | Path | None = None) -> dict[str, Any]:
    """Load hyperparameters (config over constants)."""
    repo = Path(__file__).resolve().parents[3]
    path = Path(cfg) if cfg else repo / "configs" / "third_party.yaml"
    with path.open(encoding="utf-8") as fh:
        s = yaml.safe_load(fh) or {}
    return {
        "smoothing_alpha": float(s.get("smoothing_alpha", 20.0)),
        "min_samples": int(s.get("min_samples", 20)),
        "seed": int(s.get("seed", 42)),
    }


class ThirdPartyRiskScorer:
    """Beta-Binomial third-party risk. Closed-form, deterministic."""

    name = "third_party_risk"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or load_third_party_params()
        self._source_prior: dict[str, float] = {}
        self._global_prior = 0.0

    def fit(self, link_outcomes: pd.DataFrame) -> ThirdPartyRiskScorer:
        """Estimate smoothed source priors from caller-supplied PIT outcomes.

        ``link_outcomes`` columns: ``source`` (str), ``tp_ever`` (0/1 —
        any third-party disposition on the link inside the fit window,
        ``<= as_of`` by caller guarantee). One row per (account, phone).
        """
        df = link_outcomes.copy()
        for banned in BANNED_COLUMNS:
            if banned in df.columns:
                df = df.drop(columns=[banned])
        alpha = float(self.params["smoothing_alpha"])
        self._global_prior = float(df["tp_ever"].mean()) if len(df) else 0.0
        self._source_prior = {}
        for src, g in df.groupby(df["source"].astype(str)):
            n = len(g)
            if n < int(self.params["min_samples"]):
                continue  # thin sources fall back to global at score time
            k = float(g["tp_ever"].sum())
            self._source_prior[str(src)] = (k + alpha * self._global_prior) / (n + alpha)
        return self

    def source_prior(self, source: str) -> float:
        """Smoothed P(third-party | source); global fallback when thin/unseen."""
        return float(self._source_prior.get(str(source), self._global_prior))

    def score(self, links: pd.DataFrame) -> pd.DataFrame:
        """Score links: posterior = Beta(prior) updated with past counts.

        ``links`` columns: ``contact_point_ref``, ``source``,
        ``n_tp_past`` (int, third-party dispositions ``<= as_of``),
        ``n_attempts_past`` (int, dialled attempts ``<= as_of``).
        Returns ``contact_point_ref`` + ``risk`` + audit columns
        (``prior``, ``n_tp_past``, ``n_attempts_past``).
        """
        df = links.copy()
        for banned in BANNED_COLUMNS:
            if banned in df.columns:
                df = df.drop(columns=[banned])
        alpha = float(self.params["smoothing_alpha"])
        priors = df["source"].astype(str).map(self.source_prior).astype(float)
        a = alpha * priors + df["n_tp_past"].astype(float)
        b = alpha * (1.0 - priors) + (
            df["n_attempts_past"].astype(float) - df["n_tp_past"].astype(float)
        ).clip(lower=0.0)
        out = pd.DataFrame({
            "contact_point_ref": df["contact_point_ref"].astype(str).to_numpy(),
            "risk": (a / (a + b)).clip(0.0, 1.0).to_numpy(),
            "prior": priors.to_numpy(),
            "n_tp_past": df["n_tp_past"].astype(int).to_numpy(),
            "n_attempts_past": df["n_attempts_past"].astype(int).to_numpy(),
        })
        return out

    def gold_check(self, verified: pd.DataFrame) -> dict[str, Any]:
        """Eval-only hook: rank-AUC of risk on verified gold (no fitting).

        ``verified`` columns: ``contact_point_ref``, ``verified_status``,
        plus the ``score`` input columns (``source``, counts). Rows with
        other statuses are ignored. Never call with fit data.
        """
        scored = self.score(verified)
        status = verified["verified_status"].astype(str).reset_index(drop=True)
        scored = scored.reset_index(drop=True)
        pos = scored["risk"][status == GOLD_POSITIVE].to_numpy()
        neg = scored["risk"][status == GOLD_NEGATIVE].to_numpy()
        n_pos, n_neg = len(pos), len(neg)
        if n_pos == 0 or n_neg == 0:
            auc = float("nan")
        else:
            # Mann-Whitney rank AUC: P(positive outranks negative).
            order = np.argsort(np.concatenate([pos, neg]))
            ranks = np.empty(len(order))
            ranks[order] = np.arange(1, len(order) + 1)
            auc = float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        return {"n_positive": n_pos, "n_negative": n_neg, "rank_auc": auc}

    def to_dict(self) -> dict[str, Any]:
        """Audit-log snapshot (params + fitted priors)."""
        return {
            "params": dict(self.params),
            "global_prior": self._global_prior,
            "source_priors": dict(self._source_prior),
        }
