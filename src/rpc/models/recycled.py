"""Recycled-number thin risk classifier (proxy labels + PU learning, review only).

What it is:
    A deliberately thin risk score for review prioritisation. Positive-Unlabelled
    (PU) framing: proxy-positive rows (wrong-number evidence and friends) are
    positives; every other row is UNLABELLED, never a negative. An Elkan-Noto
    estimator converts P(proxy-positive | x) into P(recycled | x). Output is a
    risk score plus a review ranking at the 1:100 review rate, with an eval
    hook that scores the 27 ``not_borrower_number`` verified rows as gold.

Proxy-label definition and its limits:
    Proxy-positive = at least one wrong-number disposition on the link, or a
    high bureau wrong-number rate (configurable via the caller's proxy frame).
    Limits (per the dataset audit): 330 links show BOTH rpc_* and wrong_number
    (contact-then-stranger sequences -- a proxy-positive can still be the
    borrower's line); agent wrong-number judgments run noisy in both
    directions; true recycled status is UNKNOWN, so this score stays
    rule-plus-review and never auto-decides.

What it is NOT (boundaries):
    - Never an auto-decision: this module exposes no threshold-to-action
      mapping, no ``decide``/``predict_action`` method, and no recommended
      cutoff. The suppression cutoff lives in the decision layer
      (cost-ratio rule), not here.
    - Not a validity classifier: unlabelled rows are not negatives, so its
      scores must not train or evaluate any "valid vs not" model.

Point-in-time requirements:
    - Features must be PIT at the scoring origin (``received_at <= as_of``).
    - Proxy labels must derive from evidence strictly ``<= as_of``; verified
      rows are eval-only gold and must never be features (even membership
      carries future selection info -- 66/250 verified phones were never
      dialled).

CN / open questions affecting this module:
    - Recycled confirmations (annotations) first -- ask-CN #8; until then the
      proxy stays weak and the 1:100 review rate stays a capacity rule.
    - Lawful basis for cross-account/lender contact graph features.
    - Per-lender economics for review capacity (single vs per-lender tables).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.rpc.models.baselines._gbm_common import make_lgbm, to_matrix

REVIEW_RATE = 0.01  # rule-plus-review operating point: 1 review per 100 scored
POSITIVE_STATUS = "not_borrower_number"  # the 27 verified gold positives
NEGATIVE_STATUS = "borrower_number"  # verified negatives for the gold check
_MIN_FIT_CLASSES = 2
_MIN_LABEL_FREQ = 1e-6


class RecycledRiskScorer:
    """Thin PU-learning recycled-risk scorer (scores for review, never actions)."""

    name = "recycled_risk"

    def __init__(self, review_rate: float = REVIEW_RATE, seed: int = 42) -> None:
        if not 0.0 < review_rate <= 1.0:
            msg = "review_rate must be in (0, 1]"
            raise ValueError(msg)
        self.review_rate = review_rate
        self.seed = seed
        self._model: Any = None
        self._label_freq = 1.0  # c = P(proxy-positive | truly recycled)
        self._proxy_rate = 0.0
        self._degenerate = True
        self._features: pd.DataFrame | None = None

    # -- fitting ---------------------------------------------------------
    def fit(self, features: pd.DataFrame, proxy: pd.Series | np.ndarray) -> RecycledRiskScorer:
        """Fit P(proxy-positive | x); unlabelled (0) rows are NOT negatives.

        Elkan-Noto: c = mean P(s=1|x) over proxy-positives, then
        P(recycled|x) = min(P(s=1|x) / c, 1). Degenerate inputs (no
        positives, single class, unusable matrix) flag ``degenerate_`` and
        predict a constant instead of learning noise.
        """
        s = np.asarray(proxy, dtype=float).reshape(len(features))
        mask = ~np.isnan(s)
        if mask.sum() == 0:
            msg = "no finite proxy labels"
            raise ValueError(msg)
        s = s[mask]
        feats = features.iloc[np.flatnonzero(mask)].reset_index(drop=True)
        self._proxy_rate = float((s == 1.0).mean())
        x, _ = to_matrix(feats)
        if x.shape[1] == 0 or np.unique(s).size < _MIN_FIT_CLASSES or (s == 1.0).sum() == 0:
            self._model, self._degenerate = None, True
            return self
        model: Any = make_lgbm({"seed": self.seed})
        try:
            model.fit(x, s)
            ps = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        except Exception:
            self._model, self._degenerate = None, True
            return self
        pos = ps[s == 1.0]
        c = float(pos.mean()) if len(pos) else 0.0
        self._model = model
        self._label_freq = c if c > _MIN_LABEL_FREQ else 1.0
        self._degenerate = False
        return self

    @property
    def degenerate_(self) -> bool:
        return self._degenerate

    # -- prediction ------------------------------------------------------
    def predict_risk(self, features: pd.DataFrame) -> np.ndarray:
        """Risk scores in [0, 1] for review prioritisation (not decisions)."""
        if self._model is None:
            return np.full(len(features), np.clip(self._proxy_rate, 0.0, 1.0))
        x, _ = to_matrix(features)
        if x.shape[1] == 0:
            return np.full(len(features), np.clip(self._proxy_rate, 0.0, 1.0))
        try:
            ps = np.asarray(self._model.predict_proba(x)[:, 1], dtype=float)
        except Exception:
            return np.full(len(features), np.clip(self._proxy_rate, 0.0, 1.0))
        return np.clip(ps / self._label_freq, 0.0, 1.0)

    def rank_for_review(
        self,
        contact_point_refs: Sequence[str],
        features: pd.DataFrame | None = None,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Order refs by descending risk; flag the review slice (no actions).

        ``in_review`` marks the top ``top_k`` rows, defaulting to the 1:100
        operating rate (at least 1 row). There is deliberately no threshold
        or action column -- review is a human step owned outside this module.
        """
        refs = list(contact_point_refs)
        feats = features if features is not None else self._features
        if feats is None:
            msg = "pass features or attach_features first"
            raise ValueError(msg)
        if "contact_point_ref" in feats.columns:
            aligned = pd.DataFrame({"contact_point_ref": refs}).merge(
                feats, on="contact_point_ref", how="left"
            )
            feat_rows = aligned.drop(columns=["contact_point_ref"])
        else:
            feat_rows = feats.iloc[: len(refs)].reset_index(drop=True)
        risk = self.predict_risk(feat_rows)
        order = np.argsort(-risk, kind="stable")
        if top_k is None:
            k = max(1, int(len(refs) * self.review_rate))
        else:
            k = max(0, min(int(top_k), len(refs)))
        flagged = np.zeros(len(refs), dtype=bool)
        flagged[order[:k]] = True
        ranks = np.empty(len(refs), dtype=int)
        ranks[order] = np.arange(1, len(refs) + 1)
        out = pd.DataFrame(
            {
                "contact_point_ref": refs,
                "recycled_risk": np.clip(risk, 0.0, 1.0),
                "rank": ranks,
                "in_review": flagged,
            }
        )
        return out.sort_values("rank", kind="stable").reset_index(drop=True)

    # -- gold-check hook ---------------------------------------------------
    def evaluate_verified(
        self,
        verified: pd.DataFrame,
        risk_scores: pd.Series | np.ndarray | None = None,
        ref_col: str = "contact_point_ref",
        status_col: str = "verified_status",
    ) -> dict[str, object]:
        """Score the verified gold set: 27 not_borrower positives vs negatives.

        ``verified`` rows mirror ``verified_contact_points.csv`` with the
        contact reference already hashed per the mapping plan. Third-party,
        switched-off and invalid statuses are ignored with counts (a
        third-party line is a different risk; switched-off/invalid are not
        recycled evidence). Returns PR-AUC plus precision/recall at the
        1:100 review-budget cutoff.
        """
        if ref_col not in verified.columns or status_col not in verified.columns:
            msg = f"verified needs {ref_col} and {status_col} columns"
            raise ValueError(msg)
        if risk_scores is None:
            if self._features is None:
                msg = "pass risk_scores or attach_features first"
                raise ValueError(msg)
            risk = self.predict_risk(self._features)
            refs = self._features["contact_point_ref"].to_numpy(dtype=object)
            risk_by_ref = dict(zip(refs, risk))
        else:
            arr = np.asarray(risk_scores, dtype=float).reshape(len(verified))
            risk_by_ref = dict(zip(verified[ref_col].to_numpy(dtype=object), arr))
        pos = verified[verified[status_col] == POSITIVE_STATUS]
        neg = verified[verified[status_col] == NEGATIVE_STATUS]
        ignored = verified[~verified[status_col].isin([POSITIVE_STATUS, NEGATIVE_STATUS])]
        rows = []
        for _, r in pd.concat([pos, neg]).iterrows():
            if r[ref_col] in risk_by_ref:
                label = 1.0 if r[status_col] == POSITIVE_STATUS else 0.0
                rows.append((float(risk_by_ref[r[ref_col]]), label))
        if not rows:
            return {
                "n_pos": len(pos),
                "n_neg": len(neg),
                "n_ignored": len(ignored),
                "n_scored": 0,
                "note": "no verified refs matched risk scores",
            }
        scores = np.array([r[0] for r in rows])
        labels = np.array([r[1] for r in rows])
        order = np.argsort(-scores, kind="stable")
        k = max(1, int(len(scores) * self.review_rate))
        top = order[:k]
        tp = float(labels[top].sum())
        scored = len(scores)
        return {
            "n_pos": len(pos),
            "n_neg": len(neg),
            "n_ignored": len(ignored),
            "n_scored": scored,
            "pr_auc": _pr_auc(labels, scores),
            "precision_at_review_budget": (tp / k) if k else float("nan"),
            "recall_at_review_budget": (tp / labels.sum()) if labels.sum() else float("nan"),
            "review_k": int(k),
        }

    # -- serve-shaped adapter ----------------------------------------------
    def attach_features(self, features: pd.DataFrame) -> RecycledRiskScorer:
        """Attach the scoring-origin frame (must carry contact_point_ref)."""
        if "contact_point_ref" not in features.columns:
            msg = "features need a contact_point_ref column"
            raise ValueError(msg)
        self._features = features.copy()
        return self

    def score(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        """Serve-shaped frame: ``recycled_risk`` + confidence (no p_rpc).

        Interface gap (flagged, not patched): the eval ``Scorer`` protocol
        expects ``p_rpc`` as the target, but recycled risk is NOT an RPC
        probability. Consumers must read ``recycled_risk`` (merged into
        ``ContactPointScore.recycled_risk`` via ``merge_risk_into_scores``)
        and must not interpret any ``p_rpc``-shaped column from this module.
        """
        refs = list(contact_point_refs)
        if self._features is None:
            return pd.DataFrame(
                {"contact_point_ref": refs, "recycled_risk": 0.0, "confidence": 0.0}
            )
        fr = pd.DataFrame({"contact_point_ref": refs}).merge(
            self._features, on="contact_point_ref", how="left"
        )
        feat = fr.drop(columns=["contact_point_ref"])
        risk = self.predict_risk(feat)
        _ = as_of
        conf = 0.0 if self._degenerate else 0.6
        return pd.DataFrame(
            {
                "contact_point_ref": refs,
                "recycled_risk": np.clip(risk, 0.0, 1.0),
                "confidence": conf,
            }
        )


def _pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision over recall steps (tiny, dependency-light)."""
    order = np.argsort(-scores, kind="stable")
    ranked = y[order]
    total = ranked.sum()
    if total == 0 or len(ranked) == 0:
        return float("nan")
    tp_cum = np.cumsum(ranked)
    precision = tp_cum / np.arange(1, len(ranked) + 1)
    recall_step = ranked / total
    return float((precision * recall_step).sum())


def merge_risk_into_scores(scores: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    """Adapter: base RPC scores + recycled risk -> decision-layer frame.

    Keeps ``p_rpc`` from the base model untouched and adds/overwrites only
    ``recycled_risk`` (and ``confidence`` when the risk side is confident).
    The single construction point before ``ContactPointScore`` assembly.
    """
    if "contact_point_ref" not in scores.columns or "contact_point_ref" not in risk.columns:
        msg = "both frames need contact_point_ref"
        raise ValueError(msg)
    if "recycled_risk" not in risk.columns:
        msg = "risk frame needs a recycled_risk column"
        raise ValueError(msg)
    out = scores.merge(
        risk[["contact_point_ref", "recycled_risk"]], on="contact_point_ref", how="left"
    )
    out["recycled_risk"] = out["recycled_risk"].fillna(0.0).clip(0.0, 1.0)
    return out


__all__ = [
    "NEGATIVE_STATUS",
    "POSITIVE_STATUS",
    "REVIEW_RATE",
    "RecycledRiskScorer",
    "merge_risk_into_scores",
]
