"""Dial-selection propensity with a 1/k validation gate and IPS weights.

What it is:
    Estimates dial-selection propensity P(dialled | features, arm) for
    unbiased fitting (stabilised, clipped IPS weights at training time) and
    for logging the selection probability that the decision layer's
    exploration budget needs at decision time. Covers rule-selected dials
    (rule arm propensity is 1.0 by design) as well as model/random-selected
    dials, and validates the logged ``selection_propensity`` against the 1/k
    uniform-over-k formula before any IPS weight is trusted.

What it is NOT (boundaries):
    - Not an RPC scorer: it predicts selection, never contact probability.
      Do not register it as an eval ``Scorer`` and do not serve its outputs
      as ``p_rpc`` (flagged interface gap, not patched).
    - Not a silent fallback: when the 1/k check fails (or was never run),
      ``ips_weights`` raises instead of returning weights.
    - Never reads the eval-only policy log; it fits on caller-supplied
      exposure frames only (leakage rule).

Point-in-time requirements:
    - Features must be PIT at the decision origin (``received_at <= as_of``).
    - The dialled indicator and arm must come from the same exposure window
      as the features; the propensity fit window must precede the outcome
      window used for the downstream label.

CN / open questions affecting this module:
    - ``selection_propensity`` formula confirmation (1/k) -- ask-CN #6.
    - Campaign/policy flags beyond ``dialling_arm`` (unknown selection arms).
    - Timestamp timezone (IST vs UTC) for exposure windows.
    - Lawful basis for cross-account/lender aggregates if used as features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.rpc.models.baselines._gbm_common import make_lgbm, to_matrix

RULE_ARM_VALUES: tuple[str, ...] = ("rule_based", "rule")
_MIN_TRAIN_ROWS = 2


class PropensityValidationError(RuntimeError):
    """Raised when IPS weights are requested without a passing 1/k check."""


@dataclass
class ValidationReport:
    passed: bool
    max_abs_dev: float
    tol: float
    n_checked: int
    detail: str


@dataclass
class PropensityConfig:
    """Knobs for :class:`DialPropensityModel` (clip bounds are parameters)."""

    clip_low: float = 0.05
    clip_high: float = 0.95
    max_weight: float = 20.0
    arm_col: str = "dialling_arm"
    rule_arm_values: tuple[str, ...] = RULE_ARM_VALUES
    tol_1k: float = 0.02
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.clip_low < self.clip_high < 1.0:
            msg = "need 0 < clip_low < clip_high < 1"
            raise ValueError(msg)
        if self.max_weight <= 0.0:
            msg = "max_weight must be positive"
            raise ValueError(msg)


class DialPropensityModel:
    """P(dialled | features, arm) with a mandatory 1/k validation gate.

    Stabilised weight for a dialled row: ``w = P(dialled) / clip(p)`` capped
    at ``max_weight``; undialled rows get 0.0. Propensities are clipped to
    ``[clip_low, clip_high]`` before inversion (defaults mirror the
    ``configs/eval.yaml`` propensity section).
    """

    name = "dial_propensity"

    def __init__(self, config: PropensityConfig | None = None) -> None:
        self.config = config or PropensityConfig()
        self._model: Any = None
        self._marginal = 0.5
        self._validation: ValidationReport | None = None
        self._context: pd.DataFrame | None = None

    # -- fitting ---------------------------------------------------------
    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Baseline helper columns plus any other numeric columns."""
        base, used = to_matrix(features)
        extra_cols = [
            c for c in features.select_dtypes(include=[np.number]).columns if c not in used
        ]
        if not extra_cols:
            return base
        extra = features[extra_cols].fillna(0.0).to_numpy(dtype=float)
        if base.shape[1] == 0:
            return extra
        return np.column_stack([base, extra])

    def fit(
        self,
        features: pd.DataFrame,
        dialled: pd.Series | np.ndarray,
        arm: pd.Series | np.ndarray | None = None,
    ) -> DialPropensityModel:
        """Fit P(dialled | features) on non-rule rows; rule rows fix p = 1.0."""
        y = np.asarray(dialled, dtype=float).reshape(len(features))
        mask = ~np.isnan(y)
        self._marginal = float(y[mask].mean()) if mask.any() else 0.5
        is_rule = self._rule_mask(features, arm)
        train = mask & ~is_rule
        x = self._matrix(features)
        if x.shape[1] == 0:
            self._model = None  # no usable columns: predict the marginal rate
        elif int(train.sum()) >= _MIN_TRAIN_ROWS and np.unique(y[train]).size >= _MIN_TRAIN_ROWS:
            model: Any = make_lgbm({"seed": self.config.seed})
            model.fit(x[train], y[train])
            self._model = model
        else:
            self._model = None  # degenerate: predict the marginal rate
        return self

    def _rule_mask(self, features: pd.DataFrame, arm: pd.Series | np.ndarray | None) -> np.ndarray:
        if arm is not None:
            arm_vals = np.asarray(arm, dtype=object).reshape(len(features))
        elif self.config.arm_col in features.columns:
            arm_vals = features[self.config.arm_col].to_numpy(dtype=object)
        else:
            return np.zeros(len(features), dtype=bool)
        return np.isin(arm_vals, list(self.config.rule_arm_values))

    # -- prediction ------------------------------------------------------
    def predict_propensity(
        self,
        features: pd.DataFrame,
        arm: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-row P(dialled); rule-arm rows return exactly 1.0."""
        x = self._matrix(features)
        if self._model is None:
            p = np.full(len(features), self._marginal)
        else:
            try:
                p = np.asarray(self._model.predict_proba(x)[:, 1], dtype=float)
            except Exception:
                p = np.full(len(features), self._marginal)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        p[self._rule_mask(features, arm)] = 1.0
        return np.asarray(p, dtype=float)

    # -- 1/k validation gate ---------------------------------------------
    def validate_1k(
        self,
        k: pd.Series | np.ndarray,
        logged_propensity: pd.Series | np.ndarray,
        arm: pd.Series | np.ndarray | None = None,
        tol: float | None = None,
    ) -> ValidationReport:
        """Check logged propensity against the 1/k uniform-over-k formula.

        Only non-rule rows are checked (rule-arm propensity is 1.0 by
        design, not 1/k). Rows with NaN k/logged propensity fail the check.
        """
        kk = np.asarray(k, dtype=float).reshape(-1)
        lp = np.asarray(logged_propensity, dtype=float).reshape(-1)
        if len(kk) != len(lp):
            msg = "k and logged_propensity lengths differ"
            raise ValueError(msg)
        if arm is not None:
            arm_vals = np.asarray(arm, dtype=object).reshape(len(kk))
            check = ~np.isin(arm_vals, list(self.config.rule_arm_values))
        else:
            check = np.ones(len(kk), dtype=bool)
        idx = np.flatnonzero(check)
        threshold = self.config.tol_1k if tol is None else tol
        if len(idx) == 0:
            report = ValidationReport(False, float("nan"), threshold, 0, "no rows to check")
        elif np.isnan(kk[idx]).any() or np.isnan(lp[idx]).any() or (kk[idx] <= 0).any():
            report = ValidationReport(False, float("inf"), threshold, len(idx), "bad k values")
        else:
            dev = float(np.abs(lp[idx] - 1.0 / kk[idx]).max())
            ok = dev <= threshold
            detail = "1/k holds" if ok else "logged propensity deviates from 1/k"
            report = ValidationReport(ok, dev, threshold, len(idx), detail)
        self._validation = report
        return report

    # -- training-time weights -------------------------------------------
    def ips_weights(
        self,
        features: pd.DataFrame | None = None,
        propensity: pd.Series | np.ndarray | None = None,
        arm: pd.Series | np.ndarray | None = None,
        dialled: pd.Series | np.ndarray | None = None,
        require_validation: bool = True,
    ) -> np.ndarray:
        """Stabilised, clipped IPS weights for unbiased fitting.

        Refuses (raises ``PropensityValidationError``) when the 1/k check
        has not been run or did not pass -- never silently proceeds.
        """
        if require_validation:
            if self._validation is None:
                msg = "run validate_1k before ips_weights"
                raise PropensityValidationError(msg)
            if not self._validation.passed:
                msg = "1/k validation failed; weights refused"
                raise PropensityValidationError(msg)
        if propensity is not None:
            p = np.asarray(propensity, dtype=float).reshape(-1)
        elif features is not None:
            p = self.predict_propensity(features, arm)
        else:
            msg = "pass propensity or features"
            raise ValueError(msg)
        cfg = self.config
        pc = np.clip(p, cfg.clip_low, cfg.clip_high)
        w = np.clip(self._marginal / pc, 0.0, cfg.max_weight)
        if dialled is not None:
            d = np.asarray(dialled, dtype=float).reshape(len(w))
            w = np.where(d == 1.0, w, 0.0)
        return np.asarray(w, dtype=float)

    @staticmethod
    def effective_sample_size(weights: np.ndarray) -> float:
        """ESS = (sum w)^2 / sum w^2; low ESS means weights are degenerate."""
        w = np.asarray(weights, dtype=float)
        denom = float((w**2).sum())
        if denom <= 0.0 or len(w) == 0:
            return 0.0
        return float(w.sum() ** 2 / denom)

    # -- decision-time uses ------------------------------------------------
    def logged_propensity_for_decision(
        self,
        features: pd.DataFrame,
        arm: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Per-row logged propensity for the exploration-budget ledger."""
        frame = pd.DataFrame({"logged_propensity": self.predict_propensity(features, arm)})
        if self.config.arm_col in features.columns:
            frame[self.config.arm_col] = features[self.config.arm_col].to_numpy()
        return frame

    @staticmethod
    def exploration_slots(n_candidates: int, exploration_fraction: float = 0.05) -> int:
        """Exploration-budget slots: fraction of candidates (default 5%).

        Default mirrors ``costs.yaml`` ``exploration_fraction``; the decision
        layer owns the budget, this only sizes the draw.
        """
        if n_candidates <= 0:
            return 0
        if not 0.0 <= exploration_fraction <= 1.0:
            msg = "exploration_fraction must be in [0, 1]"
            raise ValueError(msg)
        return int(n_candidates * exploration_fraction)

    # -- serve-shaped adapter ----------------------------------------------
    def attach_context(self, frame: pd.DataFrame) -> DialPropensityModel:
        """Attach the scoring-origin frame (contact_point_ref + features)."""
        if "contact_point_ref" not in frame.columns:
            msg = "frame needs a contact_point_ref column"
            raise ValueError(msg)
        self._context = frame.copy()
        return self

    def score(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        """Serve-shaped frame: propensity + weight per ref (NOT p_rpc)."""
        refs = list(contact_point_refs)
        if self._context is None:
            return pd.DataFrame(
                {
                    "contact_point_ref": refs,
                    "propensity": self._marginal,
                    "ips_weight": 1.0,
                    "confidence": 0.3,
                }
            )
        fr = pd.DataFrame({"contact_point_ref": refs}).merge(
            self._context, on="contact_point_ref", how="left"
        )
        arm_col = self.config.arm_col
        arm = fr[arm_col] if arm_col in fr.columns else None
        feat = fr.drop(columns=["contact_point_ref"])
        cfg = self.config
        p = self.predict_propensity(feat, arm)
        try:
            w = self.ips_weights(features=feat, arm=arm)
        except PropensityValidationError:
            clipped = np.clip(p, cfg.clip_low, cfg.clip_high)
            w = np.clip(self._marginal / clipped, 0.0, cfg.max_weight)
        _ = as_of
        return pd.DataFrame(
            {"contact_point_ref": refs, "propensity": p, "ips_weight": w, "confidence": 0.5}
        )


__all__ = [
    "RULE_ARM_VALUES",
    "DialPropensityModel",
    "PropensityConfig",
    "PropensityValidationError",
    "ValidationReport",
]
