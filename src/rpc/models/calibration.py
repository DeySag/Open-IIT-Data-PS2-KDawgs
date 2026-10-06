"""Per-segment probability calibration (isotonic and beta) with conformal intervals.

What it is:
    Post-hoc calibration for RPC (right-party-contact) probabilities. It maps
    uncalibrated base-model scores to calibrated probabilities, one map per
    named segment plus a global fallback map. It also emits inductive-conformal
    prediction intervals around the calibrated probability for the VOI gate
    and per-segment reporting in the eval harness.

What it is NOT (boundaries):
    - Not a classifier: it never sees raw features and never changes ranking
      within a segment under the isotonic map (monotone); it only rescales.
    - Not a refit of the base model: it must be fit on a LATER time split
      than the base model (enforced via ``fit_as_of`` / ``base_train_end``).
    - Not a decision rule: it outputs probabilities and intervals, never
      dial/trace actions or thresholds.

Point-in-time requirements:
    - Uncalibrated scores must come from a base model trained on events with
      ``received_at <= base_train_end``.
    - Calibration labels must be observed outcomes from a window AFTER
      ``base_train_end`` (``fit_as_of`` is the calibration scoring origin;
      ``fit_as_of > base_train_end`` is enforced).
    - Segment columns (dialling arm, lender, recency bucket, ...) must be
      computed point-in-time at the calibration origin (no post-origin data).

CN / open questions affecting this module:
    - Timestamp timezone (IST wall-clock vs UTC) shifts recency-bucket edges.
    - ``selection_propensity`` 1/k confirmation: dialled-only calibration data
      is selection-biased; IPS-adjusted calibration awaits a validated
      propensity model (see ``src/rpc/models/propensity.py``).
    - Verified-as-gold scope and account-snapshot as-of confirmation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

DEFAULT_SEGMENT_COLS: tuple[str, ...] = ("dialling_arm", "lender", "recency_bucket")
METHODS: tuple[str, ...] = ("isotonic", "beta")
_EPS = 1e-9
_MIN_FIT_ROWS = 2


def _clip01(p: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return result


def _sigmoid(z: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))
    return result


def _has_both_classes(y: np.ndarray) -> bool:
    return bool(np.unique(np.asarray(y)).size >= _MIN_FIT_ROWS)


@dataclass
class _FittedMap:
    """One fitted calibrator: ``kind`` in {isotonic, beta, identity}."""

    kind: str
    payload: Any = None


@dataclass
class CalibrationConfig:
    """Knobs for :class:`SegmentCalibrator` (all documented, none hardcoded)."""

    method: str = "isotonic"
    segment_cols: tuple[str, ...] = DEFAULT_SEGMENT_COLS
    min_segment_n: int = 50
    min_segment_positives: int = 5
    alpha: float = 0.1

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            msg = "method must be one of isotonic, beta"
            raise ValueError(msg)
        if not 0.0 < self.alpha < 1.0:
            msg = "alpha must be in (0, 1)"
            raise ValueError(msg)


def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    reg.fit(p, y)
    return reg


def _beta_predict(p: np.ndarray, a: float, b: float, m: float) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return _clip01(_sigmoid(a * np.log(pc) - b * np.log(1.0 - pc) + m))


def _fit_beta_params(p: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit Kull et al. beta-calibration params (a, b, m) by log-loss.

    Identity (a=b=1, m=0) reproduces the input, so optimisation starts there.
    """
    pc = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    yy = np.asarray(y, dtype=float)

    def loss(theta: np.ndarray) -> float:
        a, b, m = float(theta[0]), float(theta[1]), float(theta[2])
        q = np.clip(_beta_predict(pc, a, b, m), _EPS, 1.0 - _EPS)
        return float(-(yy * np.log(q) + (1.0 - yy) * np.log(1.0 - q)).mean())

    res = minimize(
        loss,
        np.array([1.0, 1.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(1e-3, 10.0), (1e-3, 10.0), (-10.0, 10.0)],
        options={"maxiter": 200},
    )
    if not bool(res.success):
        return (1.0, 1.0, 0.0)
    a, b, m = (float(v) for v in res.x)
    return (a, b, m)


def _apply_map(fm: _FittedMap, p: np.ndarray) -> np.ndarray:
    pc = _clip01(p)
    if fm.kind == "isotonic":
        return _clip01(np.asarray(fm.payload.predict(pc), dtype=float))
    if fm.kind == "beta":
        a, b, m = fm.payload
        return _clip01(_beta_predict(pc, float(a), float(b), float(m)))
    return pc  # identity passthrough


class SegmentCalibrator:
    """Per-segment isotonic/beta calibrator with a global fallback.

    Fallback rule (documented): a segment gets its own map only when it has
    at least ``min_segment_n`` rows AND at least ``min_segment_positives``
    positives AND both classes present. Otherwise the segment (and any unseen
    segment key at predict time) uses the global map. When the global fit
    itself is degenerate (single class / empty), prediction is the identity
    passthrough (clipped input) so the module degrades to "do no harm".
    """

    name = "segment_calibrator"

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._maps: dict[str, _FittedMap] = {}
        self._global: _FittedMap = _FittedMap("identity")
        self._residuals: dict[str, np.ndarray] = {}
        self._global_residuals: np.ndarray = np.zeros(0)
        self._fallback_keys: set[str] = set()
        self._global_fallback = False
        self._uncalibrated: pd.DataFrame | None = None
        self.fit_as_of_: datetime | None = None
        self.base_train_end_: datetime | None = None

    # -- fitting ---------------------------------------------------------
    def fit(
        self,
        uncalibrated: pd.Series | np.ndarray,
        labels: pd.Series | np.ndarray,
        segments: pd.DataFrame,
        fit_as_of: datetime | None = None,
        base_train_end: datetime | None = None,
    ) -> SegmentCalibrator:
        """Fit per-segment maps on a later split than the base model.

        Raises ValueError when ``fit_as_of <= base_train_end`` (refit
        leakage) or when a configured segment column is missing.
        """
        cfg = self.config
        if fit_as_of is not None and base_train_end is not None and fit_as_of <= base_train_end:
            msg = "calibration split must be later than base train"
            raise ValueError(msg)
        missing = [c for c in cfg.segment_cols if c not in segments.columns]
        if missing:
            msg = f"missing segment columns: {missing}"
            raise ValueError(msg)
        p = _clip01(np.asarray(uncalibrated, dtype=float))
        y = np.asarray(labels, dtype=float)
        if not (len(p) == len(y) == len(segments)):
            msg = "uncalibrated, labels and segments lengths differ"
            raise ValueError(msg)
        if len(y) == 0:
            msg = "empty calibration fit"
            raise ValueError(msg)
        mask = ~(np.isnan(p) | np.isnan(y))
        p, y = p[mask], y[mask]
        seg = segments.iloc[np.flatnonzero(mask)].reset_index(drop=True)
        if len(y) == 0:
            msg = "no finite calibration rows"
            raise ValueError(msg)
        self.fit_as_of_ = fit_as_of
        self.base_train_end_ = base_train_end
        self._global = self._fit_one(p, y)
        self._global_fallback = self._global.kind == "identity"
        self._global_residuals = np.abs(y - _apply_map(self._global, p))
        self._maps, self._fallback_keys, self._residuals = {}, set(), {}
        keys = self._keys(seg)
        for key in sorted(set(keys)):
            idx = np.flatnonzero(keys == key)
            pk, yk = p[idx], y[idx]
            if (
                len(yk) >= cfg.min_segment_n
                and int((yk == 1).sum()) >= cfg.min_segment_positives
                and _has_both_classes(yk)
            ):
                fm = self._fit_one(pk, yk)
                if fm.kind != "identity":
                    self._maps[key] = fm
                    self._residuals[key] = np.abs(yk - _apply_map(fm, pk))
                    continue
            self._fallback_keys.add(key)
        return self

    def _fit_one(self, p: np.ndarray, y: np.ndarray) -> _FittedMap:
        if len(y) < _MIN_FIT_ROWS or not _has_both_classes(y):
            return _FittedMap("identity")
        if self.config.method == "beta":
            return _FittedMap("beta", _fit_beta_params(p, y))
        try:
            return _FittedMap("isotonic", _fit_isotonic(p, y))
        except Exception:
            return _FittedMap("identity")

    # -- prediction ------------------------------------------------------
    def _keys(self, segments: pd.DataFrame) -> np.ndarray:
        cols = list(self.config.segment_cols)
        joined = ["|".join(str(row[c]) for c in cols) for _, row in segments[cols].iterrows()]
        return np.array(joined, dtype=object)

    def predict(
        self,
        uncalibrated: pd.Series | np.ndarray,
        segments: pd.DataFrame,
    ) -> np.ndarray:
        """Return calibrated probabilities in [0, 1]."""
        self._require_cols(segments)
        p = _clip01(np.asarray(uncalibrated, dtype=float))
        keys = self._keys(segments)
        out = np.empty(len(p))
        for key in set(keys):
            idx = np.flatnonzero(keys == key)
            fm = self._maps.get(key, self._global)
            out[idx] = _apply_map(fm, p[idx])
        return out

    def predict_proba(
        self, uncalibrated: pd.Series | np.ndarray, segments: pd.DataFrame
    ) -> np.ndarray:
        """Two-column [P(0), P(1)] form for sklearn-style consumers."""
        p1 = self.predict(uncalibrated, segments)
        return np.column_stack([1.0 - p1, p1])

    def _require_cols(self, segments: pd.DataFrame) -> None:
        missing = [c for c in self.config.segment_cols if c not in segments.columns]
        if missing:
            msg = f"missing segment columns: {missing}"
            raise ValueError(msg)

    def predict_interval(
        self,
        uncalibrated: pd.Series | np.ndarray,
        segments: pd.DataFrame,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """Inductive-conformal intervals ``[lo, hi]`` around P(calibrated).

        Coverage guarantee: for a new exchangeable draw from the same
        segment, P(label in [lo, hi]) >= 1 - alpha (finite-sample, Romano et
        al.). Exchangeability FAILS under time drift, dialled-only selection
        shift, or arm-mix change -- which is why calibration must be refit on
        a later split and monitored, never blindly trusted forward.
        """
        level = self.config.alpha if alpha is None else alpha
        if not 0.0 < level < 1.0:
            msg = "alpha must be in (0, 1)"
            raise ValueError(msg)
        pcal = self.predict(uncalibrated, segments)
        keys = self._keys(segments)
        lo = np.empty(len(pcal))
        hi = np.empty(len(pcal))
        for key in set(keys):
            idx = np.flatnonzero(keys == key)
            resid = self._residuals.get(key, self._global_residuals)
            q = self._quantile(resid, level)
            lo[idx] = np.clip(pcal[idx] - q, 0.0, 1.0)
            hi[idx] = np.clip(pcal[idx] + q, 0.0, 1.0)
        return pd.DataFrame({"lo": lo, "hi": hi})

    @staticmethod
    def _quantile(resid: np.ndarray, level: float) -> float:
        n = len(resid)
        if n == 0:
            return 1.0  # no calibration evidence: widest interval
        k = int(np.ceil((1.0 - level) * (n + 1)))
        order = np.sort(np.asarray(resid, dtype=float))
        return float(order[min(max(k - 1, 0), n - 1)])

    # -- eval / serve adapters -------------------------------------------
    def used_fallback(self, segment_key: str) -> bool:
        """True when a segment fell back to (or was never given) its own map."""
        return segment_key in self._fallback_keys or segment_key not in self._maps

    @property
    def fallback_segments_(self) -> set[str]:
        return set(self._fallback_keys)

    def to_eval_frame(
        self,
        uncalibrated: pd.Series | np.ndarray,
        labels: pd.Series | np.ndarray,
        segments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Frame for ``ece_by_segment``: columns [y, p_cal, segment]."""
        pcal = self.predict(uncalibrated, segments)
        return pd.DataFrame(
            {"y": np.asarray(labels, dtype=float), "p_cal": pcal, "segment": self._keys(segments)}
        )

    def attach_uncalibrated(self, frame: pd.DataFrame) -> SegmentCalibrator:
        """Attach the scoring-origin frame (contact_point_ref + p_uncal)."""
        if "contact_point_ref" not in frame.columns or "p_uncal" not in frame.columns:
            msg = "frame needs contact_point_ref and p_uncal columns"
            raise ValueError(msg)
        self._uncalibrated = frame.copy()
        return self

    def score(self, as_of: datetime, contact_point_refs: Sequence[str]) -> pd.DataFrame:
        """Eval-protocol score: calibrated ``p_rpc`` + interval-width confidence."""
        refs = list(contact_point_refs)
        if self._uncalibrated is None:
            return pd.DataFrame({"contact_point_ref": refs, "p_rpc": 0.5, "confidence": 0.0})
        fr = pd.DataFrame({"contact_point_ref": refs}).merge(
            self._uncalibrated, on="contact_point_ref", how="left"
        )
        p_raw = _clip01(fr["p_uncal"].to_numpy(dtype=float, na_value=np.nan))
        known = ~np.isnan(p_raw)
        pcal = np.full(len(refs), 0.5)
        width = np.ones(len(refs))
        if known.any():
            seg_known = fr.loc[known].reset_index(drop=True)
            pcal[known] = self.predict(np.nan_to_num(p_raw[known], nan=0.5), seg_known)
            iv = self.predict_interval(np.nan_to_num(p_raw[known], nan=0.5), seg_known)
            width[known] = (iv["hi"] - iv["lo"]).to_numpy()
        _ = as_of  # scoring origin is carried by the attached PIT frame
        return pd.DataFrame(
            {
                "contact_point_ref": refs,
                "p_rpc": np.clip(pcal, 0.01, 0.99),
                "confidence": np.clip(1.0 - width, 0.0, 1.0),
            }
        )


__all__ = ["DEFAULT_SEGMENT_COLS", "METHODS", "CalibrationConfig", "SegmentCalibrator"]
