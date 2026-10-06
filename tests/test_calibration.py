"""Tests for the per-segment calibrator (inline fixtures only, no real data)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.rpc.models.calibration import CalibrationConfig, SegmentCalibrator

SEG_COLS = ["dialling_arm", "lender", "recency_bucket"]
RULE_ARM_SHARE = 0.7
LENDER_SHARE = 0.5
RECENCY_CUT = 0.5
MIN_TRAIN_COVERAGE = 0.5
N_FALLBACK_ROWS = 60
N_DEGENERATE_ROWS = 80
N_SCORE_REFS = 10


def make_frame(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < 0.3 + 0.4 * p).astype(float)
    return pd.DataFrame(
        {
            "p_uncal": p,
            "y": y,
            "dialling_arm": np.where(rng.uniform(0, 1, n) < RULE_ARM_SHARE, "rule_based", "random"),
            "lender": np.where(rng.uniform(0, 1, n) < LENDER_SHARE, "L01", "L02"),
            "recency_bucket": np.where(p > RECENCY_CUT, "recent", "stale"),
        }
    )


def small_config(**overrides: object) -> CalibrationConfig:
    base: dict[str, object] = {"min_segment_n": 40, "min_segment_positives": 3}
    base.update(overrides)
    return CalibrationConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["isotonic", "beta"])
def test_fit_predict_roundtrip(method: str) -> None:
    df = make_frame()
    cal = SegmentCalibrator(small_config(method=method))
    cal.fit(df["p_uncal"], df["y"], df[SEG_COLS])
    pcal = cal.predict(df["p_uncal"], df[SEG_COLS])
    assert pcal.shape == (len(df),)
    assert bool(np.all((pcal >= 0.0) & (pcal <= 1.0)))
    assert bool(np.all(np.isfinite(pcal)))
    proba = cal.predict_proba(df["p_uncal"], df[SEG_COLS])
    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_fallback_triggers_on_tiny_segments() -> None:
    df = make_frame(n=N_FALLBACK_ROWS)
    cal = SegmentCalibrator(CalibrationConfig(min_segment_n=10_000, min_segment_positives=1_000))
    cal.fit(df["p_uncal"], df["y"], df[SEG_COLS])
    assert len(cal.fallback_segments_) > 0
    key = sorted(cal.fallback_segments_)[0]
    assert cal.used_fallback(key)
    pcal = cal.predict(df["p_uncal"], df[SEG_COLS])
    assert bool(np.all(np.isfinite(pcal)))


def test_unseen_segment_key_uses_global_map() -> None:
    df = make_frame()
    cal = SegmentCalibrator(small_config())
    cal.fit(df["p_uncal"], df["y"], df[SEG_COLS])
    new_seg = df[SEG_COLS].iloc[:5].copy()
    new_seg["lender"] = "L99"
    pcal = cal.predict(df["p_uncal"].iloc[:5], new_seg)
    assert bool(np.all(np.isfinite(pcal)))


def test_degenerate_single_class_is_identity() -> None:
    df = make_frame(n=N_DEGENERATE_ROWS)
    df["y"] = 0.0
    cal = SegmentCalibrator(CalibrationConfig(min_segment_n=10, min_segment_positives=2))
    cal.fit(df["p_uncal"], df["y"], df[SEG_COLS])
    pcal = cal.predict(df["p_uncal"], df[SEG_COLS])
    assert np.allclose(pcal, np.clip(df["p_uncal"].to_numpy(), 0.0, 1.0))


def test_later_split_enforced() -> None:
    df = make_frame(n=N_DEGENERATE_ROWS)
    cal = SegmentCalibrator()
    base_end = datetime(2026, 5, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="later than base"):
        cal.fit(
            df["p_uncal"],
            df["y"],
            df[SEG_COLS],
            fit_as_of=base_end,
            base_train_end=base_end,
        )


def test_conformal_interval_properties() -> None:
    df = make_frame()
    cal = SegmentCalibrator(small_config(alpha=0.2))
    cal.fit(df["p_uncal"], df["y"], df[SEG_COLS])
    iv = cal.predict_interval(df["p_uncal"], df[SEG_COLS])
    assert list(iv.columns) == ["lo", "hi"]
    assert bool((iv["lo"] <= iv["hi"]).all())
    assert bool((iv["lo"] >= 0.0).all()) and bool((iv["hi"] <= 1.0).all())
    pcal = cal.predict(df["p_uncal"], df[SEG_COLS])
    assert bool(((iv["lo"] <= pcal) & (pcal <= iv["hi"])).all())
    y = df["y"].to_numpy()
    covered = ((y >= iv["lo"].to_numpy()) & (y <= iv["hi"].to_numpy())).mean()
    assert covered > MIN_TRAIN_COVERAGE  # weak sanity on-train, not a guarantee


def test_score_adapter_shape() -> None:
    df = make_frame()
    cal = SegmentCalibrator(small_config())
    cal.fit(df["p_uncal"], df["y"], df[SEG_COLS])
    refs = [f"cp{i:03d}" for i in range(N_SCORE_REFS)]
    uncal = pd.DataFrame(
        {
            "contact_point_ref": refs,
            "p_uncal": np.linspace(0.1, 0.9, N_SCORE_REFS),
            "dialling_arm": "random",
            "lender": "L01",
            "recency_bucket": "recent",
        }
    )
    cal.attach_uncalibrated(uncal)
    out = cal.score(datetime(2026, 6, 1, tzinfo=UTC), refs)
    assert list(out.columns) == ["contact_point_ref", "p_rpc", "confidence"]
    assert (out["contact_point_ref"] == refs).all()
    assert bool(out["p_rpc"].between(0.01, 0.99).all())
