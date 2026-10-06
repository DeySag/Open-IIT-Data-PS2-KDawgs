"""Tests for the dial-selection propensity model (inline fixtures, no real data)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.rpc.models.propensity import (
    DialPropensityModel,
    PropensityConfig,
    PropensityValidationError,
)

RULE_ARM_SHARE = 0.6
N_FIXTURE_ROWS = 240
N_CORRUPT_ROWS = 10
N_SCORE_REFS = 12
EXPECTED_SLOTS_AT_200 = 10
EXPLORATION_N = 200


def make_frame(n: int = N_FIXTURE_ROWS, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arm = np.where(rng.uniform(0, 1, n) < RULE_ARM_SHARE, "rule_based", "random")
    k = rng.choice([1, 2, 3, 4], size=n).astype(float)
    logged = np.where(arm == "rule_based", 1.0, 1.0 / k)
    dialled = np.where(
        arm == "rule_based",
        1.0,
        (rng.uniform(0, 1, n) < logged).astype(float),
    )
    return pd.DataFrame(
        {
            "n_attempts": rng.integers(0, 8, n).astype(float),
            "answer_rate": rng.uniform(0, 1, n),
            "consec_failures": rng.integers(0, 5, n).astype(float),
            "dialling_arm": arm,
            "k_options": k,
            "selection_propensity": logged,
            "dialled": dialled,
        }
    )


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["dialled", "k_options", "selection_propensity"])


def test_fit_predict_roundtrip_rule_arm_is_one() -> None:
    df = make_frame()
    model = DialPropensityModel()
    model.fit(feature_frame(df), df["dialled"])
    p = model.predict_propensity(feature_frame(df))
    assert p.shape == (len(df),)
    assert bool(np.all((p > 0.0) & (p <= 1.0)))
    rule = df["dialling_arm"] == "rule_based"
    assert bool((p[rule.to_numpy()] == 1.0).all())


def test_validate_1k_pass_then_weights() -> None:
    df = make_frame()
    model = DialPropensityModel()
    feats = feature_frame(df)
    model.fit(feats, df["dialled"])
    report = model.validate_1k(df["k_options"], df["selection_propensity"], df["dialling_arm"])
    assert report.passed
    assert report.max_abs_dev == pytest.approx(0.0)
    w = model.ips_weights(features=feats, dialled=df["dialled"])
    assert w.shape == (len(df),)
    assert bool(np.all(np.isfinite(w)))
    assert bool((w <= model.config.max_weight + 1e-9).all())
    ess = DialPropensityModel.effective_sample_size(w)
    assert 0.0 < ess <= len(df)


def test_weights_refused_before_validation() -> None:
    df = make_frame()
    model = DialPropensityModel()
    feats = feature_frame(df)
    model.fit(feats, df["dialled"])
    with pytest.raises(PropensityValidationError, match="validate_1k"):
        model.ips_weights(features=feats)


def test_failed_1k_refuses_weights() -> None:
    df = make_frame()
    bad_logged = df["selection_propensity"].copy()
    bad_logged.iloc[:N_CORRUPT_ROWS] = 0.99  # corrupt the 1/k relationship
    model = DialPropensityModel(PropensityConfig(tol_1k=0.02))
    feats = feature_frame(df)
    model.fit(feats, df["dialled"])
    report = model.validate_1k(df["k_options"], bad_logged, df["dialling_arm"])
    assert not report.passed
    with pytest.raises(PropensityValidationError, match="1/k validation failed"):
        model.ips_weights(features=feats)


def test_degenerate_single_class_predicts_marginal() -> None:
    df = make_frame()
    df["dialled"] = 1.0
    model = DialPropensityModel()
    feats = feature_frame(df)
    model.fit(feats, df["dialled"])
    p = model.predict_propensity(feats)
    assert bool(np.all(np.isfinite(p)))
    non_rule = (df["dialling_arm"] != "rule_based").to_numpy()
    assert np.allclose(p[non_rule], model._marginal)


def test_decision_time_helpers() -> None:
    df = make_frame()
    model = DialPropensityModel()
    feats = feature_frame(df)
    model.fit(feats, df["dialled"])
    ledger = model.logged_propensity_for_decision(feats)
    assert list(ledger.columns)[:1] == ["logged_propensity"]
    assert bool(ledger["logged_propensity"].between(0.0, 1.0).all())
    assert DialPropensityModel.exploration_slots(EXPLORATION_N) == EXPECTED_SLOTS_AT_200
    assert DialPropensityModel.exploration_slots(0) == 0


def test_score_adapter_shape() -> None:
    df = make_frame()
    model = DialPropensityModel()
    feats = feature_frame(df).iloc[:N_SCORE_REFS].copy()
    refs = [f"cp{i:03d}" for i in range(N_SCORE_REFS)]
    feats["contact_point_ref"] = refs
    model.fit(feats.drop(columns=["contact_point_ref"]), df["dialled"].iloc[:N_SCORE_REFS])
    sub = df.iloc[:N_SCORE_REFS]
    model.validate_1k(sub["k_options"], sub["selection_propensity"])
    model.attach_context(feats)
    out = model.score(datetime(2026, 6, 1, tzinfo=UTC), refs)
    assert list(out.columns) == ["contact_point_ref", "propensity", "ips_weight", "confidence"]
    assert (out["contact_point_ref"] == refs).all()
