"""Tests for the recycled-risk thin classifier (inline fixtures, no real data)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.rpc.models.recycled import (
    RecycledRiskScorer,
    merge_risk_into_scores,
)

WRONG_COUNT_CUT = 2
BUREAU_RATE_CUT = 0.22
N_REVIEW_REFS = 200
EXPECTED_REVIEWS_AT_1_PCT = 2
N_GOLD_POS = 9
N_GOLD_NEG = 90
N_GOLD_IGNORED = 21
N_GOLD_SCORED = 99
N_SCORE_REFS = 12


def make_frame(n: int = 240, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wrong = rng.integers(0, 4, n).astype(float)
    bureau_wrong = rng.uniform(0, 0.3, n)
    proxy = ((wrong >= WRONG_COUNT_CUT) | (bureau_wrong > BUREAU_RATE_CUT)).astype(float)
    return pd.DataFrame(
        {
            "n_attempts": rng.integers(0, 8, n).astype(float),
            "n_wrong_number": wrong,
            "bureau_wrong_rate": bureau_wrong,
            "answer_rate": rng.uniform(0, 1, n),
            "proxy_positive": proxy,
        }
    )


def test_fit_predict_roundtrip() -> None:
    df = make_frame()
    feats = df.drop(columns=["proxy_positive"])
    model = RecycledRiskScorer().fit(feats, df["proxy_positive"])
    assert not model.degenerate_
    risk = model.predict_risk(feats)
    assert risk.shape == (len(df),)
    assert bool(np.all((risk >= 0.0) & (risk <= 1.0)))
    assert bool(np.all(np.isfinite(risk)))
    assert risk[df["proxy_positive"].to_numpy() == 1.0].mean() >= risk.mean()


def test_unlabelled_rows_are_not_treated_as_negatives() -> None:
    df = make_frame()
    feats = df.drop(columns=["proxy_positive"])
    model = RecycledRiskScorer().fit(feats, df["proxy_positive"])
    risk = model.predict_risk(feats)
    # PU property: mean predicted risk must exceed the raw proxy-positive
    # rate (dividing by the label frequency c < 1 lifts positives above the
    # naive "unlabelled = negative" fit).
    assert risk.mean() >= model._proxy_rate


def test_outputs_scores_not_decisions() -> None:
    model = RecycledRiskScorer()
    for forbidden in ("decide", "predict_action", "predict_label", "threshold_", "action_"):
        assert not hasattr(model, forbidden), forbidden
    df = make_frame(n=N_REVIEW_REFS)
    feats = df.drop(columns=["proxy_positive"])
    model.fit(feats, df["proxy_positive"])
    refs = [f"cp{i:03d}" for i in range(N_REVIEW_REFS)]
    ranked = model.rank_for_review(refs, feats)
    assert list(ranked.columns) == ["contact_point_ref", "recycled_risk", "rank", "in_review"]
    assert "action" not in ranked.columns
    assert ranked["rank"].tolist() == list(range(1, N_REVIEW_REFS + 1))
    assert int(ranked["in_review"].sum()) == EXPECTED_REVIEWS_AT_1_PCT


def test_degenerate_all_unlabelled() -> None:
    df = make_frame()
    df["proxy_positive"] = 0.0
    model = RecycledRiskScorer().fit(df.drop(columns=["proxy_positive"]), df["proxy_positive"])
    assert model.degenerate_
    risk = model.predict_risk(df.drop(columns=["proxy_positive"]))
    assert bool(np.all(np.isfinite(risk)))
    assert bool(np.all((risk >= 0.0) & (risk <= 1.0)))


def test_evaluate_verified_hook() -> None:
    df = make_frame(n=120)
    feats = df.drop(columns=["proxy_positive"])
    model = RecycledRiskScorer().fit(feats, df["proxy_positive"])
    risk = model.predict_risk(feats)
    refs = [f"vcp{i:03d}" for i in range(120)]
    statuses = (
        ["not_borrower_number"] * N_GOLD_POS
        + ["borrower_number"] * N_GOLD_NEG
        + ["third_party_number"] * 15
        + ["switched_off"] * 4
        + ["invalid_number"] * 2
    )
    verified = pd.DataFrame({"contact_point_ref": refs, "verified_status": statuses})
    out = model.evaluate_verified(verified, risk)
    assert out["n_pos"] == N_GOLD_POS
    assert out["n_neg"] == N_GOLD_NEG
    assert out["n_ignored"] == N_GOLD_IGNORED
    assert out["n_scored"] == N_GOLD_SCORED
    assert np.isfinite(out["pr_auc"])


def test_merge_risk_into_scores_keeps_p_rpc() -> None:
    scores = pd.DataFrame(
        {"contact_point_ref": ["a", "b"], "p_rpc": [0.7, 0.2], "confidence": [0.8, 0.6]}
    )
    risk = pd.DataFrame({"contact_point_ref": ["a", "b"], "recycled_risk": [0.05, 0.9]})
    out = merge_risk_into_scores(scores, risk)
    assert out["p_rpc"].tolist() == [0.7, 0.2]
    assert out["recycled_risk"].tolist() == [0.05, 0.9]


def test_score_adapter_shape() -> None:
    df = make_frame(n=N_SCORE_REFS)
    feats = df.drop(columns=["proxy_positive"]).copy()
    refs = [f"cp{i:03d}" for i in range(N_SCORE_REFS)]
    feats["contact_point_ref"] = refs
    features_only = feats.drop(columns=["contact_point_ref"])
    model = RecycledRiskScorer().fit(features_only, df["proxy_positive"])
    model.attach_features(feats)
    out = model.score(datetime(2026, 6, 1, tzinfo=UTC), refs)
    assert list(out.columns) == ["contact_point_ref", "recycled_risk", "confidence"]
    assert (out["contact_point_ref"] == refs).all()
