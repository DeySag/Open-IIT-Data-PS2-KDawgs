"""Unit + leakage + boundary tests for the third-party risk scorer."""

from __future__ import annotations

import pandas as pd

from src.rpc.models.third_party import ThirdPartyRiskScorer, load_third_party_params

N_SOURCE_LINKS = 100
N_THIN_LINKS = 5
N_EMPLOYER_TP = 90
N_KYC_TP = 10
N_GOLD_EACH = 2


def _fit_frame() -> pd.DataFrame:
    rows = []
    # employer: 90/100 tp; kyc: 10/100 tp — wide, deliberate separation.
    for i in range(N_SOURCE_LINKS):
        rows.append({"source": "employer", "tp_ever": 1 if i < N_EMPLOYER_TP else 0})
    for i in range(N_SOURCE_LINKS):
        rows.append({"source": "kyc_origination", "tp_ever": 1 if i < N_KYC_TP else 0})
    return pd.DataFrame(rows)


def test_source_prior_ordering_and_smoothing() -> None:
    m = ThirdPartyRiskScorer().fit(_fit_frame())
    assert m.source_prior("employer") > m.source_prior("kyc_origination")
    assert 0.0 < m.source_prior("employer") < 1.0
    # unseen source falls back to global, never crashes
    assert m.source_prior("no_such_source") == m._global_prior


def test_thin_source_falls_back_to_global() -> None:
    df = pd.DataFrame([{"source": "rare_src", "tp_ever": 1}] * N_THIN_LINKS)
    params = load_third_party_params() | {"min_samples": 20}
    m = ThirdPartyRiskScorer(params).fit(df)
    assert m.source_prior("rare_src") == m._global_prior


def test_past_evidence_moves_posterior() -> None:
    m = ThirdPartyRiskScorer().fit(_fit_frame())
    links = pd.DataFrame([
        {"contact_point_ref": "clean", "source": "kyc_origination",
         "n_tp_past": 0, "n_attempts_past": 10},
        {"contact_point_ref": "dirty", "source": "kyc_origination",
         "n_tp_past": 3, "n_attempts_past": 10},
    ])
    out = m.score(links).set_index("contact_point_ref")["risk"]
    assert out["dirty"] > out["clean"]
    assert out.between(0.0, 1.0).all()


def test_banned_columns_never_read() -> None:
    m = ThirdPartyRiskScorer().fit(_fit_frame())
    base = pd.DataFrame([{"contact_point_ref": "P1", "source": "employer",
                           "n_tp_past": 1, "n_attempts_past": 4}])
    dirty = base.copy()
    dirty["verified_status"] = "third_party_number"
    dirty["true_state"] = "third_party"
    assert m.score(base)["risk"].iloc[0] == m.score(dirty)["risk"].iloc[0]


def test_gold_check_ranks_perfect_separation() -> None:
    m = ThirdPartyRiskScorer().fit(_fit_frame())
    verified = pd.DataFrame([
        {"contact_point_ref": "T1", "source": "employer", "n_tp_past": 5,
         "n_attempts_past": 5, "verified_status": "third_party_number"},
        {"contact_point_ref": "T2", "source": "employer", "n_tp_past": 4,
         "n_attempts_past": 5, "verified_status": "third_party_number"},
        {"contact_point_ref": "B1", "source": "kyc_origination", "n_tp_past": 0,
         "n_attempts_past": 5, "verified_status": "borrower_number"},
        {"contact_point_ref": "B2", "source": "kyc_origination", "n_tp_past": 0,
         "n_attempts_past": 6, "verified_status": "borrower_number"},
    ])
    got = m.gold_check(verified)
    assert got["n_positive"] == N_GOLD_EACH
    assert got["rank_auc"] == 1.0


def test_never_auto_decides() -> None:
    m = ThirdPartyRiskScorer().fit(_fit_frame())
    assert not hasattr(m, "decide")
    assert not hasattr(m, "predict_action")
    assert not hasattr(m, "threshold")
