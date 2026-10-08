"""Tests for the eval-input exporter (pure mapping helpers, no DB)."""

from __future__ import annotations

import pandas as pd

from src.rpc.eval.prepare import (
    build_borrowers,
    build_contact_points,
    build_policy_log,
    build_verified_gold,
    hash_refs,
)

N_LINKS = 3
HASH_CHARS = 16
_HALF_PROPENSITY = 0.5


def test_hash_refs_stable_and_opaque() -> None:
    ids = pd.Series(["PH1", "PH1", "PH2"])
    got = hash_refs(ids, None)
    assert got.iloc[0] == got.iloc[1] != got.iloc[2]
    assert all(len(v) == HASH_CHARS for v in got)
    assert not got.str.contains("PH").any()


def test_contact_points_keys_and_primary() -> None:
    phones = pd.DataFrame({
        "phone_id": [f"PH{i}" for i in range(N_LINKS)],
        "account_id": ["A1"] * N_LINKS,
        "source": ["KYC", "bureau", "employer"],
        "priority_slot": ["0", "2", "1"],
        "added_date": ["2026-04-01"] * N_LINKS,
    })
    out = build_contact_points(phones, pd.Series({"A1": "L01"}), None)
    assert out["is_primary"].tolist() == [True, False, False]
    assert (out["borrower_id"] == "A1").all()
    assert (out["lender_id"] == "L01").all()
    assert out["contact_point_ref"].nunique() == N_LINKS


def test_borrowers_allowlist_only() -> None:
    accounts = pd.DataFrame({
        "account_id": ["A1"],
        "lender_id": ["L01"],
        "portfolio": ["X"],
        "true_state": ["valid"],  # hidden column must not pass
    })
    out = build_borrowers(accounts)
    assert out["borrower_id"].iloc[0] == "A1"
    assert "true_state" not in out.columns


def test_policy_log_exposure_shape() -> None:
    dials = pd.DataFrame({
        "phone_id": ["PH1"],
        "account_id": ["A1"],
        "dialling_arm": ["random_contact_point"],
        "selection_propensity": ["0.5"],
        "attempt_ts": ["2026-04-02 10:00:00"],
    })
    out = build_policy_log(dials, None)
    assert out["dialled"].all()
    assert out["selection_propensity"].iloc[0] == _HALF_PROPENSITY


def test_verified_gold_hashes_ids() -> None:
    verified = pd.DataFrame({
        "phone_id": ["PH1"],
        "account_id": ["A1"],
        "verified_status": ["third_party_number"],
    })
    out = build_verified_gold(verified, None)
    assert out["contact_point_ref"].iloc[0] == hash_refs(pd.Series(["PH1"]), None).iloc[0]
    assert "phone_id" not in out.columns
