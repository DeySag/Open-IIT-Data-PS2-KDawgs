"""P2 label + split leakage tests (all fixtures; no real CN rows pasted)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.rpc.eval import labels as L
from src.rpc.eval.splits import (
    assign_account_folds,
    check_account_containment,
    check_no_shared_contact_leak,
    check_splits,
    load_official_splits,
    make_rolling_splits,
    random_arm_slice,
    train_accounts_only,
)


def utc(y: int, m: int, d: int, h: int = 0) -> datetime:
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def _canon_events(rows: list[dict]) -> pd.DataFrame:
    """Build canonical dial+disposition companions from raw-ish attempt rows."""
    ev: list[dict] = []
    for i, r in enumerate(rows):
        ts = r["ts"]
        for kind in ("dial", "disp"):
            payload = (
                {"network_response": r["nr"], "ring_seconds": 5.0}
                if kind == "dial"
                else {"disposition": r["disp"]}
            )
            ev.append(
                {
                    "event_id": f"e{i}-{kind}",
                    "event_type": "dial_attempt" if kind == "dial" else "disposition",
                    "lender_id": "L1",
                    "borrower_id": r["account"],
                    "account_id": r["account"],
                    "contact_point_ref": r["ref"],
                    "occurred_at": ts.isoformat(),
                    "received_at": (ts + timedelta(minutes=1)).isoformat(),
                    "payload": json.dumps(payload),
                }
            )
    return pd.DataFrame(ev)


def _attempts() -> pd.DataFrame:
    t = utc(2026, 4, 10, 10)
    return _canon_events(
        [
            {"account": "A1", "ref": "P1", "ts": t, "nr": "answered", "disp": "promise_to_pay"},
            {"account": "A1", "ref": "P2", "ts": t, "nr": "answered", "disp": "third_party"},
            {"account": "A1", "ref": "P3", "ts": t, "nr": "answered", "disp": "RPC"},
            {"account": "A1", "ref": "P4", "ts": t, "nr": "answered", "disp": "callback"},
            {"account": "A1", "ref": "P5", "ts": t, "nr": "answered", "disp": "dispute"},
            # 24-pattern analogue: rpc-like disposition on non-answered network.
            {"account": "A1", "ref": "P6", "ts": t, "nr": "no_answer", "disp": "promise_to_pay"},
            # Ambiguous: language_barrier on answered network.
            {"account": "A1", "ref": "P7", "ts": t, "nr": "answered", "disp": "language_barrier"},
            # Raw answered alone is not enough: covered by P2/P7 above.
        ]
    )


def test_rpc_requires_answered_and_sanctioned() -> None:
    ev = _attempts()
    lab = L.observed_labels(ev, utc(2026, 4, 9), ["P1", "P2", "P3", "P4", "P5", "P6", "P7"])
    got = lab.set_index("contact_point_ref")["rpc_next_7d"].to_dict()
    assert got["P1"] == 1.0  # answered + promise_to_pay
    assert got["P2"] == 0.0  # answered but third_party, not RPC
    assert got["P3"] == 1.0  # answered + RPC
    assert got["P4"] == 1.0
    assert got["P5"] == 1.0  # dispute counts (old code missed it)
    assert got["P6"] == 0.0  # quarantined: non-answered + rpc-like
    assert got["P7"] == 0.0  # language_barrier ambiguous, excluded


def test_raw_answered_alone_is_not_label() -> None:
    raw = pd.DataFrame(
        [
            {"account_id": "A1", "phone_id": "P1", "attempt_ts": "2026-04-10 10:00:00",
             "network_response": "answered", "disposition": "third_party_contact"},
            {"account_id": "A1", "phone_id": "P2", "attempt_ts": "2026-04-10 10:05:00",
             "network_response": "answered", "disposition": "rpc_ptp"},
        ]
    )
    at = L.attempt_table_from_raw(raw)
    assert bool(at.set_index("phone_id").loc["P1", "is_rpc"]) is False
    assert bool(at.set_index("phone_id").loc["P2", "is_rpc"]) is True


def test_strict_variant_drops_hung_up_refused() -> None:
    assert L.attempt_is_rpc_raw("answered", "rpc_hung_up", strict=False) is True
    assert L.attempt_is_rpc_raw("answered", "rpc_hung_up", strict=True) is False
    assert L.attempt_is_rpc_raw("answered", "rpc_refused", strict=True) is False
    assert L.attempt_is_rpc_raw("answered", "rpc_ptp", strict=True) is True
    assert L.attempt_is_rpc_raw("answered", "rpc_hardship", strict=True) is True
    ev = _canon_events(
        [
            {"account": "A1", "ref": "P1", "ts": utc(2026, 4, 10, 10),
             "nr": "answered", "disp": "RPC"},
            {"account": "A1", "ref": "P2", "ts": utc(2026, 4, 10, 10),
             "nr": "answered", "disp": "promise_to_pay"},
        ]
    )
    lab = L.observed_labels(ev, utc(2026, 4, 9), ["P1", "P2"])
    strict_col = "rpc_next_7d_strict"
    got = lab.set_index("contact_point_ref")[strict_col].to_dict()
    assert got["P1"] == 0.0  # canonical RPC collapsed -> excluded from strict
    assert got["P2"] == 1.0


def test_undialled_censored_never_negative() -> None:
    ev = _attempts()
    lab = L.observed_labels(ev, utc(2026, 4, 9), ["P1", "P9"])
    row = lab.set_index("contact_point_ref").loc["P9"]
    assert bool(row["censored"]) is True
    assert pd.isna(row["rpc_next_7d"])
    train = L.train_labels_only(lab)
    assert "P9" not in set(train["contact_point_ref"])


def test_attribution_per_account_phone_not_phone_alone() -> None:
    t = utc(2026, 4, 10, 10)
    ev = _canon_events(
        [
            {"account": "A1", "ref": "SHARED", "ts": t, "nr": "answered", "disp": "promise_to_pay"},
            {"account": "A2", "ref": "SHARED", "ts": t, "nr": "answered", "disp": "third_party"},
        ]
    )
    keys = pd.DataFrame(
        [{"account_id": "A1", "contact_point_ref": "SHARED"},
         {"account_id": "A2", "contact_point_ref": "SHARED"}]
    )
    lab = L.observed_labels(ev, utc(2026, 4, 9), keys=keys)
    got = lab.set_index("account_id")["rpc_next_7d"].to_dict()
    assert got["A1"] == 1.0
    assert got["A2"] == 0.0


def test_verified_holdout_never_trains() -> None:
    ev = _attempts()
    verified = pd.DataFrame([{"account_id": "A1", "contact_point_ref": "P1"}])
    lab = L.observed_labels(ev, utc(2026, 4, 9), ["P1", "P3"], verified_keys=verified)
    assert bool(lab.set_index("contact_point_ref").loc["P1", "verified_holdout"]) is True
    train = L.train_labels_only(lab)
    assert "P1" not in set(train["contact_point_ref"])
    assert "P3" in set(train["contact_point_ref"])


def test_post_cutoff_payments_never_features() -> None:
    pays = pd.DataFrame(
        [
            {"account_id": "A1", "payment_ts": "2026-04-09 10:00:00", "amount": 100.0},
            {"account_id": "A1", "payment_ts": "2026-05-01 10:00:00", "amount": 100.0},
        ]
    )
    raw = pd.DataFrame(
        [{"account_id": "A1", "phone_id": "P1", "attempt_ts": "2026-04-10 10:00:00",
          "network_response": "answered", "disposition": "rpc_ptp"}]
    )
    # as_of before the late payment: only the early window is visible.
    weak_early = L.payment_weak_labels(pays, raw, utc(2026, 4, 11),
                                       keys=pd.DataFrame([{"account_id": "A1", "contact_point_ref": "P1"}]))
    assert set(weak_early.columns) >= {"pay_weak"}
    # Late-only payment after as_of must not create a weak label.
    pays_late = pays.iloc[1:2]
    weak_late = L.payment_weak_labels(pays_late, raw, utc(2026, 4, 11),
                                      keys=pd.DataFrame([{"account_id": "A1", "contact_point_ref": "P1"}]))
    assert bool(weak_late.iloc[0]["pay_weak"]) is False


def test_train_labels_come_from_train_split_only(tmp_path) -> None:
    sp = tmp_path / "splits.csv"
    pd.DataFrame(
        [{"account_id": "A1", "split": "train"}, {"account_id": "A2", "split": "test"}]
    ).to_csv(sp, index=False)
    official = load_official_splits(str(sp))
    lab = pd.DataFrame(
        [{"account_id": "A1", "contact_point_ref": "P1", "rpc_next_7d": 1.0},
         {"account_id": "A2", "contact_point_ref": "P2", "rpc_next_7d": 1.0}]
    )
    tr = train_accounts_only(lab, official)
    assert set(tr["account_id"]) == {"A1"}


def test_splits_embargo_account_containment_and_random_arm() -> None:
    splits = make_rolling_splits(utc(2026, 4, 20), 2, 7, 14, 3, 7)
    check_splits(splits, 3)
    for s in splits:
        assert (s.test_start - s.train_end).days >= 3
    folds = assign_account_folds(pd.Series(["A1", "A2", "A3", "A1"]), 2, seed=7)
    assert set(folds["account_id"]) == {"A1", "A2", "A3"}
    check_account_containment(folds)
    ev = pd.DataFrame(
        [{"account_id": "A1", "contact_point_ref": "P1", "dialling_arm": "rule_based"},
         {"account_id": "A2", "contact_point_ref": "P2", "dialling_arm": "random_contact_point"}]
    )
    rnd = random_arm_slice(ev)
    assert set(rnd["account_id"]) == {"A2"}
    check_no_shared_contact_leak(
        pd.DataFrame([{"account_id": "A1", "contact_point_ref": "P1"}]),
        pd.DataFrame([{"account_id": "A1", "fold": 0}]),
    )
    with pytest.raises(AssertionError):
        check_account_containment(
            pd.DataFrame([{"account_id": "A1", "fold": 0}, {"account_id": "A1", "fold": 1}])
        )


def test_ever_rpc_respects_as_of() -> None:
    ev = _attempts()
    hist = L.ever_rpc_labels(ev, utc(2026, 4, 11), ["P1", "P2", "P9"])
    got = hist.set_index("contact_point_ref")
    assert got.loc["P1", "ever_rpc"] == 1.0
    assert got.loc["P2", "ever_rpc"] == 0.0
    assert bool(got.loc["P9", "censored"]) is True
    before = L.ever_rpc_labels(ev, utc(2026, 4, 1), ["P1"])
    assert bool(before.iloc[0]["censored"]) is True
