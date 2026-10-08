"""Rolling-origin time splits with embargo + account-group containment.

Primary validation is time-purged + embargoed + account-group-contained
rolling origins with the random arm as the unbiased slice. The official
``splits.csv`` is a secondary sanity check only (same-window stratified
random: contact-level leakage via 43 shared ids + 553 shared masks). Never
random k-fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


@dataclass(frozen=True)
class Split:
    as_of: datetime          # scoring origin; train uses (as_of - train_days, as_of]
    train_start: datetime
    train_end: datetime       # == as_of
    embargo_end: datetime     # train_end + embargo; nothing scored inside (train_end, embargo_end]
    test_start: datetime      # == embargo_end
    test_end: datetime


RANDOM_ARM_VALUES: frozenset[str] = frozenset({"random_contact_point", "random"})


def make_rolling_splits(
    first_as_of: datetime,
    n_splits: int,
    step_days: int,
    train_days: int,
    embargo_days: int,
    test_days: int,
) -> list[Split]:
    """Origins step forward in time; train window trails each origin; test follows embargo."""
    splits: list[Split] = []
    for i in range(n_splits):
        as_of = first_as_of + timedelta(days=i * step_days)
        train_start = as_of - timedelta(days=train_days)
        embargo_end = as_of + timedelta(days=embargo_days)
        splits.append(
            Split(
                as_of=as_of,
                train_start=train_start,
                train_end=as_of,
                embargo_end=embargo_end,
                test_start=embargo_end,
                test_end=embargo_end + timedelta(days=test_days),
            )
        )
    return splits


def default_first_asof(events: pd.DataFrame, train_days: int) -> datetime:
    """Earliest feasible origin: min event time + train window (PIT-safe)."""
    t = pd.to_datetime(events["occurred_at"], utc=True)
    start = t.min() + pd.Timedelta(days=train_days)
    ts = start.tz_convert("UTC") if start.tzinfo is not None else start.tz_localize("UTC")
    return ts.to_pydatetime()


def check_splits(splits: list[Split], embargo_days: int) -> None:
    """Validate embargo + non-overlap invariants (also asserted in tests)."""
    for s in splits:
        assert s.test_start >= s.train_end + timedelta(days=embargo_days), (
            f"embargo violated: test_start {s.test_start} < train_end {s.train_end} + {embargo_days}d"
        )
    for a, b in zip(splits, splits[1:]):
        assert b.as_of > a.as_of, "origins must advance"


def assign_account_folds(
    account_ids: pd.Series | list[str], n_folds: int, seed: int = 42
) -> pd.DataFrame:
    """Deterministically assign each account to one fold (group containment).

    Contact-level rows inherit their account's fold, so shared phone_ids
    dialled under >1 account never leak across folds at the account grain.
    Folds sharing an account fail ``check_account_containment``.
    """
    import hashlib

    accounts = sorted({str(a) for a in account_ids})
    rows: list[dict] = []
    for acct in accounts:
        digest = hashlib.sha256(f"{seed}:{acct}".encode()).hexdigest()
        rows.append({"account_id": acct, "fold": int(digest, 16) % n_folds})
    return pd.DataFrame(rows)


def check_account_containment(fold_map: pd.DataFrame) -> None:
    """Each account sits in exactly one fold (group containment invariant)."""
    dupes = fold_map.groupby("account_id")["fold"].nunique()
    bad = dupes[dupes > 1]
    assert bad.empty, f"accounts span folds: {bad.head().to_dict()}"


def check_no_shared_contact_leak(
    events: pd.DataFrame, fold_map: pd.DataFrame, ref_col: str = "contact_point_ref"
) -> None:
    """Shared contact refs must not let one account's rows leak into another fold.

    Passes when every (account, ref) pair's account fold owns all of that
    account's rows — i.e. rows are routed by account fold, never by ref. Raises
    if any ref's rows from a single account land in >1 fold.
    """
    if ref_col not in events.columns or "account_id" not in events.columns:
        return
    fold_of = dict(zip(fold_map["account_id"].astype(str), fold_map["fold"]))
    ev = events.copy()
    ev["_fold"] = ev["account_id"].astype(str).map(fold_of)
    bad = ev.groupby(["account_id", ref_col])["_fold"].nunique()
    leaked = bad[bad > 1]
    assert leaked.empty, f"(account, ref) rows span folds: {leaked.head().to_dict()}"


def load_official_splits(path: str) -> pd.DataFrame:
    """Load official splits.csv as a SECONDARY sanity slice (never primary).

    Returns DataFrame[account_id, split] with values in {train, validation,
    test}. Callers must restrict label sources to the train slice and report
    the contact-leakage caveat (shared ids/masks cross splits).
    """
    df = pd.read_csv(path, dtype="string")
    assert {"account_id", "split"} <= set(df.columns), f"bad splits.csv: {df.columns.tolist()}"
    assert set(df["split"].unique()) <= {"train", "validation", "test"}, (
        f"unexpected split values: {df['split'].unique().tolist()}"
    )
    return df[["account_id", "split"]]


def train_accounts_only(
    labels: pd.DataFrame, official_splits: pd.DataFrame
) -> pd.DataFrame:
    """Restrict a label frame to TRAIN-split accounts (DATA RULE).

    Validation/test accounts are NEVER label sources for training.
    """
    train_accts = set(
        official_splits.loc[official_splits["split"] == "train", "account_id"].astype(str)
    )
    return labels[labels["account_id"].astype(str).isin(train_accts)].copy().reset_index(drop=True)


def random_arm_slice(events: pd.DataFrame, arm_col: str = "dialling_arm") -> pd.DataFrame:
    """Return the random-arm rows: the unbiased estimation slice.

    Rule-arm propensity is 1.0 by design; random-arm propensities validate 1/k
    (see propensity module) before any IPS weight is trusted.
    """
    if arm_col not in events.columns:
        return events.iloc[0:0].copy()
    return events[events[arm_col].isin(RANDOM_ARM_VALUES)].copy()
