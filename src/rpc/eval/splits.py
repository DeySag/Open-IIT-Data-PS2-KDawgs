"""Rolling-origin time splits with embargo. Never random k-fold."""

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
