"""Dimension loaders for official extracts.

Each loader reads a source CSV, tags it with ``loaded_at``, and inserts
idempotently into the matching store table. None of these tables are
mutated by event ingest; they are rebuilt from scratch on every ``make data``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.rpc.ingest.store import EventStore


def _stamp(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["loaded_at"] = datetime.now(UTC)
    return frame


def load_accounts(store: EventStore, path: str | Path, quarantined_fields: list[str] | None = None) -> int:
    """Load accounts.csv as the accounts dimension.

    The official extract column names differ from the canonical store schema,
    so we map them explicitly before insertion. This prevents positional inserts
    from shuffling values like ``town_id`` into ``outstanding``.
    """
    frame = pd.read_csv(path, dtype="string", keep_default_na=True)
    if "account_id" not in frame.columns:
        raise ValueError(f"accounts.csv missing account_id: {path}")

    source_map = {
        "account_id": "account_id",
        "lender_id": "lender_id",
        "portfolio": "product",
        "income_type": "income_type",
        "town_id": "town_id",
        "preferred_language": "preferred_language",
        "bucket_start": "dpd_bucket",
        "dpd_start": "dpd_start",
        "emi_amount": "emi_amount",
        "overdue_start": "overdue_start",
        "outstanding": "outstanding",
        "salary_credit_day": "salary_credit_day",
        "bureau_score_band": "bureau_score_band",
        "other_active_loans": "other_active_loans",
        "paid_other_lenders_30d": "paid_other_lenders_30d",
        "last_bounce_reason": "last_bounce_reason",
        "ability_to_pay_estimate": "ability_to_pay_estimate",
        "prev_ptp_count": "prev_ptp_count",
        "prev_ptp_broken": "prev_ptp_broken",
    }
    mapped = frame.rename(columns=source_map).copy()
    mapped["borrower_id"] = mapped["account_id"]
    mapped["quarantined_fields"] = ",".join(quarantined_fields or [])
    mapped["as_of_confirmed"] = False

    # Preserve only the canonical dimension schema.
    canonical_cols = [
        "account_id",
        "lender_id",
        "borrower_id",
        "dpd_bucket",
        "outstanding",
        "product",
        "bureau_score_band",
        "income_type",
        "preferred_language",
        "town_id",
        "dpd_start",
        "overdue_start",
        "emi_amount",
        "other_active_loans",
        "paid_other_lenders_30d",
        "last_bounce_reason",
        "salary_credit_day",
        "ability_to_pay_estimate",
        "prev_ptp_count",
        "prev_ptp_broken",
        "quarantined_fields",
        "as_of_confirmed",
    ]
    return store.load_accounts(_stamp(mapped[canonical_cols]))


def load_lenders(store: EventStore, path: str | Path) -> int:
    frame = pd.read_csv(path, dtype="string", keep_default_na=True)
    return store.load_lenders(_stamp(frame))


def load_agents(store: EventStore, path: str | Path) -> int:
    frame = pd.read_csv(path, dtype="string", keep_default_na=True)
    return store.load_agents(_stamp(frame))


def load_splits(store: EventStore, path: str | Path) -> int:
    frame = pd.read_csv(path, dtype="string", keep_default_na=True)
    if "account_id" not in frame.columns:
        raise ValueError(f"splits.csv missing account_id: {path}")
    return store.load_splits(_stamp(frame))


def load_verified_contact_points(store: EventStore, path: str | Path) -> int:
    """Load verified_contact_points.csv as held-out gold (eval only).

    The source file stores a raw phone ID, not the canonical contact-point hash,
    and it does not carry lender_id explicitly. We derive both before insert.
    """
    frame = pd.read_csv(path, dtype="string", keep_default_na=True)
    if "account_id" not in frame.columns:
        raise ValueError(f"verified_contact_points.csv missing account_id: {path}")
    if "phone_id" not in frame.columns:
        raise ValueError(f"verified_contact_points.csv missing phone_id: {path}")

    lookup = store.read_dimension("accounts")[['account_id', 'lender_id']].drop_duplicates()
    if not lookup.empty:
        lender_map = lookup.set_index('account_id')['lender_id']
        frame['lender_id'] = frame['account_id'].map(lender_map)
    else:
        frame['lender_id'] = pd.NA

    mapped = frame.rename(columns={'phone_id': 'contact_point_ref'}).copy()
    mapped['contact_point_ref'] = mapped['contact_point_ref'].astype(str)
    mapped['true_state'] = pd.NA
    mapped['loaded_at'] = datetime.now(UTC)

    required = [
        'contact_point_ref',
        'account_id',
        'lender_id',
        'verified_status',
        'verified_date',
        'true_state',
        'loaded_at',
    ]
    return store.load_verified_contact_points(mapped[required])
