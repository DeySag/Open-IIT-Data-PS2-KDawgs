"""Pre-ingest enrichment: join account-keyed source rows to ``lender_id``.

Official extracts are account-keyed only; the canonical envelope requires
``lender_id``, so every event source must be enriched from ``accounts.csv``
before ingest. Rows whose account has no lender stay null and are rejected
downstream as ``missing_required_field`` -- never silently dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class LenderLookupError(ValueError):
    """Accounts lookup unusable (carries the accounts path)."""


class DuplicateAccountError(LenderLookupError):
    """Duplicate account_id rows in the accounts file."""


class LenderJoinError(ValueError):
    """Source rows cannot be joined to lender_id (carries the source name)."""


def load_lender_lookup(accounts_path: str | Path) -> pd.Series:
    """Load the account_id -> lender_id lookup from ``accounts.csv``."""
    accounts = pd.read_csv(
        accounts_path,
        dtype="string",
        keep_default_na=True,
        usecols=["account_id", "lender_id"],
    )
    if accounts["account_id"].duplicated().any():
        raise DuplicateAccountError(accounts_path)
    if accounts["lender_id"].isna().any():
        raise LenderLookupError(accounts_path)
    return accounts.set_index("account_id")["lender_id"]


def attach_lender_id(
    raw: pd.DataFrame, lookup: pd.Series, *, source: str
) -> tuple[pd.DataFrame, int]:
    """Fill ``lender_id`` on account-keyed rows from the lookup.

    Returns (frame, n_unmatched). Existing non-null values are kept;
    unmatched accounts stay null for downstream rejection.
    """
    if "account_id" not in raw.columns:
        raise LenderJoinError(source)
    if "lender_id" in raw.columns and raw["lender_id"].notna().all():
        return raw, 0
    out = raw.copy()
    mapped = out["account_id"].map(lookup)
    if "lender_id" in out.columns:
        out["lender_id"] = out["lender_id"].fillna(mapped)
    else:
        out["lender_id"] = mapped
    n_unmatched = int(out["lender_id"].isna().sum())
    logger.info(
        "enrich source=%s rows=%d lender_matched=%d unmatched=%d",
        source,
        len(out),
        len(out) - n_unmatched,
        n_unmatched,
    )
    return out, n_unmatched
