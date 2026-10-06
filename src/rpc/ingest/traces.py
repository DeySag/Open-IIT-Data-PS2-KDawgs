"""Skip-trace outcome history loader.

``skip_traces.csv`` is VOI outcome history -- observed
``cost_inr`` / result pairs for the trace-value calculation. It is stored in
its own ``trace_history`` table and must never be read as a model predictor.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.rpc.ingest.store import EventStore, IngestConfig

logger = logging.getLogger(__name__)


class TraceHistoryError(ValueError):
    """Trace file unloadable (carries the file path)."""

REQUIRED_TRACE_COLUMNS = (
    "trace_id",
    "account_id",
    "trace_date",
    "trigger_rule",
    "result",
    "new_contact_point_id",
    "cost_inr",
)


def load_trace_history(
    path: str | Path,
    db_path: str | Path | None = None,
    config: IngestConfig | None = None,
) -> dict[str, int]:
    """Load trace outcomes into the store; returns {loaded, duplicate, rejected}."""
    started = datetime.now(UTC)
    raw = pd.read_csv(path, dtype="string", keep_default_na=True)
    missing = [c for c in REQUIRED_TRACE_COLUMNS if c not in raw.columns]
    if missing:
        raise TraceHistoryError(path)

    frame = raw[list(REQUIRED_TRACE_COLUMNS)].copy()
    frame["trace_date"] = pd.to_datetime(frame["trace_date"], errors="coerce").dt.date
    frame["cost_inr"] = pd.to_numeric(frame["cost_inr"], errors="coerce")
    bad = frame["trace_id"].isna() | frame["account_id"].isna() | frame["trace_date"].isna()
    rejected = frame[bad]
    accepted = frame[~bad].drop_duplicates(subset="trace_id", keep="first")
    within_dupes = len(frame) - len(rejected) - len(accepted)

    store = EventStore((config.db_path if config else None) or db_path)
    try:
        inserted, store_dupes = store.insert_trace_history(accepted)
    finally:
        store.close()

    result = {
        "loaded": inserted,
        "duplicate": store_dupes + within_dupes,
        "rejected": len(rejected),
    }
    logger.info(
        "trace-history path=%s rows=%d loaded=%d duplicate=%d rejected=%d elapsed_s=%.1f",
        path,
        len(raw),
        result["loaded"],
        result["duplicate"],
        result["rejected"],
        (datetime.now(UTC) - started).total_seconds(),
    )
    return result


def read_trace_history(
    account_ids: list[str] | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read trace outcomes (VOI inputs only; never a predictor)."""
    store = EventStore(db_path)
    try:
        return store.read_trace_history(account_ids=account_ids)
    finally:
        store.close()
