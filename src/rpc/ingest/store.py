"""DuckDB-backed append-only event store.

Tables
------
events
    ``event_id`` primary key plus canonical envelope columns, ``payload`` JSON
    and ``ingested_at``. Deduplicated on ``event_id`` keeping the earliest
    ``received_at``.
dead_letter
    Redacted raw rows that failed validation, keyed by a deterministic
    ``row_hash`` so replaying a batch never duplicates dead-letter rows.
dirty_contact_points
    ``(contact_point_ref, lender_id)`` pairs needing feature correction after
    a late event arrived past their scoring watermark.
watermarks
    Per-contact-point scoring watermarks written by downstream consumers
    (feature pipeline); read by ingestion to detect late events.

All bulk operations are SQL over staged views -- no per-row Python loops.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

# Rows per bulk-insert chunk (bounds peak memory on wide batches).
_INSERT_CHUNK_ROWS = 500_000

EVENT_COLUMNS = [
    "event_id",
    "event_type",
    "lender_id",
    "borrower_id",
    "account_id",
    "contact_point_ref",
    "occurred_at",
    "received_at",
    "payload",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR PRIMARY KEY,
    event_type VARCHAR NOT NULL,
    lender_id VARCHAR NOT NULL,
    borrower_id VARCHAR NOT NULL,
    account_id VARCHAR NOT NULL,
    contact_point_ref VARCHAR NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    payload VARCHAR NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS dead_letter (
    row_hash VARCHAR PRIMARY KEY,
    raw_json VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS dirty_contact_points (
    contact_point_ref VARCHAR NOT NULL,
    lender_id VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    marked_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (contact_point_ref, lender_id)
);
CREATE TABLE IF NOT EXISTS watermarks (
    contact_point_ref VARCHAR NOT NULL,
    lender_id VARCHAR NOT NULL,
    watermark_ts TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (contact_point_ref, lender_id)
);
CREATE TABLE IF NOT EXISTS trace_history (
    trace_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    trace_date DATE NOT NULL,
    trigger_rule VARCHAR NOT NULL,
    result VARCHAR NOT NULL,
    new_contact_point_id VARCHAR,
    cost_inr DOUBLE NOT NULL
);
"""


def default_db_path() -> str:
    """Database file path: ``INGEST_DB_PATH`` env var or the repo default."""
    return os.environ.get("INGEST_DB_PATH", str(Path("data") / "event_store.duckdb"))


@dataclass(frozen=True)
class IngestConfig:
    """Ingestion parameters (config, not constants)."""

    db_path: str = ""
    clock_skew_tolerance_seconds: float = 300.0
    sample_rate: float = 0.01
    sample_seed: int = 42
    chunk_rows: int = 500_000

    def __post_init__(self) -> None:
        if not self.db_path:
            object.__setattr__(self, "db_path", default_db_path())

    @classmethod
    def from_yaml(cls, path: str | Path) -> IngestConfig:
        with Path(path).open() as f:
            data = yaml.safe_load(f) or {}
        section = data.get("ingest", data)
        known = {k: v for k, v in section.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class EventStore:
    """Thin wrapper around a DuckDB event-store file."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path or default_db_path())
        parent = Path(self.db_path).parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.db_path)
        self.con.execute(_SCHEMA)

    def close(self) -> None:
        self.con.close()

    # -- writes ---------------------------------------------------------

    def insert_canonical(self, df: pd.DataFrame, ingested_at: datetime) -> tuple[int, int]:
        """Bulk insert accepted events; returns (inserted, duplicates).

        Deduplicates on ``event_id``: rows already present are skipped and
        counted, except when the staged row has an earlier ``received_at``,
        in which case the stored row is refreshed to the earliest version.
        """
        if df.empty:
            return 0, 0
        inserted = 0
        duplicates = 0
        staged = df[EVENT_COLUMNS].copy()
        staged["ingested_at"] = ingested_at
        self.con.register("_staged_all", staged)
        try:
            chunk = _INSERT_CHUNK_ROWS
            total_chunks = (len(staged) + chunk - 1) // chunk
            for i in range(total_chunks):
                offset = i * chunk
                self.con.execute(
                    "CREATE OR REPLACE TEMP VIEW staged AS "
                    f"SELECT * FROM _staged_all LIMIT {chunk} OFFSET {offset}"
                )
                dupes = self.con.execute(
                    "SELECT COUNT(*) FROM staged s WHERE EXISTS "
                    "(SELECT 1 FROM events e WHERE e.event_id = s.event_id)"
                ).fetchone()[0]
                duplicates += int(dupes)
                # Keep-earliest refresh for already-stored ids.
                self.con.execute(
                    "UPDATE events SET "
                    "event_type = s.event_type, lender_id = s.lender_id, "
                    "borrower_id = s.borrower_id, account_id = s.account_id, "
                    "contact_point_ref = s.contact_point_ref, "
                    "occurred_at = s.occurred_at, received_at = s.received_at, "
                    "payload = s.payload "
                    "FROM staged s "
                    "WHERE events.event_id = s.event_id "
                    "AND s.received_at < events.received_at"
                )
                # Rows inserted = staged rows minus already-stored ids (no
                # full-table COUNT scans; those scale with the stored table).
                staged_n = min(chunk, len(staged) - offset)
                self.con.execute(
                    "INSERT INTO events SELECT s.* FROM staged s "
                    "WHERE NOT EXISTS "
                    "(SELECT 1 FROM events e WHERE e.event_id = s.event_id)"
                )
                inserted += int(staged_n - dupes)
        finally:
            self.con.unregister("_staged_all")
        return inserted, duplicates

    def insert_dead_letter(self, rows: pd.DataFrame) -> int:
        """Bulk insert dead-letter rows (idempotent via ``row_hash``)."""
        if rows.empty:
            return 0
        self.con.register(
            "_dead_staged",
            rows[["row_hash", "raw_json", "reason", "source", "ingested_at"]],
        )
        try:
            before = self.con.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0]
            self.con.execute("INSERT OR IGNORE INTO dead_letter SELECT * FROM _dead_staged")
            after = self.con.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0]
        finally:
            self.con.unregister("_dead_staged")
        return int(after - before)

    def mark_dirty(self, df: pd.DataFrame, marked_at: datetime) -> int:
        """Mark contact points with late events past their scoring watermark.

        A row is late when a watermark exists for its
        ``(contact_point_ref, lender_id)`` and either its ``received_at`` is
        after the watermark (arrived after scoring) or its ``occurred_at`` is
        older than the watermark (belongs to an already-scored period).
        """
        if df.empty:
            return 0
        keys = df[["contact_point_ref", "lender_id", "received_at", "occurred_at"]].copy()
        self.con.register("_dirty_staged", keys)
        try:
            self.con.execute(
                "CREATE OR REPLACE TEMP VIEW _late AS "
                "SELECT DISTINCT s.contact_point_ref AS contact_point_ref, "
                "s.lender_id AS lender_id "
                "FROM _dirty_staged s JOIN watermarks w "
                "ON w.contact_point_ref = s.contact_point_ref "
                "AND w.lender_id = s.lender_id "
                "WHERE s.received_at > w.watermark_ts OR s.occurred_at < w.watermark_ts"
            )
            new_marks = self.con.execute(
                "SELECT COUNT(*) FROM _late l WHERE NOT EXISTS "
                "(SELECT 1 FROM dirty_contact_points d "
                "WHERE d.contact_point_ref = l.contact_point_ref "
                "AND d.lender_id = l.lender_id)"
            ).fetchone()[0]
            self.con.execute(
                "INSERT OR IGNORE INTO dirty_contact_points "
                "SELECT contact_point_ref, lender_id, 'late_event', ? FROM _late",
                [marked_at],
            )
        finally:
            self.con.unregister("_dirty_staged")
        return int(new_marks)

    def set_watermark(self, contact_point_ref: str, lender_id: str, watermark_ts: datetime) -> None:
        """Upsert the scoring watermark for a contact point (feature pipeline)."""
        now = datetime.now(UTC)
        self.con.execute(
            "INSERT OR REPLACE INTO watermarks VALUES (?, ?, ?, ?)",
            [contact_point_ref, lender_id, watermark_ts, now],
        )

    # -- reads ----------------------------------------------------------

    def read_events(
        self,
        received_before: str | datetime | None = None,
        event_types: list[str] | None = None,
        lender_id: str | None = None,
        contact_point_refs: list[str] | None = None,
    ) -> pd.DataFrame:
        """Point-in-time-safe event reads for downstream consumers."""
        clauses: list[str] = []
        params: list[Any] = []
        if received_before is not None:
            clauses.append("received_at < ?")
            params.append(received_before)
        if event_types:
            placeholders = ", ".join(["?"] * len(event_types))
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(event_types)
        if lender_id is not None:
            clauses.append("lender_id = ?")
            params.append(lender_id)
        if contact_point_refs:
            placeholders = ", ".join(["?"] * len(contact_point_refs))
            clauses.append(f"contact_point_ref IN ({placeholders})")
            params.extend(contact_point_refs)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return self.con.execute(
            "SELECT event_id, event_type, lender_id, borrower_id, account_id, "
            "contact_point_ref, occurred_at, received_at, payload, ingested_at "
            f"FROM events{where} ORDER BY occurred_at",
            params,
        ).fetchdf()

    def counts(self) -> dict[str, int]:
        out = {}
        for table in ("events", "dead_letter", "dirty_contact_points", "watermarks"):
            out[table] = int(self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return out

    # -- trace history (VOI inputs only; never a predictor) ----------------

    def insert_trace_history(self, df: pd.DataFrame) -> tuple[int, int]:
        """Bulk insert trace outcomes; idempotent on ``trace_id``."""
        if df.empty:
            return 0, 0
        cols = [
            "trace_id",
            "account_id",
            "trace_date",
            "trigger_rule",
            "result",
            "new_contact_point_id",
            "cost_inr",
        ]
        self.con.register("_trace_staged", df[cols].copy())
        try:
            dupes = self.con.execute(
                "SELECT COUNT(*) FROM _trace_staged s WHERE EXISTS "
                "(SELECT 1 FROM trace_history t WHERE t.trace_id = s.trace_id)"
            ).fetchone()[0]
            self.con.execute(
                "INSERT INTO trace_history SELECT s.* FROM _trace_staged s "
                "WHERE NOT EXISTS "
                "(SELECT 1 FROM trace_history t WHERE t.trace_id = s.trace_id)"
            )
        finally:
            self.con.unregister("_trace_staged")
        return int(len(df) - dupes), int(dupes)

    def read_trace_history(self, account_ids: list[str] | None = None) -> pd.DataFrame:
        """Read trace outcomes for VOI; never use as a model predictor."""
        if account_ids:
            placeholders = ", ".join(["?"] * len(account_ids))
            return self.con.execute(
                "SELECT trace_id, account_id, trace_date, trigger_rule, result, "
                "new_contact_point_id, cost_inr FROM trace_history "
                f"WHERE account_id IN ({placeholders}) ORDER BY trace_date",
                account_ids,
            ).fetchdf()
        return self.con.execute(
            "SELECT trace_id, account_id, trace_date, trigger_rule, result, "
            "new_contact_point_id, cost_inr FROM trace_history ORDER BY trace_date"
        ).fetchdf()
