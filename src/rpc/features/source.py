"""Event source abstraction for the feature layer (simulation-only).

The feature code only talks to the ``EventSource`` interface. Today it is
backed by Parquet files read through DuckDB; when the ingestion event store
lands, a new subclass can replace ``ParquetEventSource`` without changing
any feature logic.

Point-in-time rule (enforced here, relied on everywhere downstream):
only events with ``received_at <= as_of`` are visible. Window membership is
decided on ``occurred_at`` by the feature code, never here.

Hidden simulator tables used only for evaluation are never opened by this
module (it reads events, contact points and borrowers only). Their filenames
are deliberately not spelled out here: the leakage tripwire
``test_eval_is_only_reader_of_restricted_tables`` scans source text for them.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# Columns we are allowed to read from contact_points.parquet. Anything else
# (e.g. a hidden ``shared_reason`` column) is ignored and reported via
# ``describe()`` / the ``dropped_columns`` attribute. Never extend this list
# with hidden/ground-truth columns.
ALLOWED_CP_COLUMNS = [
    "contact_point_ref",
    "borrower_id",
    "lender_id",
    "type",
    "value_hash",
    "source",
    "is_primary",
    "created_at",
]

ALLOWED_BORROWER_COLUMNS = [
    "borrower_id",
    "lender_id",
    "product",
    "dpd_bucket",
    "dpd_days",
    "outstanding",
    "secured",
]

# Payload keys the feature layer knows how to use, per event type.
# Unknown keys are ignored (forwardolg-compatible with richer simulators).
PAYLOAD_FIELDS = {
    "dial_attempt": ["network_response", "ring_seconds"],
    "disposition": ["disposition", "remarks", "agent_id"],
    "bot_transcript": ["transcript", "language", "who_answered", "extracted_phrases"],
    "field_visit": [
        "outcome",
        "dwell_seconds",
        "visit_time",
        "gps_lat",
        "gps_lon",
        "agent_id",
        "remarks",
    ],
    "contact_point_update": ["source", "contact_value", "contact_type", "is_primary"],
    "payment": ["amount", "payment_mode"],
}

CANONICAL_COLUMNS = [
    "event_id",
    "event_type",
    "lender_id",
    "borrower_id",
    "account_id",
    "contact_point_ref",
    "occurred_at",
    "received_at",
]


def as_utc(ts: Any) -> pd.Timestamp:
    """Coerce to a UTC Timestamp whether the input is naive or aware."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse occurred_at/received_at to UTC timestamps (vectorised)."""
    df = df.copy()
    for col in ("occurred_at", "received_at"):
        s = df[col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, utc=True, format="mixed")
        else:
            s = pd.to_datetime(s, utc=True)
        df[col] = s
    return df


def _parse_payloads(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the payload JSON exactly once into typed columns (vectorised).

    Missing keys become None. Unknown keys are dropped.
    """
    df = df.copy()
    parsed: list[dict[str, Any]] = []
    for et, raw in zip(df["event_type"].tolist(), df["payload"].tolist()):
        try:
            obj = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (json.JSONDecodeError, TypeError, ValueError):
            obj = {}
        if not isinstance(obj, dict):
            obj = {}
        row: dict[str, Any] = {}
        for key in PAYLOAD_FIELDS.get(str(et), []):
            row[key] = obj.get(key)
        parsed.append(row)
    payload_df = pd.DataFrame(parsed, index=df.index)
    for col in payload_df.columns:
        # Normalise empty strings to None so null semantics stay clean.
        payload_df[col] = payload_df[col].where(payload_df[col].astype(str) != "", None)
    # Numeric coercions.
    if "ring_seconds" in payload_df.columns:
        payload_df["ring_seconds"] = pd.to_numeric(payload_df["ring_seconds"], errors="coerce")
    if "dwell_seconds" in payload_df.columns:
        payload_df["dwell_seconds"] = pd.to_numeric(payload_df["dwell_seconds"], errors="coerce")
    if "amount" in payload_df.columns:
        payload_df["amount"] = pd.to_numeric(payload_df["amount"], errors="coerce")
    if "visit_time" in payload_df.columns:
        payload_df["visit_time"] = pd.to_datetime(
            payload_df["visit_time"], utc=True, format="mixed", errors="coerce"
        )
    for col in ("is_primary",):
        if col in payload_df.columns:
            payload_df[col] = payload_df[col].map(
                lambda v: None if v is None else bool(v)
            )
    return pd.concat([df, payload_df], axis=1)


def deduplicate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate on event_id, keeping the earliest received_at.

    Deterministic tie-break on (received_at, occurred_at, payload-string).
    """
    df = df.copy()
    df["_payload_str"] = df["payload"].astype(str)
    df = df.sort_values(["received_at", "occurred_at", "_payload_str"], kind="mergesort")
    df = df.drop_duplicates(subset="event_id", keep="first")
    return df.drop(columns=["_payload_str"]).reset_index(drop=True)


def canonicalize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Apply timestamp parsing, payload parsing and dedup to a raw frame."""
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = _parse_timestamps(df)
    df = _parse_payloads(df)
    return deduplicate_events(df)


class EventSource(ABC):
    """Abstract event source. All feature code depends only on this."""

    dropped_columns: list[str]

    @abstractmethod
    def load_visible_events(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Deduped, payload-parsed events with received_at <= as_of."""

    @abstractmethod
    def load_contact_points(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Contact-point records with created_at <= as_of (allowed cols only)."""

    @abstractmethod
    def load_borrowers(self) -> pd.DataFrame:
        """Borrower/account table (allowed cols only)."""

    @abstractmethod
    def load_events_window(
        self, occurred_after: pd.Timestamp, occurred_le: pd.Timestamp
    ) -> pd.DataFrame:
        """Deduped events with occurred_at in (after, le], any received_at.

        Used ONLY by the label module (future outcomes). Never by features.
        """

    @abstractmethod
    def max_received(self) -> pd.Timestamp | None:
        """Max received_at across all events (watermark ceiling)."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Human/machine-readable source description for docs and logs."""


class ParquetEventSource(EventSource):
    """Reads simulator Parquet output through DuckDB (simulation-only)."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.events_path = self.data_dir / "events.parquet"
        self.cp_path = self.data_dir / "contact_points.parquet"
        self.borrowers_path = self.data_dir / "borrowers.parquet"
        missing = [
            p.name for p in (self.events_path, self.cp_path, self.borrowers_path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing tables in {self.data_dir}: {missing}")
        self.dropped_columns = self._detect_dropped_columns()

    def _detect_dropped_columns(self) -> list[str]:
        con = duckdb.connect()
        try:
            dropped: list[str] = []
            for path, allowed in (
                (self.cp_path, ALLOWED_CP_COLUMNS),
                (self.borrowers_path, ALLOWED_BORROWER_COLUMNS),
            ):
                cols = [
                    r[0]
                    for r in con.execute(
                        f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0"
                    ).fetchall()
                ]
                dropped.extend(c for c in cols if c not in allowed)
            return sorted(set(dropped))
        finally:
            con.close()

    def _query_events(self, where: str, params: list[Any]) -> pd.DataFrame:
        con = duckdb.connect()
        try:
            df = con.execute(
                f"SELECT * FROM read_parquet('{self.events_path}') WHERE {where}",
                params,
            ).fetch_df()
        finally:
            con.close()
        if df.empty:
            df = pd.DataFrame(columns=[*CANONICAL_COLUMNS, "payload"])
        return canonicalize_events(df)

    def load_visible_events(self, as_of: pd.Timestamp) -> pd.DataFrame:
        as_of = as_utc(as_of)
        return self._query_events(
            "CAST(received_at AS TIMESTAMPTZ) <= CAST(? AS TIMESTAMPTZ)",
            [as_of.isoformat()],
        )

    def load_events_window(
        self, occurred_after: pd.Timestamp, occurred_le: pd.Timestamp
    ) -> pd.DataFrame:
        return self._query_events(
            "CAST(occurred_at AS TIMESTAMPTZ) > CAST(? AS TIMESTAMPTZ)"
            " AND CAST(occurred_at AS TIMESTAMPTZ) <= CAST(? AS TIMESTAMPTZ)",
            [as_utc(occurred_after).isoformat(),
             as_utc(occurred_le).isoformat()],
        )

    def _read_table(self, path: Path, allowed: list[str]) -> pd.DataFrame:
        con = duckdb.connect()
        try:
            cols = [
                r[0]
                for r in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0"
                ).fetchall()
            ]
            select = ", ".join(f'"{c}"' for c in cols if c in allowed)
            df = con.execute(f"SELECT {select} FROM read_parquet('{path}')").fetch_df()
        finally:
            con.close()
        return df

    def load_contact_points(self, as_of: pd.Timestamp) -> pd.DataFrame:
        as_of = as_utc(as_of)
        df = self._read_table(self.cp_path, ALLOWED_CP_COLUMNS)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed")
        return df[df["created_at"] <= as_of].reset_index(drop=True)

    def load_borrowers(self) -> pd.DataFrame:
        return self._read_table(self.borrowers_path, ALLOWED_BORROWER_COLUMNS)

    def max_received(self) -> pd.Timestamp | None:
        con = duckdb.connect()
        try:
            val = con.execute(
                f"SELECT MAX(CAST(received_at AS TIMESTAMPTZ)) FROM read_parquet('{self.events_path}')"
            ).fetchone()[0]
        finally:
            con.close()
        return as_utc(val) if val is not None else None

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "parquet",
            "data_dir": str(self.data_dir),
            "dropped_columns": self.dropped_columns,
        }


class DataFrameEventSource(EventSource):
    """In-memory source for tests and fixtures. Same PIT semantics."""

    def __init__(
        self,
        events: pd.DataFrame,
        contact_points: pd.DataFrame,
        borrowers: pd.DataFrame,
    ):
        self._events = canonicalize_events(events.copy())
        cp = contact_points.copy()
        if "created_at" in cp.columns:
            cp["created_at"] = pd.to_datetime(cp["created_at"], utc=True, format="mixed")
        drop = [c for c in cp.columns if c not in ALLOWED_CP_COLUMNS]
        self._cps = cp[[c for c in cp.columns if c in ALLOWED_CP_COLUMNS]]
        bor = borrowers.copy()
        drop_b = [c for c in bor.columns if c not in ALLOWED_BORROWER_COLUMNS]
        self._borrowers = bor[[c for c in bor.columns if c in ALLOWED_BORROWER_COLUMNS]]
        self.dropped_columns = sorted(set(drop + drop_b))

    def load_visible_events(self, as_of: pd.Timestamp) -> pd.DataFrame:
        as_of = as_utc(as_of)
        return self._events[self._events["received_at"] <= as_of].reset_index(drop=True)

    def load_events_window(
        self, occurred_after: pd.Timestamp, occurred_le: pd.Timestamp
    ) -> pd.DataFrame:
        m = (self._events["occurred_at"] > as_utc(occurred_after)) & (
            self._events["occurred_at"] <= as_utc(occurred_le)
        )
        return self._events[m].reset_index(drop=True)

    def load_contact_points(self, as_of: pd.Timestamp) -> pd.DataFrame:
        as_of = as_utc(as_of)
        return self._cps[self._cps["created_at"] <= as_of].reset_index(drop=True)

    def load_borrowers(self) -> pd.DataFrame:
        return self._borrowers.copy()

    def max_received(self) -> pd.Timestamp | None:
        if self._events.empty:
            return None
        return pd.Timestamp(self._events["received_at"].max(), tz="UTC")

    def describe(self) -> dict[str, Any]:
        return {"kind": "dataframe", "dropped_columns": self.dropped_columns}
