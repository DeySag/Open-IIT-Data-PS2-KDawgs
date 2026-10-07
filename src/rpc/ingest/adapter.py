"""Mapping-driven ingestion adapter + event-store writer.

Mapping-driven ingestion adapter + event-store writer. Public API::

    ingest(batch_or_path, source) -> {"accepted", "duplicate", "rejected", "dirty_marked"}
    read_events(received_before=None, event_types=None, lender_id=None,
                contact_point_refs=None) -> DataFrame
    replay(path, source) -> {...}

Notes for neighbouring workstreams (coordinator): ``src/rpc/features/source.py``
does not exist, so there is no ``EventSource`` protocol to implement. Downstream
consumers should read through :func:`read_events` here; the signature matches
what the feature store needs for point-in-time reads.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.rpc.contracts import InputEvent
from src.rpc.ingest.mapping import (
    HIDDEN_GROUND_TRUTH_COLUMNS,
    _parse_payload_or_none,
    apply_mapping,
    load_mapping,
)
from src.rpc.ingest.normalize import REDACTED, redact_record
from src.rpc.ingest.store import EventStore, IngestConfig, default_db_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------


class UnsupportedInputError(ValueError):
    """Ingest input type not supported (carries path and suffix)."""


def load_input(batch_or_path: pd.DataFrame | list[dict] | str | Path) -> pd.DataFrame:
    """Load a batch from a DataFrame, list of dicts, or file path (vectorised).

    Supported extensions: .csv, .parquet, .ndjson/.jsonl/.ndj, .json.
    """
    if isinstance(batch_or_path, pd.DataFrame):
        return batch_or_path.copy()
    if isinstance(batch_or_path, list):
        return pd.DataFrame(batch_or_path)
    return _load_file(Path(batch_or_path))


def _load_file(path: Path) -> pd.DataFrame:
    """Read one source file into a DataFrame based on its extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype="string", keep_default_na=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".ndjson", ".jsonl", ".ndj"):
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise UnsupportedInputError(path, suffix)


def _json_default(value: Any) -> str:
    return str(value)


def _row_hash(source: str, raw_json: str) -> str:
    return hashlib.sha256(f"{source}|{raw_json}".encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# pydantic validation helpers
# ---------------------------------------------------------------------------


def _canonical_row_to_kwargs(row: pd.Series) -> dict[str, Any] | None:
    """Build InputEvent kwargs from a canonical row; None when unbuildable."""
    try:
        occurred = row["occurred_at"]
        received = row["received_at"]
        occurred_dt = occurred.to_pydatetime() if hasattr(occurred, "to_pydatetime") else occurred
        received_dt = received.to_pydatetime() if hasattr(received, "to_pydatetime") else received
        payload = _parse_payload_or_none(row["payload"])
        if payload is None:
            return None
        return {
            "event_id": str(row["event_id"]),
            "event_type": str(row["event_type"]),
            "lender_id": str(row["lender_id"]),
            "borrower_id": str(row["borrower_id"]),
            "account_id": str(row["account_id"]),
            "contact_point_ref": str(row["contact_point_ref"]),
            "occurred_at": occurred_dt,
            "received_at": received_dt,
            "payload": payload,
        }
    except Exception:
        return None


def _safe_error_kinds(exc: ValidationError, limit: int = 5) -> list[str]:
    """Summarise a ValidationError without echoing input values (no PII in logs)."""
    kinds: list[str] = []
    for err in exc.errors(include_url=False)[:limit]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        kinds.append(f"{loc}:{err.get('type', '?')}")
    return kinds


def _pydantic_check(frame: pd.DataFrame) -> tuple[list[Any], dict[str, int]]:
    """Validate canonical rows with the full InputEvent schema.

    Returns (index labels that failed, {error-kind summary: count}).
    Linear single pass; callers only pass rejected rows + a small sample.
    """
    failed: list[Any] = []
    kinds: dict[str, int] = {}
    for idx, row in frame.iterrows():
        kwargs = _canonical_row_to_kwargs(row)
        if kwargs is None:
            failed.append(idx)
            kinds["unbuildable"] = kinds.get("unbuildable", 0) + 1
            continue
        try:
            InputEvent(**kwargs)
        except ValidationError as exc:
            failed.append(idx)
            for kind in _safe_error_kinds(exc):
                kinds[kind] = kinds.get(kind, 0) + 1
        except Exception:
            failed.append(idx)
            kinds["unexpected"] = kinds.get("unexpected", 0) + 1
    return failed, kinds


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------


class IngestAdapter:
    """Mapping-driven adapter: source records -> canonical events -> store."""

    def __init__(self, config: IngestConfig | None = None) -> None:
        self.config = config or IngestConfig()

    # -- main API ------------------------------------------------------

    def ingest(
        self, batch_or_path: pd.DataFrame | list[dict] | str | Path, source: str
    ) -> dict[str, int]:
        """Ingest a batch; returns {accepted, duplicate, rejected, dirty_marked}."""
        started = datetime.now(UTC)
        now = datetime.now(UTC)
        raw = load_input(batch_or_path)
        n_in = len(raw)

        dropped = [c for c in raw.columns if c in HIDDEN_GROUND_TRUTH_COLUMNS]
        if dropped:
            raw = raw.drop(columns=dropped)

        mapping = load_mapping(source)
        store = EventStore(self.config.db_path)
        try:
            if mapping is None:
                return self._reject_all_for_missing_mapping(store, raw, source, n_in, now)

            canonical = apply_mapping(raw, mapping, source)
            rejected_mask = canonical["_reason"].notna()
            rejected = canonical[rejected_mask]
            accepted = canonical[~rejected_mask]

            # Within-batch dedupe: keep earliest received_at.
            within_dupes = 0
            if not accepted.empty:
                accepted = accepted.sort_values("received_at", kind="stable")
                before = len(accepted)
                accepted = accepted.drop_duplicates(subset="event_id", keep="first")
                within_dupes = before - len(accepted)

            self._recheck_rejected(rejected, source)

            # Full pydantic validation on a 1% random sample of accepted rows.
            # Sample failures are quarantined to dead_letter.
            accepted, sample_failed = self._quarantine_sample(accepted, mapping, source)

            # Dead-letter rows: redacted raw + reason; idempotent via row_hash.
            store.insert_dead_letter(
                self._assemble_dead(raw, mapping, rejected, sample_failed, source, now)
            )

            inserted, store_dupes = store.insert_canonical(accepted, now)
            dirty_marked = store.mark_dirty(accepted, now)

            result = {
                "accepted": int(inserted),
                "duplicate": int(store_dupes + within_dupes),
                "rejected": int(len(rejected) + len(sample_failed)),
                "dirty_marked": int(dirty_marked),
            }
            elapsed = (datetime.now(UTC) - started).total_seconds()
            logger.info(
                "ingest source=%s rows=%d hidden_dropped=%d accepted=%d duplicate=%d "
                "rejected=%d dirty_marked=%d elapsed_s=%.1f",
                source,
                n_in,
                len(dropped),
                result["accepted"],
                result["duplicate"],
                result["rejected"],
                result["dirty_marked"],
                elapsed,
            )
            return result
        finally:
            store.close()

    def _reject_all_for_missing_mapping(
        self,
        store: EventStore,
        raw: pd.DataFrame,
        source: str,
        n_in: int,
        now: datetime,
    ) -> dict[str, int]:
        """Reject a whole batch when no mapping file exists (never raises)."""
        logger.warning(
            "ingest source=%s rows=%d: no mapping file; all rows rejected",
            source,
            n_in,
        )
        dead = self._dead_rows(raw, None, ["missing_mapping"] * n_in, source, now)
        store.insert_dead_letter(dead)
        return {"accepted": 0, "duplicate": 0, "rejected": n_in, "dirty_marked": 0}

    @staticmethod
    def _recheck_rejected(rejected: pd.DataFrame, source: str) -> None:
        """Run full pydantic validation on ALL rejected rows (confirms rejection).

        Summaries only (locations/kinds, no values) are logged; nothing stored.
        """
        if rejected.empty:
            return
        failed, kinds = _pydantic_check(rejected)
        logger.info(
            "ingest source=%s: pydantic re-checked %d rejected rows "
            "(%d failed as expected, kinds=%s)",
            source,
            len(rejected),
            len(failed),
            kinds,
        )

    def _quarantine_sample(
        self, accepted: pd.DataFrame, mapping: dict[str, Any], source: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pydantic-validate a random sample; split off failures for quarantine."""
        empty = accepted.iloc[0:0]
        if accepted.empty:
            return accepted, empty
        rate = float(mapping.get("validation", {}).get("sample_rate", self.config.sample_rate))
        seed = int(mapping.get("validation", {}).get("sample_seed", self.config.sample_seed))
        if rate <= 0:
            return accepted, empty
        sample = (
            accepted.sample(frac=min(rate, 1.0), random_state=seed)
            if len(accepted) > 1
            else accepted
        )
        failed_idx, kinds = _pydantic_check(sample)
        if kinds:
            logger.info(
                "ingest source=%s: pydantic sample %d rows, failures=%d kinds=%s",
                source,
                len(sample),
                len(failed_idx),
                kinds,
            )
        if not failed_idx:
            return accepted, empty
        sample_failed = accepted.loc[failed_idx]
        return accepted.drop(index=sample_failed.index), sample_failed

    def _assemble_dead(
        self,
        raw: pd.DataFrame,
        mapping: dict[str, Any],
        rejected: pd.DataFrame,
        sample_failed: pd.DataFrame,
        source: str,
        now: datetime,
    ) -> pd.DataFrame:
        """Build the dead-letter frame for rejected + quarantined rows."""
        columns = ["row_hash", "raw_json", "reason", "source", "ingested_at"]
        dead_parts: list[pd.DataFrame] = []
        if not rejected.empty:
            dead_parts.append(
                self._dead_rows(
                    raw.loc[rejected.index],
                    mapping,
                    rejected["_reason"].tolist(),
                    source,
                    now,
                )
            )
        if not sample_failed.empty:
            dead_parts.append(
                self._dead_rows(
                    raw.loc[sample_failed.index],
                    mapping,
                    ["pydantic_sample_failed"] * len(sample_failed),
                    source,
                    now,
                )
            )
        if not dead_parts:
            return pd.DataFrame(columns=columns)
        return pd.concat(dead_parts, ignore_index=True)

    def _dead_rows(
        self,
        raw: pd.DataFrame,
        mapping: dict[str, Any] | None,
        reasons: list[str],
        source: str,
        now: datetime,
    ) -> pd.DataFrame:
        """Build dead-letter rows with PII redacted (vectorised redact fields).

        Beyond the mapped contact field, free-text and masked-identifier
        columns (remarks, masked numbers, raw address text) are redacted too:
        they can carry names or quasi-identifiers and must never land in the
        store or logs (audit §11).
        """
        if mapping is None:
            redact_fields: list[str] = []  # unknown source: redact everything below
        else:
            raw_field = str(mapping.get("contact_point", {}).get("raw_field", ""))
            redact_fields = [raw_field] if raw_field else []
            redact_fields += [
                c
                for c in ("remark", "remarks", "phone_masked", "address_text")
                if c in raw.columns and c not in redact_fields
            ]
        records = raw.to_dict(orient="records")
        hashes: list[str] = []
        raws: list[str] = []
        for record in records:
            if mapping is None:
                redacted = {k: REDACTED for k in record}
            else:
                redacted = redact_record(record, redact_fields)
            raw_json = json.dumps(redacted, sort_keys=True, default=_json_default)
            raws.append(raw_json)
            hashes.append(_row_hash(source, raw_json))
        return pd.DataFrame(
            {
                "row_hash": hashes,
                "raw_json": raws,
                "reason": reasons,
                "source": [source] * len(records),
                "ingested_at": [now] * len(records),
            }
        )

    def read_events(
        self,
        received_before: str | datetime | None = None,
        event_types: list[str] | None = None,
        lender_id: str | None = None,
        contact_point_refs: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read canonical events with optional point-in-time-safe filters."""
        store = EventStore(self.config.db_path)
        try:
            return store.read_events(
                received_before=received_before,
                event_types=event_types,
                lender_id=lender_id,
                contact_point_refs=contact_point_refs,
            )
        finally:
            store.close()

    def replay(self, path: str | Path, source: str) -> dict[str, int]:
        """Re-ingest a file; idempotent by event_id/row_hash."""
        return self.ingest(path, source)

    def set_watermark(self, contact_point_ref: str, lender_id: str, watermark_ts: datetime) -> None:
        """Record a scoring watermark (used by the feature pipeline)."""
        store = EventStore(self.config.db_path)
        try:
            store.set_watermark(contact_point_ref, lender_id, watermark_ts)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# module-level convenience API (default store path)
# ---------------------------------------------------------------------------


def ingest(
    batch_or_path: pd.DataFrame | list[dict] | str | Path,
    source: str,
    db_path: str | Path | None = None,
    config: IngestConfig | None = None,
) -> dict[str, int]:
    """Ingest a batch into the event store."""
    cfg = config or IngestConfig(db_path=db_path or default_db_path())
    return IngestAdapter(cfg).ingest(batch_or_path, source)


def read_events(
    received_before: str | datetime | None = None,
    event_types: list[str] | None = None,
    lender_id: str | None = None,
    contact_point_refs: list[str] | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read canonical events with optional filters (feature-store entry point)."""
    cfg = IngestConfig(db_path=db_path or default_db_path())
    return IngestAdapter(cfg).read_events(
        received_before=received_before,
        event_types=event_types,
        lender_id=lender_id,
        contact_point_refs=contact_point_refs,
    )


def replay(
    path: str | Path,
    source: str,
    db_path: str | Path | None = None,
) -> dict[str, int]:
    """Re-ingest a file; idempotent by event_id/row_hash."""
    cfg = IngestConfig(db_path=db_path or default_db_path())
    return IngestAdapter(cfg).replay(path, source)
