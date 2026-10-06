"""Ingestion adapters, field mappings, dedupe, event store."""

from src.rpc.ingest.adapter import IngestAdapter, ingest, read_events, replay
from src.rpc.ingest.enrich import (
    DuplicateAccountError,
    LenderJoinError,
    LenderLookupError,
    attach_lender_id,
    load_lender_lookup,
)
from src.rpc.ingest.normalize import hash_normalized, normalize_address, normalize_phone
from src.rpc.ingest.store import EventStore, IngestConfig, default_db_path
from src.rpc.ingest.traces import TraceHistoryError, load_trace_history, read_trace_history

__all__ = [
    "DuplicateAccountError",
    "EventStore",
    "IngestAdapter",
    "IngestConfig",
    "LenderJoinError",
    "LenderLookupError",
    "TraceHistoryError",
    "attach_lender_id",
    "default_db_path",
    "hash_normalized",
    "ingest",
    "load_lender_lookup",
    "load_trace_history",
    "normalize_address",
    "normalize_phone",
    "read_events",
    "read_trace_history",
    "replay",
]
