"""Ingestion adapters, field mappings, dedupe, event store."""

from src.rpc.ingest.adapter import IngestAdapter, ingest, read_events, replay
from src.rpc.ingest.normalize import hash_normalized, normalize_address, normalize_phone
from src.rpc.ingest.store import EventStore, IngestConfig, default_db_path

__all__ = [
    "EventStore",
    "IngestAdapter",
    "IngestConfig",
    "default_db_path",
    "hash_normalized",
    "ingest",
    "normalize_address",
    "normalize_phone",
    "read_events",
    "replay",
]
