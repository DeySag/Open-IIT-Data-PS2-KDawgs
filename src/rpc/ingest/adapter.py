"""Ingestion adapter - transforms CN format to canonical envelope."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd

from src.rpc.contracts import (
    InputEvent,
    EventType,
    NetworkResponse,
    Disposition,
    DialAttemptPayload,
    DispositionPayload,
    BotTranscriptPayload,
    FieldVisitPayload,
    ContactPointUpdatePayload,
    PaymentPayload,
)


class FieldMapping:
    """Maps source-specific field names to canonical envelope."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for canonical, source in self.mapping.items():
            if source in record:
                result[canonical] = record[source]
        return result


def load_field_mapping(lender_id: str) -> FieldMapping:
    """Load field mapping for a lender from configs/field_mappings/."""
    mapping_path = Path(f"configs/field_mappings/{lender_id}.yaml")
    if mapping_path.exists():
        import yaml
        with open(mapping_path) as f:
            mapping = yaml.safe_load(f)
        return FieldMapping(mapping)
    # Default identity mapping
    return FieldMapping({})


def parse_event(record: dict[str, Any], lender_id: str) -> InputEvent:
    """Parse a raw CN record into canonical InputEvent."""
    mapping = load_field_mapping(lender_id)
    mapped = mapping.apply(record)

    event_type = EventType(mapped["event_type"])

    payload_map = {
        EventType.DIAL_ATTEMPT: DialAttemptPayload,
        EventType.DISPOSITION: DispositionPayload,
        EventType.BOT_TRANSCRIPT: BotTranscriptPayload,
        EventType.FIELD_VISIT: FieldVisitPayload,
        EventType.CONTACT_POINT_UPDATE: ContactPointUpdatePayload,
        EventType.PAYMENT: PaymentPayload,
    }

    payload_cls = payload_map[event_type]
    payload_data = json.loads(mapped.get("payload", "{}")) if isinstance(mapped.get("payload"), str) else mapped.get("payload", {})
    payload = payload_cls(**payload_data)

    return InputEvent(
        event_id=UUID(mapped["event_id"]),
        event_type=event_type,
        lender_id=mapped["lender_id"],
        borrower_id=mapped["borrower_id"],
        account_id=mapped["account_id"],
        contact_point_ref=mapped["contact_point_ref"],
        occurred_at=datetime.fromisoformat(mapped["occurred_at"].replace("Z", "+00:00")),
        received_at=datetime.fromisoformat(mapped["received_at"].replace("Z", "+00:00")),
        payload=payload,
    )


def ingest_parquet(path: str, lender_id: str) -> list[InputEvent]:
    """Ingest events from parquet file."""
    df = pd.read_parquet(path)
    events = []
    for _, row in df.iterrows():
        try:
            events.append(parse_event(row.to_dict(), lender_id))
        except Exception as e:
            # Log and skip bad records
            print(f"Failed to parse event: {e}")
    return events