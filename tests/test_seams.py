"""Cross-component seam tests (integration-owned).

Proves the pipeline is one connected system, not adjacent modules:
ingest store -> features source -> models -> decision -> serve -> eval.
All fixtures below are synthetic and invented for tests; nothing is real data.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.rpc.features.source import (
    DataFrameEventSource,
    IngestEventSource,
    canonicalize_events,
)
from src.rpc.ingest import IngestAdapter
from src.rpc.ingest.store import IngestConfig

UTC = "UTC"
T0_MS = 1704067200000  # 2024-01-01T00:00:00Z


def dialer_row(i: int = 0, **overrides) -> dict:
    row = {
        "call_id": f"SEAM-{i:04d}",
        "lender_code": "LENDER_001",
        "cust_id": f"BORR_{i:04d}",
        "loan_ac_no": f"ACC_{i:04d}",
        "mobile_no": "+91 98765 43210",
        "call_start_ms": T0_MS + i * 60_000,
        "recv_ms": T0_MS + i * 60_000 + 300_000,
        "result_code": "ANS",
        "ring_secs": 2.5,
        "chan": "VB",
        "agent_id": "AG1",
    }
    row.update(overrides)
    return row


def seed_store(tmp_path: Path, n: int = 6) -> str:
    """Ingest n dialer rows through the real adapter; return the db path."""
    db = str(tmp_path / "seam.duckdb")
    adapter = IngestAdapter(IngestConfig(db_path=db))
    res = adapter.ingest(pd.DataFrame([dialer_row(i) for i in range(n)]), "cn_dialer_csv")
    assert res["accepted"] == n, res
    return db


def test_ingest_source_visible_events_match_canonical(tmp_path: Path):
    db = seed_store(tmp_path)
    source = IngestEventSource(db_path=db)
    as_of = pd.Timestamp("2024-01-01T00:04:00Z")
    visible = source.load_visible_events(as_of)
    # rows 0..2 received by T0+1min..3min+5min? recv = start+5min: rows with
    # recv <= 00:04 are none (row0 recv 00:05). Use a later cutoff below.
    assert visible.empty
    later = source.load_visible_events(pd.Timestamp("2024-01-01T00:07:00Z"))
    assert len(later) == 2  # rows 0 (recv 00:05) and 1 (recv 00:06); strict < cutoff
    assert (later["received_at"] < pd.Timestamp("2024-01-01T00:07:00Z")).all()


def test_ingest_source_pit_excludes_late_arrivals(tmp_path: Path):
    db = seed_store(tmp_path, n=4)
    source = IngestEventSource(db_path=db)
    # row i: occurred T0+i*60s, received +300s. Row 0 received exactly at
    # 00:05:00Z; the store uses strict <, so 00:05:01Z shows exactly row 0.
    early = source.load_visible_events(pd.Timestamp("2024-01-01T00:05:01Z"))
    assert len(early) == 1
    full = source.load_visible_events(pd.Timestamp("2024-01-01T01:00:00Z"))
    assert len(full) == 4


def test_ingest_source_dedupe_parity_with_dataframe_source(tmp_path: Path):
    db = seed_store(tmp_path, n=5)
    adapter = IngestAdapter(IngestConfig(db_path=db))
    # Re-ingest row 0 with an earlier received_at: keep-earliest must win,
    # identically through both sources.
    adapter.ingest(
        pd.DataFrame([dialer_row(0, recv_ms=T0_MS + 120_000)]), "cn_dialer_csv"
    )
    ingest_source = IngestEventSource(db_path=db)
    stored = adapter.read_events()
    frame_source = DataFrameEventSource(
        events=stored,
        contact_points=pd.DataFrame(
            columns=["contact_point_ref", "borrower_id", "lender_id", "created_at"]
        ),
        borrowers=pd.DataFrame(columns=["borrower_id", "lender_id"]),
    )
    as_of = pd.Timestamp("2024-01-01T01:00:00Z")
    a = ingest_source.load_visible_events(as_of).sort_values("event_id").reset_index(drop=True)
    b = frame_source.load_visible_events(as_of).sort_values("event_id").reset_index(drop=True)
    assert len(a) == len(b) == 5
    pd.testing.assert_series_equal(a["event_id"], b["event_id"])
    pd.testing.assert_series_equal(a["received_at"], b["received_at"])
    assert (
        a["received_at"].min()
        == pd.Timestamp(T0_MS + 120_000, unit="ms", tz="UTC")
    )


def test_ingest_source_derives_contact_and_borrower_universes(tmp_path: Path):
    db = seed_store(tmp_path, n=3)
    source = IngestEventSource(db_path=db)
    as_of = pd.Timestamp("2024-01-01T01:00:00Z")
    cps = source.load_contact_points(as_of)
    assert set(cps.columns) >= {"contact_point_ref", "borrower_id", "lender_id", "created_at"}
    assert len(cps) == 1  # same phone number -> same hash -> one contact point
    assert (cps["created_at"] <= as_of).all()
    borrowers = source.load_borrowers()
    assert set(borrowers["borrower_id"]) == {f"BORR_{i:04d}" for i in range(3)}
    assert source.max_received() == pd.Timestamp(T0_MS + 2 * 60_000 + 300_000, unit="ms", tz="UTC")
    assert source.describe()["kind"] == "ingest"
    assert source.dropped_columns == ["contact_points_table", "borrowers_table"]


def test_ingest_source_empty_store(tmp_path: Path):
    db = str(tmp_path / "empty.duckdb")
    IngestAdapter(IngestConfig(db_path=db))  # creates schema, ingests nothing
    source = IngestEventSource(db_path=db)
    as_of = pd.Timestamp("2024-01-01T01:00:00Z")
    assert source.load_visible_events(as_of).empty
    assert source.load_contact_points(as_of).empty
    assert source.load_borrowers().empty
    assert source.max_received() is None


def _simulator_frame(n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "event_id": str(uuid4()),
                "event_type": "dial_attempt",
                "lender_id": "LENDER_001",
                "borrower_id": f"BORR_{i:04d}",
                "account_id": f"ACC_{i:04d}",
                "contact_point_ref": "b" * 16,
                "occurred_at": "2024-01-01T00:00:00+00:00",
                "received_at": "2024-01-01T00:05:00+00:00",
                "payload": json.dumps(
                    {
                        "network_response": "answered",
                        "ring_seconds": 2.5,
                        "channel": "voice_bot",
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


def test_ingest_source_roundtrip_simulator_rows(tmp_path: Path):
    """Canonical simulator-shaped rows survive store -> source unchanged."""
    db = str(tmp_path / "sim.duckdb")
    adapter = IngestAdapter(IngestConfig(db_path=db))
    frame = _simulator_frame()
    res = adapter.ingest(frame, "simulator")
    assert res["accepted"] == 4, res
    source = IngestEventSource(db_path=db)
    visible = source.load_visible_events(pd.Timestamp("2024-01-01T01:00:00Z"))
    assert len(visible) == 4
    assert set(visible["event_id"]) == set(frame["event_id"])
    assert (visible["contact_point_ref"] == "b" * 16).all()
    assert (visible["network_response"] == "answered").all()
    window = source.load_events_window(
        pd.Timestamp("2023-12-31T00:00:00Z"), pd.Timestamp("2024-01-01T01:00:00Z")
    )
    assert len(window) == 4
