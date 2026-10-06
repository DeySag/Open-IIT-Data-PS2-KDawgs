"""Tests for the ingestion adapter + event store (adapter conformance fixtures).

All source data below are inline test fixtures; nothing here is
real CN data.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd

from src.rpc.ingest import IngestAdapter, ingest, read_events, replay
from src.rpc.ingest.normalize import hash_phone_series
from src.rpc.ingest.store import EventStore, IngestConfig

T0_MS = 1704067200000  # 2024-01-01T00:00:00Z
TABLES = ("events", "dead_letter", "dirty_contact_points")


def make_adapter(tmp_path: Path) -> IngestAdapter:
    return IngestAdapter(IngestConfig(db_path=store_path(tmp_path)))


def store_path(tmp_path: Path) -> str:
    """DuckDB file for a test (one store per test via tmp_path)."""
    return str(tmp_path / "store.duckdb")


def dialer_row(i: int = 0, **overrides) -> dict:
    row = {
        "call_id": f"CALL-{i:04d}",
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


def ndjson_row(i: int = 0, **overrides) -> dict:
    row = {
        "meta": {
            "id": f"NDSP-{i:04d}",
            "lender": "LENDER_002",
            "recv_ts": "2024-03-05T14:25:10+05:30",
        },
        "customer": {"borrower": f"BORR_N{i:04d}", "account": f"ACC_N{i:04d}"},
        "contact": {"mobile": "09876543210"},
        "call": {"start": "05-03-2024 14:20:00"},
        "outcome": {
            "code": "PTP",
            "remarks": "conformance fixture remark",
            "agent": "AG7",
            "channel": "telecaller",
        },
    }
    row.update(overrides)
    return row


def canonical_row(i: int = 0, **overrides) -> dict:
    """Already-canonical envelope row (as framed from validated InputEvents)."""
    row = {
        "event_id": str(uuid4()),
        "event_type": "dial_attempt",
        "lender_id": "LENDER_001",
        "borrower_id": f"BORR_S{i:04d}",
        "account_id": f"ACC_S{i:04d}",
        "contact_point_ref": "a" * 16,
        "occurred_at": "2024-01-01T00:00:00+00:00",
        "received_at": "2024-01-01T00:05:00+00:00",
        "payload": json.dumps(
            {"network_response": "answered", "ring_seconds": 2.5, "channel": "voice_bot"}
        ),
    }
    row.update(overrides)
    return row


def fetch_table(db_path: str, table: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute(f"SELECT * FROM {table} ORDER BY 1").fetchdf()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# source formats: renamed fields, timestamp formats, enum maps
# ---------------------------------------------------------------------------


def test_dialer_csv_mapping(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    res = adapter.ingest(pd.DataFrame([dialer_row()]), "cn_dialer_csv")
    assert res == {"accepted": 1, "duplicate": 0, "rejected": 0, "dirty_marked": 0}
    events = adapter.read_events()
    assert len(events) == 1
    row = events.iloc[0]
    assert row["event_type"] == "dial_attempt"
    assert row["lender_id"] == "LENDER_001"
    payload = json.loads(row["payload"])
    assert payload["network_response"] == "answered"  # ANS mapped
    assert payload["channel"] == "voice_bot"  # VB mapped
    assert payload["ring_seconds"] == 2.5
    # epoch-ms IST instant converted to UTC ISO-8601
    assert row["occurred_at"] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert row["received_at"] == pd.Timestamp("2024-01-01T00:05:00Z")


def test_disposition_ndjson_nested_mapping(tmp_path: Path):
    path = tmp_path / "dispo.ndjson"
    path.write_text("\n".join(json.dumps(ndjson_row(i)) for i in range(3)))
    adapter = make_adapter(tmp_path)
    res = adapter.ingest(path, "cn_disposition_ndjson")
    assert res["accepted"] == 3 and res["rejected"] == 0
    events = adapter.read_events()
    assert set(events["event_type"]) == {"disposition"}
    payload = json.loads(events.iloc[0]["payload"])
    assert payload["disposition"] == "promise_to_pay"  # PTP mapped
    assert payload["channel"] == "telecaller"
    # custom "%d-%m-%Y %H:%M:%S" IST wall clock -> UTC
    assert events.iloc[0]["occurred_at"] == pd.Timestamp("2024-03-05T08:50:00Z")
    assert events.iloc[0]["received_at"] == pd.Timestamp("2024-03-05T08:55:10Z")


def test_api_identity_mapping(tmp_path: Path):
    path = tmp_path / "events.parquet"
    pd.DataFrame([canonical_row(i) for i in range(5)]).to_parquet(path, index=False)
    adapter = make_adapter(tmp_path)
    res = adapter.ingest(path, "api")
    assert res == {"accepted": 5, "duplicate": 0, "rejected": 0, "dirty_marked": 0}
    events = adapter.read_events()
    assert len(events) == 5
    assert (events["contact_point_ref"] == "a" * 16).all()


# ---------------------------------------------------------------------------
# idempotent replay + dedupe
# ---------------------------------------------------------------------------


def test_idempotent_replay(tmp_path: Path):
    path = tmp_path / "dialer.csv"
    pd.DataFrame([dialer_row(i) for i in range(10)]).to_csv(path, index=False)
    adapter = make_adapter(tmp_path)
    first = adapter.ingest(path, "cn_dialer_csv")
    assert first == {"accepted": 10, "duplicate": 0, "rejected": 0, "dirty_marked": 0}
    before = {t: fetch_table(store_path(tmp_path), t) for t in TABLES}
    second = adapter.replay(path, "cn_dialer_csv")
    assert second["accepted"] == 0
    assert second["duplicate"] == 10
    assert second["rejected"] == 0
    after = {t: fetch_table(store_path(tmp_path), t) for t in TABLES}
    for table, frame in before.items():
        pd.testing.assert_frame_equal(frame, after[table])


def test_dedupe_keeps_earliest_received_at(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    rows = [
        dialer_row(0, recv_ms=T0_MS + 600_000),  # later received_at
        dialer_row(0, recv_ms=T0_MS + 300_000),  # earlier received_at, same call_id
    ]
    res = adapter.ingest(pd.DataFrame(rows), "cn_dialer_csv")
    assert res["accepted"] == 1 and res["duplicate"] == 1
    events = adapter.read_events()
    assert len(events) == 1
    assert events.iloc[0]["received_at"] == pd.Timestamp(T0_MS + 300_000, unit="ms", tz="UTC")


def test_duplicate_counts_on_replay_with_rejects(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    rows = [dialer_row(0), dialer_row(1, result_code="ZZZ_UNKNOWN")]
    first = adapter.ingest(pd.DataFrame(rows), "cn_dialer_csv")
    assert first["accepted"] == 1 and first["rejected"] == 1
    second = adapter.ingest(pd.DataFrame(rows), "cn_dialer_csv")
    assert second["accepted"] == 0
    assert second["duplicate"] == 1  # only the accepted row can duplicate
    assert second["rejected"] == 1
    assert len(adapter.read_events()) == 1
    assert len(fetch_table(store_path(tmp_path), "dead_letter")) == 1


# ---------------------------------------------------------------------------
# late / out-of-order events
# ---------------------------------------------------------------------------


def test_late_event_marks_dirty(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    adapter.ingest(pd.DataFrame([dialer_row(0)]), "cn_dialer_csv")
    ref = adapter.read_events().iloc[0]["contact_point_ref"]
    # Scoring watermark at T0+10min; late event occurred before it, received after.
    adapter.set_watermark(ref, "LENDER_001", datetime(2024, 1, 1, 0, 10, tzinfo=UTC))
    late = dialer_row(1, call_start_ms=T0_MS + 120_000, recv_ms=T0_MS + 780_000)
    res = adapter.ingest(pd.DataFrame([late]), "cn_dialer_csv")
    assert res["dirty_marked"] == 1
    dirty = fetch_table(store_path(tmp_path), "dirty_contact_points")
    assert len(dirty) == 1
    assert dirty.iloc[0]["contact_point_ref"] == ref
    assert dirty.iloc[0]["reason"] == "late_event"


def test_out_of_order_does_not_change_final_state(tmp_path: Path):
    batch_a = [dialer_row(0, recv_ms=T0_MS + 600_000), dialer_row(0, recv_ms=T0_MS + 300_000)]
    batch_b = list(reversed(batch_a))
    adapters = [make_adapter(tmp_path / f"db{i}") for i in range(2)]
    (tmp_path / "db0").mkdir(exist_ok=True)
    (tmp_path / "db1").mkdir(exist_ok=True)
    adapters[0] = IngestAdapter(IngestConfig(db_path=str(tmp_path / "db0" / "s.duckdb")))
    adapters[1] = IngestAdapter(IngestConfig(db_path=str(tmp_path / "db1" / "s.duckdb")))
    r0 = adapters[0].ingest(pd.DataFrame(batch_a), "cn_dialer_csv")
    r1 = adapters[1].ingest(pd.DataFrame(batch_b), "cn_dialer_csv")
    assert r0 == r1 == {"accepted": 1, "duplicate": 1, "rejected": 0, "dirty_marked": 0}
    cols = [
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
    e0 = adapters[0].read_events()[cols].reset_index(drop=True)
    e1 = adapters[1].read_events()[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(e0, e1)


# ---------------------------------------------------------------------------
# malformed rows -> dead_letter, never exceptions
# ---------------------------------------------------------------------------


def _dead_reasons(tmp_path: Path) -> list[str]:
    return fetch_table(store_path(tmp_path), "dead_letter")["reason"].tolist()


def test_missing_mapping_rejects_without_exception(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    res = adapter.ingest(pd.DataFrame([dialer_row(0)]), "no_such_source_xyz")
    assert res == {"accepted": 0, "duplicate": 0, "rejected": 1, "dirty_marked": 0}
    assert _dead_reasons(tmp_path) == ["missing_mapping"]


def test_missing_required_field(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    res = adapter.ingest(pd.DataFrame([dialer_row(0, cust_id=None)]), "cn_dialer_csv")
    assert res["rejected"] == 1 and res["accepted"] == 0
    assert _dead_reasons(tmp_path) == ["missing_required_field"]


def test_unknown_enum(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    res = adapter.ingest(pd.DataFrame([dialer_row(0, result_code="BOGUS")]), "cn_dialer_csv")
    assert res["rejected"] == 1 and res["accepted"] == 0
    assert _dead_reasons(tmp_path) == ["unknown_enum"]


def test_bad_timestamp(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    res = adapter.ingest(
        pd.DataFrame([dialer_row(0, call_start_ms="not-a-timestamp")]), "cn_dialer_csv"
    )
    assert res["rejected"] == 1 and res["accepted"] == 0
    assert _dead_reasons(tmp_path) == ["bad_timestamp"]


def test_time_inversion_rejected(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    # received an hour before occurred: beyond the 300s tolerance.
    res = adapter.ingest(pd.DataFrame([dialer_row(0, recv_ms=T0_MS - 3_600_000)]), "cn_dialer_csv")
    assert res["rejected"] == 1 and res["accepted"] == 0
    assert _dead_reasons(tmp_path) == ["time_inversion"]


def test_mixed_batch_never_raises(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    rows = [
        dialer_row(0),
        dialer_row(1, cust_id=""),
        dialer_row(2, result_code="BOGUS"),
        dialer_row(3, call_start_ms="garbage"),
        dialer_row(4, recv_ms=T0_MS - 9_999_999),
        dialer_row(5, mobile_no=""),
    ]
    res = adapter.ingest(pd.DataFrame(rows), "cn_dialer_csv")  # must not raise
    assert res["accepted"] == 1
    assert res["rejected"] == 5
    assert len(adapter.read_events()) == 1


# ---------------------------------------------------------------------------
# normalisation + PII discipline
# ---------------------------------------------------------------------------


def test_same_phone_five_formats_same_hash():
    formats = pd.Series(
        ["+91 98765 43210", "09876543210", "98765-43210", "919876543210", "+91-98765-43210"]
    )
    hashes = hash_phone_series(formats)
    assert (hashes == hashes.iloc[0]).all()


def test_raw_values_absent_from_store_and_logs(tmp_path: Path, caplog):
    adapter = make_adapter(tmp_path)
    raw_numbers = [
        "+91 98111 22233",
        "09811122233",
        "98111-22233",
        "919811122233",
        "+91-98111-22233",
    ]
    rows = [dialer_row(i, call_id=f"PII-{i}", mobile_no=n) for i, n in enumerate(raw_numbers)]
    # one malformed row to exercise the dead-letter path with PII present
    rows.append(dialer_row(99, call_id="PII-99", mobile_no=raw_numbers[0], result_code="BOGUS"))
    with caplog.at_level(logging.INFO, logger="src.rpc.ingest"):
        res = adapter.ingest(pd.DataFrame(rows), "cn_dialer_csv")
    assert res["accepted"] == 5 and res["rejected"] == 1
    refs = adapter.read_events()["contact_point_ref"]
    assert (refs == refs.iloc[0]).all()  # all formats -> same hash
    db_path = store_path(tmp_path)
    dumped = ""
    for table in ("events", "dead_letter", "dirty_contact_points", "watermarks"):
        dumped += fetch_table(db_path, table).to_string() + "\n"
    for raw in [*raw_numbers, "9811122233"]:
        assert raw not in dumped, f"raw PII leaked into store: {raw!r}"
        assert raw not in caplog.text, f"raw PII leaked into logs: {raw!r}"


# ---------------------------------------------------------------------------
# tenancy, filters, hidden columns
# ---------------------------------------------------------------------------


def test_tenancy_lender_isolation(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    rows = [dialer_row(0, lender_code="LENDER_A"), dialer_row(1, lender_code="LENDER_B")]
    adapter.ingest(pd.DataFrame(rows), "cn_dialer_csv")
    only_a = adapter.read_events(lender_id="LENDER_A")
    assert len(only_a) == 1 and (only_a["lender_id"] == "LENDER_A").all()
    only_b = read_events(lender_id="LENDER_B", db_path=store_path(tmp_path))
    assert len(only_b) == 1 and (only_b["lender_id"] == "LENDER_B").all()


def test_read_events_filters(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    adapter.ingest(pd.DataFrame([dialer_row(0), dialer_row(1)]), "cn_dialer_csv")
    adapter.ingest([ndjson_row(0)], "cn_disposition_ndjson")
    ref = adapter.read_events().iloc[0]["contact_point_ref"]
    assert len(adapter.read_events(event_types=["disposition"])) == 1
    assert len(adapter.read_events(contact_point_refs=[ref])) >= 1
    assert len(adapter.read_events(received_before="2024-01-01T00:05:00+00:00")) == 0
    assert len(adapter.read_events(received_before="2024-01-01T00:05:01+00:00")) == 1
    assert len(adapter.read_events(received_before="2024-01-01T00:07:00+00:00")) == 2


def test_hidden_ground_truth_columns_dropped(tmp_path: Path):
    adapter = make_adapter(tmp_path)
    df = pd.DataFrame([dialer_row(0)])
    df["true_state"] = "ZZ_HIDDEN_STATE"
    df["borrower_avoiding"] = True
    df["shared_reason"] = "ZZ_HIDDEN_REASON"
    res = adapter.ingest(df, "cn_dialer_csv")
    assert res["accepted"] == 1
    dumped = (
        adapter.read_events().to_string()
        + fetch_table(store_path(tmp_path), "dead_letter").to_string()
    )
    assert "ZZ_HIDDEN" not in dumped


def test_replay_function_is_idempotent(tmp_path: Path):
    path = tmp_path / "dialer.csv"
    pd.DataFrame([dialer_row(i) for i in range(4)]).to_csv(path, index=False)
    db = store_path(tmp_path)
    assert replay(path, "cn_dialer_csv", db_path=db)["accepted"] == 4
    second = replay(path, "cn_dialer_csv", db_path=db)
    assert second == {"accepted": 0, "duplicate": 4, "rejected": 0, "dirty_marked": 0}


def test_module_functions_share_default_contract(tmp_path: Path, monkeypatch):
    db = store_path(tmp_path)
    res = ingest(pd.DataFrame([dialer_row(0)]), "cn_dialer_csv", db_path=db)
    assert set(res) == {"accepted", "duplicate", "rejected", "dirty_marked"}
    assert len(read_events(db_path=db)) == 1
    store = EventStore(db)
    try:
        assert store.counts()["events"] == 1
    finally:
        store.close()
