"""Tests for official-extract ingest tasks 1-3: lender join, CLI, trace history.

All source data below are inline test fixtures; nothing here is
real CN data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.rpc.ingest import (
    DuplicateAccountError,
    IngestAdapter,
    attach_lender_id,
    load_lender_lookup,
    load_trace_history,
    read_trace_history,
)
from src.rpc.ingest.dimensions import load_accounts
from src.rpc.ingest.store import EventStore, IngestConfig


def store_path(tmp_path: Path) -> str:
    return str(tmp_path / "store.duckdb")


EXPECTED_TRACE_SUMMARY = {"loaded": 2, "duplicate": 0, "rejected": 0}
EXPECTED_TRACE_SUMMARY_ALL_DUPES = {"loaded": 0, "duplicate": 2, "rejected": 0}
EXPECTED_TRACE_SUMMARY_ONE_REJECT = {"loaded": 2, "duplicate": 0, "rejected": 1}
EXPECTED_TRACE_COST = 150.0


def write_accounts(tmp_path: Path) -> Path:
    path = tmp_path / "accounts.csv"
    pd.DataFrame(
        [
            {"account_id": "ACC_1", "lender_id": "L1"},
            {"account_id": "ACC_2", "lender_id": "L2"},
        ]
    ).to_csv(path, index=False)
    return path


def test_attach_lender_id_fills_and_counts_unmatched(tmp_path: Path):
    lookup = load_lender_lookup(write_accounts(tmp_path))
    raw = pd.DataFrame([{"account_id": "ACC_1"}, {"account_id": "NOPE"}])
    out, unmatched = attach_lender_id(raw, lookup, source="probe")
    assert out["lender_id"].tolist() == ["L1", pd.NA]
    assert unmatched == 1


def test_attach_lender_id_keeps_existing(tmp_path: Path):
    lookup = load_lender_lookup(write_accounts(tmp_path))
    raw = pd.DataFrame([{"account_id": "ACC_1", "lender_id": "L9"}])
    out, unmatched = attach_lender_id(raw, lookup, source="probe")
    assert out["lender_id"].tolist() == ["L9"]
    assert unmatched == 0


def test_load_lender_lookup_rejects_duplicate_accounts(tmp_path: Path):
    path = tmp_path / "dup_accounts.csv"
    pd.DataFrame(
        [
            {"account_id": "ACC_1", "lender_id": "L1"},
            {"account_id": "ACC_1", "lender_id": "L2"},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(DuplicateAccountError):
        load_lender_lookup(path)


def test_unmatched_account_rejected_downstream(tmp_path: Path):
    """An account missing from the lookup stays null and is rejected, not dropped."""
    lookup = load_lender_lookup(write_accounts(tmp_path))
    raw = pd.DataFrame(
        [
            {
                "attempt_id": "A-1",
                "account_id": "ACC_1",
                "phone_id": "P-1",
                "attempt_ts": "2026-04-02 10:00:00",
                "network_response": "answered",
                "ring_duration_s": "3",
            },
            {
                "attempt_id": "A-2",
                "account_id": "GHOST",
                "phone_id": "P-2",
                "attempt_ts": "2026-04-02 10:05:00",
                "network_response": "answered",
                "ring_duration_s": "3",
            },
        ]
    )
    enriched, unmatched = attach_lender_id(raw, lookup, source="cn_dial_attempts")
    assert unmatched == 1
    adapter = IngestAdapter(IngestConfig(db_path=store_path(tmp_path)))
    res = adapter.ingest(enriched, "cn_dial_attempts")
    assert res["accepted"] == 1
    assert res["rejected"] == 1


def trace_rows() -> list[dict]:
    return [
        {
            "trace_id": "T-1",
            "account_id": "ACC_1",
            "trace_date": "2026-04-10",
            "trigger_rule": "no_rpc_30d",
            "result": "found_new_number",
            "new_contact_point_id": "P-9",
            "cost_inr": "150",
        },
        {
            "trace_id": "T-2",
            "account_id": "ACC_2",
            "trace_date": "2026-04-11",
            "trigger_rule": "stale_contacts",
            "result": "not_found",
            "new_contact_point_id": "",
            "cost_inr": "150",
        },
    ]


def test_trace_history_roundtrip_idempotent(tmp_path: Path):
    path = tmp_path / "skip_traces.csv"
    pd.DataFrame(trace_rows()).to_csv(path, index=False)
    db = store_path(tmp_path)
    first = load_trace_history(path, db_path=db)
    assert first == EXPECTED_TRACE_SUMMARY
    second = load_trace_history(path, db_path=db)
    assert second == EXPECTED_TRACE_SUMMARY_ALL_DUPES
    got = read_trace_history(db_path=db)
    assert len(got) == len(trace_rows())
    assert float(got.loc[got["trace_id"] == "T-1", "cost_inr"].iloc[0]) == EXPECTED_TRACE_COST
    acc1 = read_trace_history(account_ids=["ACC_1"], db_path=db)
    assert acc1["trace_id"].tolist() == ["T-1"]


def test_trace_history_rejects_bad_rows(tmp_path: Path):
    path = tmp_path / "skip_traces.csv"
    rows = [
        *trace_rows(),
        {
            "trace_id": "",
            "account_id": "ACC_1",
            "trace_date": "not-a-date",
            "trigger_rule": "x",
            "result": "y",
            "new_contact_point_id": "",
            "cost_inr": "abc",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    res = load_trace_history(path, db_path=store_path(tmp_path))
    assert res == EXPECTED_TRACE_SUMMARY_ONE_REJECT


def test_load_accounts_maps_official_columns_to_store_schema(tmp_path: Path):
    path = tmp_path / "accounts.csv"
    pd.DataFrame(
        [
            {
                "account_id": "ACC_1",
                "lender_id": "L1",
                "portfolio": "unsecured",
                "income_type": "salaried",
                "town_id": "T1",
                "preferred_language": "en",
                "bucket_start": "X",
                "dpd_start": "0",
                "emi_amount": "1000",
                "overdue_start": "0",
                "outstanding": "5000",
                "salary_credit_day": "5",
                "bureau_score_band": "A",
                "other_active_loans": "0",
                "paid_other_lenders_30d": "0",
                "last_bounce_reason": "",
                "ability_to_pay_estimate": "0.5",
                "prev_ptp_count": "0",
                "prev_ptp_broken": "0",
                "dialling_arm": "rule",
            }
        ]
    ).to_csv(path, index=False)

    store = EventStore(db_path=str(tmp_path / "store.duckdb"))
    loaded = load_accounts(store, path)
    assert loaded == 1
    row = store.read_dimension("accounts").iloc[0]
    assert row["account_id"] == "ACC_1"
    assert row["borrower_id"] == "ACC_1"
    assert row["product"] == "unsecured"
    assert row["dpd_bucket"] == "X"
    assert row["town_id"] == "T1"
    assert row["outstanding"] == 5000.0


def test_cli_subset_runs_on_tmp_datasets(tmp_path: Path):
    """End-to-end CLI on a tiny fixture datasets dir (ingest + traces)."""
    accounts = pd.DataFrame(
        [
            {
                "account_id": "ACC_1",
                "lender_id": "L1",
                "portfolio": "unsecured",
                "income_type": "salaried",
                "town_id": "T1",
                "preferred_language": "en",
                "bucket_start": "X",
                "dpd_start": "0",
                "emi_amount": "1000",
                "overdue_start": "0",
                "outstanding": "5000",
                "salary_credit_day": "5",
                "bureau_score_band": "A",
                "other_active_loans": "0",
                "paid_other_lenders_30d": "0",
                "last_bounce_reason": "",
                "ability_to_pay_estimate": "0.5",
                "prev_ptp_count": "0",
                "prev_ptp_broken": "0",
                "dialling_arm": "rule",
            }
        ]
    )
    accounts.to_csv(tmp_path / "accounts.csv", index=False)
    pd.DataFrame(
        [
            {
                "attempt_id": "A-1",
                "account_id": "ACC_1",
                "phone_id": "P-1",
                "attempt_ts": "2026-04-02 10:00:00",
                "channel": "tele_agent",
                "agent_id": "AG1",
                "dialling_arm": "rule",
                "selection_propensity": "0.5",
                "network_response": "answered",
                "ring_duration_s": "3",
                "talk_duration_s": "60",
                "hangup_by": "agent",
                "disposition": "promised_to_pay",
                "remark": "will pay friday",
                "ptp_id": "PTP-1",
                "has_transcript": "False",
            }
        ]
    ).to_csv(tmp_path / "dial_attempts.csv", index=False)
    pd.DataFrame(trace_rows()).to_csv(tmp_path / "skip_traces.csv", index=False)

    db = str(tmp_path / "cli.duckdb")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.rpc.ingest",
            "--datasets",
            str(tmp_path),
            "--db",
            db,
            "--only",
            "cn_dial_attempts",
            "traces",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path.cwd(),
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["sources"]["cn_dial_attempts"]["accepted"] == 1
    assert summary["sources"]["traces"] == EXPECTED_TRACE_SUMMARY
