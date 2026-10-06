"""Official-extract ingest CLI: ``python -m src.rpc.ingest [options]``.

Ingests all 11 official CSVs from ``datasets/`` into the DuckDB event store:
account-keyed event sources are joined to ``lender_id`` from accounts.csv
first; ``dial_attempts.csv`` yields two event streams (attempt + disposition);
``skip_traces.csv`` goes to the trace-history table (VOI inputs only, never
a predictor). ``accounts``/``lenders``/``agents``/``splits``/``verified`` are
dimensions for later tasks, not events, and are skipped here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.rpc.ingest.adapter import IngestAdapter, load_input
from src.rpc.ingest.enrich import attach_lender_id, load_lender_lookup
from src.rpc.ingest.mapping import load_mapping
from src.rpc.ingest.store import IngestConfig, default_db_path
from src.rpc.ingest.traces import load_trace_history

logger = logging.getLogger(__name__)

# (dataset filename, mapping source, needs lender join). Order matters only
# for log readability; dedupe is by event_id so re-runs are idempotent.
EVENT_PLAN: tuple[tuple[str, str], ...] = (
    ("dial_attempts.csv", "cn_dial_attempts"),
    ("dial_attempts.csv", "cn_dial_dispositions"),
    ("phones.csv", "cn_phones"),
    ("addresses.csv", "cn_addresses"),
    ("payments.csv", "cn_payments"),
    ("field_visits.csv", "cn_field_visits"),
)

TRACES_FILE = "skip_traces.csv"
ACCOUNTS_FILE = "accounts.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="datasets", help="official extracts dir")
    parser.add_argument(
        "--db", default=None, help="event-store path (default: data/event_store.duckdb)"
    )
    parser.add_argument(
        "--accounts",
        default=None,
        help="accounts.csv path (default: <datasets>/accounts.csv)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="run a subset of sources (mapping names, plus 'traces')",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = Path(args.datasets)
    accounts_path = Path(args.accounts) if args.accounts else datasets / ACCOUNTS_FILE

    only = set(args.only) if args.only else None
    plan = [step for step in EVENT_PLAN if only is None or step[1] in only]
    want_traces = only is None or "traces" in only

    missing_files = [f for f, _ in plan if not (datasets / f).exists()]
    if want_traces and not (datasets / TRACES_FILE).exists():
        missing_files.append(TRACES_FILE)
    if not accounts_path.exists():
        missing_files.append(str(accounts_path))
    if missing_files:
        sys.stderr.write(f"missing input files: {sorted(set(missing_files))}\n")
        return 2
    for _, source in plan:
        if load_mapping(source) is None:
            sys.stderr.write(f"missing mapping for source={source}\n")
            return 2

    config = IngestConfig(db_path=args.db or default_db_path())
    lookup = load_lender_lookup(accounts_path)
    adapter = IngestAdapter(config)

    summary: dict[str, dict[str, int]] = {}
    for filename, source in plan:
        raw = load_input(datasets / filename)
        enriched, _ = attach_lender_id(raw, lookup, source=source)
        summary[source] = adapter.ingest(enriched, source)
    if want_traces:
        summary["traces"] = load_trace_history(
            datasets / TRACES_FILE, config=IngestConfig(db_path=config.db_path)
        )

    totals = {
        key: sum(step.get(key, 0) for step in summary.values())
        for key in ("accepted", "loaded", "duplicate", "rejected")
    }
    sys.stdout.write(json.dumps({"sources": summary, "totals": totals}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
