"""CLI: build point-in-time feature snapshots from official inputs.

Reads canonical events from the ingest event store, account context from the
official accounts extract, builds features per as-of date, writes parquet.

Q1 borrower grain: the extracts carry no borrower_id, so borrower_id IS
account_id here (borrower-level constructs run at account grain).

Examples:
    python -m src.rpc.features.build --store data/event_store.duckdb \\
        --accounts datasets/accounts.csv --as-of 2026-06-01
    python -m src.rpc.features.build --store data/event_store.duckdb \\
        --accounts datasets/accounts.csv --as-of-range 2026-05-01 2026-06-29 7
    python -m src.rpc.features.build --render-docs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.rpc.features.features import build_features, build_training_table
from src.rpc.features.source import DataFrameEventSource, IngestEventSource, as_utc
from src.rpc.features.spec import FEATURES_DOC_PATH, update_features_doc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RPC feature snapshots.")
    p.add_argument("--store", default="data/event_store.duckdb",
                   help="DuckDB event-store file (ingest output).")
    p.add_argument("--accounts", default="datasets/accounts.csv",
                   help="Official accounts extract (borrower grain = account grain).")
    p.add_argument("--as-of", default=None, help="Single snapshot timestamp (UTC).")
    p.add_argument("--as-of-range", nargs=3, default=None, metavar=("START", "END", "STEP_DAYS"),
                   help="Snapshot range: start end step_days.")
    p.add_argument("--out", default=None, help="Output parquet path.")
    p.add_argument("--with-quarantined", action="store_true",
                   help="Include quarantined account snapshot fields (off by default).")
    p.add_argument("--render-docs", action="store_true",
                   help="Regenerate the registry table in docs/features.md and exit.")
    return p.parse_args(argv)


def load_borrowers_frame(accounts_path: str | None) -> pd.DataFrame:
    """Borrower table for the builder: official accounts with Q1 grain.

    borrower_id IS account_id (no borrower key exists in the extracts).
    Column renames to canonical borrowers names happen in the builder via
    configs account_passthroughs fallbacks; this only adds the key.
    Snapshot fields stay quarantined until their as-of is confirmed.
    """
    if not accounts_path:
        return pd.DataFrame(columns=["borrower_id", "lender_id"])
    frame = pd.read_csv(accounts_path, dtype="string", keep_default_na=True)
    if "account_id" not in frame.columns:
        raise ValueError(f"accounts extract needs account_id: {accounts_path}")
    frame = frame.copy()
    frame["borrower_id"] = frame["account_id"]
    return frame


def build_source(store: str, accounts: str | None) -> DataFrameEventSource:
    """Compose the full in-memory source: store events + derived CPs + accounts.

    Events come from the store UNcanonicalized: DataFrameEventSource runs
    ``canonicalize_events`` itself on init, and canonicalizing twice would
    duplicate the parsed payload columns.
    """
    from src.rpc.ingest import read_events as ingest_read_events

    store_src = IngestEventSource(db_path=store)
    ceiling = store_src.max_received()
    if ceiling is None:
        raise ValueError(f"No events in store; ingest first: {store}")
    as_of_all = ceiling + pd.Timedelta(seconds=1)
    events = ingest_read_events(db_path=store, received_before=as_of_all)
    cps = store_src.load_contact_points(as_of_all)
    borrowers = load_borrowers_frame(accounts)
    return DataFrameEventSource(events, cps, borrowers)


def resolve_as_ofs(args: argparse.Namespace) -> list[pd.Timestamp]:
    if args.as_of_range:
        start, end, step = args.as_of_range
        dates = pd.date_range(start=pd.Timestamp(start, tz="UTC"),
                              end=pd.Timestamp(end, tz="UTC"),
                              freq=f"{int(step)}D", tz="UTC")
        return [as_utc(d) for d in dates]
    if args.as_of:
        return [pd.Timestamp(args.as_of, tz="UTC")]
    return []


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.render_docs:
        update_features_doc(FEATURES_DOC_PATH)
        print(f"Regenerated registry table in {FEATURES_DOC_PATH}")
        return
    t0 = time.time()
    source = build_source(args.store, args.accounts)
    if source.dropped_columns:
        print(f"NOTE: ignoring non-feature columns in source tables: {source.dropped_columns}")
    as_ofs = resolve_as_ofs(args)
    if not as_ofs:
        latest = source.max_received()
        if latest is None:
            raise ValueError("No events found; pass --as-of explicitly.")
        as_ofs = [as_utc(latest)]
    out_path = Path(args.out or "data/features_official.parquet")
    if len(as_ofs) == 1:
        result = build_features(as_ofs[0], source, with_quarantined=args.with_quarantined)
    else:
        result = build_training_table(as_ofs, source, with_quarantined=args.with_quarantined)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(f"snapshots={len(as_ofs)} rows={len(result)} cols={len(result.columns)} "
          f"out={out_path} seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
