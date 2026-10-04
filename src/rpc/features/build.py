"""CLI: build point-in-time feature snapshots (simulation-only).

Examples:
    python -m src.rpc.features.build --scale dev --as-of 2026-09-01
    python -m src.rpc.features.build --scale dev --as-of-range 2026-08-01 2026-09-01 7
    python -m src.rpc.features.build --scale dev --as-of 2026-09-01 --render-docs
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import pandas as pd

from src.rpc.features.features import build_features, build_training_table
from src.rpc.features.source import ParquetEventSource, as_utc
from src.rpc.features.spec import FEATURES_DOC_PATH, update_features_doc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RPC feature snapshots (simulation-only).")
    p.add_argument("--scale", choices=["dev", "full"], default="dev")
    p.add_argument("--data-dir", default="data",
                   help="Directory with events/borrowers/contact_points.parquet")
    p.add_argument("--as-of", default=None, help="Single snapshot timestamp (UTC).")
    p.add_argument("--as-of-range", nargs=3, default=None, metavar=("START", "END", "STEP_DAYS"),
                   help="Snapshot range: start end step_days.")
    p.add_argument("--out", default=None, help="Output parquet path.")
    p.add_argument("--render-docs", action="store_true",
                   help="Regenerate the registry table in docs/features.md and exit.")
    return p.parse_args(argv)


def resolve_as_ofs(args: argparse.Namespace, source: ParquetEventSource) -> list[pd.Timestamp]:
    if args.as_of_range:
        start, end, step = args.as_of_range
        dates = pd.date_range(start=pd.Timestamp(start, tz="UTC"),
                              end=pd.Timestamp(end, tz="UTC"),
                              freq=f"{int(step)}D", tz="UTC")
        return [as_utc(d) for d in dates]
    if args.as_of:
        return [pd.Timestamp(args.as_of, tz="UTC")]
    latest = source.max_received()
    if latest is None:
        raise ValueError("No events found; pass --as-of explicitly.")
    return [as_utc(latest)]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.render_docs:
        update_features_doc(FEATURES_DOC_PATH)
        print(f"Regenerated registry table in {FEATURES_DOC_PATH}")
        return
    t0 = time.time()
    source = ParquetEventSource(args.data_dir)
    if source.dropped_columns:
        print(f"NOTE: ignoring non-feature columns in source tables: {source.dropped_columns}")
    as_ofs = resolve_as_ofs(args, source)
    out_path = Path(args.out or f"data/features_{args.scale}.parquet")
    if len(as_ofs) == 1:
        result = build_features(as_ofs[0], source)
    else:
        result = build_training_table(as_ofs, source)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"snapshots={len(as_ofs)} rows={len(result)} cols={len(result.columns)} "
          f"out={out_path} seconds={time.time() - t0:.1f} peak_rss_mb={peak_mb:.0f}")


if __name__ == "__main__":
    main()
