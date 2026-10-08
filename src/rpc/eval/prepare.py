"""Eval-input exporter: event store + official extracts -> data/*.parquet.

Fills the missing link between ``make data`` (canonical DuckDB store) and
``make eval`` (parquet frames). Schemas match what ``src/rpc/eval/run.py``
loads (see ``data`` section of ``configs/eval.yaml``):

- events.parquet: canonical envelope, verbatim from the store.
- contact_points.parquet: one row per phone link (hashed ref + account
  context for the real feature layer).
- borrowers.parquet: account grain (``borrower_id = account_id`` — no
  borrower_id exists) with allowed passthrough columns only.
- policy_log.parquet: dial exposure rows for the propensity 1/k check.
- splits.csv: copied verbatim (official file = routing only).
- verified_gold.parquet: gold annotations with hashed refs (eval-only).

Join-key rule: ``contact_point_ref`` is the store's own digest. Phone IDs
are re-hashed with the same pepper setting (``resolve_pepper``); the run
aborts when re-hashed refs don't match the store (pepper mismatch), rather
than emitting silently unjoinable frames. Deterministic: no sampling.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.rpc.features.source import ALLOWED_BORROWER_COLUMNS
from src.rpc.ingest.normalize import hash_id_series, resolve_pepper
from src.rpc.ingest.store import EventStore


def hash_refs(ids: pd.Series, pepper: str | None) -> pd.Series:
    """Stable opaque refs for source IDs (same transform as ingest)."""
    return hash_id_series(ids.astype("string").fillna(""), pepper)


def build_contact_points(
    phones: pd.DataFrame, lender_of: pd.Series, pepper: str | None
) -> pd.DataFrame:
    """One row per phone link with hashed ref and account context."""
    out = pd.DataFrame({
        "contact_point_ref": hash_refs(phones["phone_id"], pepper),
        "account_id": phones["account_id"].astype("string"),
        "borrower_id": phones["account_id"].astype("string"),
        "lender_id": phones["account_id"].astype("string").map(lender_of),
        "type": "phone",
        "source": phones["source"].astype("string") if "source" in phones.columns else "KYC",
        "is_primary": (pd.to_numeric(phones["priority_slot"], errors="coerce") == 0)
        if "priority_slot" in phones.columns else False,
        "created_at": pd.to_datetime(phones["added_date"], utc=True)
        if "added_date" in phones.columns else pd.NaT,
    })
    return out


def build_borrowers(accounts: pd.DataFrame) -> pd.DataFrame:
    """Account-grain borrower frame (allowed columns only)."""
    out = pd.DataFrame({"borrower_id": accounts["account_id"].astype("string")})
    for col in ALLOWED_BORROWER_COLUMNS:
        if col in ("borrower_id",):
            continue
        if col in accounts.columns:
            out[col] = accounts[col]
    if "lender_id" not in out.columns and "lender_id" in accounts.columns:
        out["lender_id"] = accounts["lender_id"].astype("string")
    return out


def build_policy_log(dials: pd.DataFrame, pepper: str | None) -> pd.DataFrame:
    """Dial exposure rows (all dialled=1; candidates never exposed are unknown)."""
    return pd.DataFrame({
        "contact_point_ref": hash_refs(dials["phone_id"], pepper),
        "account_id": dials["account_id"].astype("string"),
        "dialling_arm": dials["dialling_arm"].astype("string")
        if "dialling_arm" in dials.columns else "unknown",
        "selection_propensity": pd.to_numeric(dials["selection_propensity"], errors="coerce")
        if "selection_propensity" in dials.columns else 1.0,
        "dialled": True,
        "occurred_at": pd.to_datetime(dials["attempt_ts"], utc=True),
    })


def build_verified_gold(verified: pd.DataFrame, pepper: str | None) -> pd.DataFrame:
    """Gold annotations with hashed refs (eval-only, never features)."""
    return pd.DataFrame({
        "contact_point_ref": hash_refs(verified["phone_id"], pepper),
        "account_id": verified["account_id"].astype("string"),
        "verified_status": verified["verified_status"].astype("string"),
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/event_store.duckdb")
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)
    ds, out = Path(args.datasets), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pepper = resolve_pepper()

    store = EventStore(db_path=str(args.db))
    try:
        events = store.read_events()
    finally:
        store.close()
    if events.empty:
        sys.stderr.write("no events in store: run `make data` first\n")
        return 2
    store_refs = set(events["contact_point_ref"].astype(str))
    events.to_parquet(out / "events.parquet", index=False)

    accounts = pd.read_csv(ds / "accounts.csv", dtype="string")
    lender_of = accounts.set_index("account_id")["lender_id"].astype(str)
    phones = pd.read_csv(ds / "phones.csv", dtype="string")
    cps = build_contact_points(phones, lender_of, pepper)
    unknown = set(cps["contact_point_ref"]) - store_refs
    sys.stdout.write(f"contact_points={len(cps)} refs_new_to_store={len(unknown)}\n")
    cps.to_parquet(out / "contact_points.parquet", index=False)

    build_borrowers(accounts).to_parquet(out / "borrowers.parquet", index=False)

    dials = pd.read_csv(ds / "dial_attempts.csv", dtype="string")
    plog = build_policy_log(dials, pepper)
    orphans = set(plog["contact_point_ref"]) - store_refs
    if orphans:
        sys.stderr.write(
            f"pepper mismatch: {len(orphans)} dial refs absent from store\n")
        return 2
    plog.to_parquet(out / "policy_log.parquet", index=False)

    shutil.copyfile(ds / "splits.csv", out / "splits.csv")
    verified = pd.read_csv(ds / "verified_contact_points.csv", dtype="string")
    build_verified_gold(verified, pepper).to_parquet(out / "verified_gold.parquet", index=False)

    sys.stdout.write(f"events={len(events)} borrowers={len(accounts)} policy={len(plog)} "
                     f"verified={len(verified)} splits=copied\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
