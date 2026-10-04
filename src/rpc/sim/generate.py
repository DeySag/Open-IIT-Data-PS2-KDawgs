"""Simulator v0 - generates synthetic CN data for development."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml


@dataclass
class SimConfig:
    n_borrowers: int
    n_contact_points_per_borrower_mean: float
    n_events_per_contact_point_mean: int
    product_mix: dict[str, float]
    dpd_bucket_distribution: dict[str, float]
    seed: int = 42


def load_config(path: str, scale: str) -> SimConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    sim = cfg["simulator"]
    n_borrowers = sim[f"n_borrowers_{scale}"]

    return SimConfig(
        n_borrowers=n_borrowers,
        n_contact_points_per_borrower_mean=sim["n_contact_points_per_borrower_mean"],
        n_events_per_contact_point_mean=sim["n_events_per_contact_point_mean"],
        product_mix=sim["product_mix"],
        dpd_bucket_distribution=sim["dpd_bucket_distribution"],
        seed=sim["seed"],
    )


def hash_contact_point(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def generate_borrowers(rng: np.random.Generator, config: SimConfig) -> pd.DataFrame:
    n = config.n_borrowers

    products = list(config.product_mix.keys())
    product_probs = list(config.product_mix.values())

    dpd_buckets = list(config.dpd_bucket_distribution.keys())
    dpd_probs = list(config.dpd_bucket_distribution.values())

    borrowers = []
    for i in range(n):
        borrower_id = f"BORR_{i:07d}"
        lender_id = f"LENDER_{rng.integers(1, 11):03d}"
        product = rng.choice(products, p=product_probs)
        dpd_bucket = rng.choice(dpd_buckets, p=dpd_probs)
        outstanding = rng.lognormal(10, 1.5)  # ~22k median
        dpd_days = {"X": 0, "1-30": 15, "31-60": 45, "61-90": 75, "90+": 120}[dpd_bucket]

        borrowers.append({
            "borrower_id": borrower_id,
            "lender_id": lender_id,
            "product": product,
            "dpd_bucket": dpd_bucket,
            "dpd_days": dpd_days,
            "outstanding": round(outstanding, 2),
            "secured": product in ("secured_retail", "msme"),
        })

    return pd.DataFrame(borrowers)


def generate_contact_points(rng: np.random.Generator, borrowers: pd.DataFrame, config: SimConfig) -> pd.DataFrame:
    rows = []
    sources = ["KYC", "later_update", "bureau", "borrower_on_call", "skip_trace"]
    source_probs = [0.40, 0.25, 0.15, 0.10, 0.10]

    for _, bor in borrowers.iterrows():
        n_cps = max(1, rng.poisson(config.n_contact_points_per_borrower_mean))
        for j in range(n_cps):
            cp_type = "phone" if rng.random() < 0.85 else "address"
            if cp_type == "phone":
                value = f"+91{rng.integers(7000000000, 9999999999)}"
            else:
                value = f"Addr_{bor['borrower_id']}_{j}"

            cp_ref = hash_contact_point(value)
            source = rng.choice(sources, p=source_probs)
            is_primary = j == 0

            rows.append({
                "contact_point_ref": cp_ref,
                "borrower_id": bor["borrower_id"],
                "lender_id": bor["lender_id"],
                "type": cp_type,
                "value_hash": cp_ref,
                "source": source,
                "is_primary": is_primary,
                "created_at": datetime.now(timezone.utc) - timedelta(days=rng.integers(0, 365)),
            })

    return pd.DataFrame(rows)


def generate_events(rng: np.random.Generator, borrowers: pd.DataFrame, cps: pd.DataFrame, config: SimConfig) -> pd.DataFrame:
    """Generate dial attempt events (simplified v0)."""
    rows = []
    network_responses = ["answered", "no_answer", "busy", "switched_off", "not_reachable", "does_not_exist", "immediate_hangup"]
    response_probs = [0.15, 0.45, 0.05, 0.15, 0.10, 0.05, 0.05]

    phone_cps = cps[cps["type"] == "phone"]

    for _, cp in phone_cps.iterrows():
        n_events = max(1, rng.poisson(config.n_events_per_contact_point_mean))
        base_time = datetime.now(timezone.utc) - timedelta(days=180)

        for k in range(n_events):
            event_time = base_time + timedelta(hours=rng.integers(0, 180*24))
            response = rng.choice(network_responses, p=response_probs)
            ring_sec = rng.exponential(5) if response != "answered" else rng.exponential(3)

            rows.append({
                "event_id": str(uuid4()),
                "event_type": "dial_attempt",
                "lender_id": cp["lender_id"],
                "borrower_id": cp["borrower_id"],
                "account_id": f"ACC_{cp['borrower_id'].split('_')[1]}",
                "contact_point_ref": cp["contact_point_ref"],
                "occurred_at": event_time.isoformat(),
                "received_at": (event_time + timedelta(minutes=rng.integers(1, 60))).isoformat(),
                "payload": json.dumps({
                    "network_response": response,
                    "ring_seconds": round(max(0.1, ring_sec), 1),
                }),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("occurred_at").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale", choices=["dev", "full"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config, args.scale)
    rng = np.random.default_rng(config.seed)

    print(f"Generating {config.n_borrowers} borrowers...")

    borrowers = generate_borrowers(rng, config)
    print(f"  Borrowers: {len(borrowers)}")

    cps = generate_contact_points(rng, borrowers, config)
    print(f"  Contact points: {len(cps)}")

    events = generate_events(rng, borrowers, cps, config)
    print(f"  Events: {len(events)}")

    # Save as parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine into single dataset for simplicity
    data = {
        "borrowers": borrowers,
        "contact_points": cps,
        "events": events,
    }

    # Write each table
    for name, df in data.items():
        df.to_parquet(output_path.parent / f"{name}.parquet", index=False)

    print(f"Saved to {output_path.parent}")


if __name__ == "__main__":
    main()