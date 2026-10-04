"""Serving configuration.

All tunables live here (and optionally in configs/serve.yaml) rather than
as constants in code. Values are assumptions for CN/SMEs to confirm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ServeConfig:
    """Configuration for the serving layer."""

    # Identity / versioning
    model_version: str = "v0.1.0"
    feature_snapshot_id: str = "fs_dev"

    # Decision validity window
    validity_hours: int = 24

    # Stale-score fallback
    max_score_age_hours: float = 24.0
    confidence_decay_floor: float = 0.1

    # Fast path
    fast_path_max_latency_ms: float = 100.0
    recycled_risk_threshold: float = 0.5

    # Dial list exclusion
    dead_contact_threshold: float = 0.8

    # Trace queue defaults (assumptions)
    default_trace_budget: float = 100000.0
    min_voi_per_rupee: float = 0.5
    trace_cost: float = 500.0
    p_find: float = 0.30
    recovery_if_reached: float = 0.25
    recovery_if_not_reached: float = 0.02
    recoverable_amount_default: float = 50000.0
    collection_cost_fraction: float = 0.10
    compliance_cost_fraction: float = 0.05

    # Security (placeholder)
    api_key: str | None = None

    @classmethod
    def load(cls, path: str | None = None) -> ServeConfig:
        """Load config, optionally merging a YAML file if present."""
        cfg = cls()
        if path:
            p = Path(path)
            if p.exists():
                with open(p) as f:
                    data = yaml.safe_load(f) or {}
                serve = data.get("serve", data)
                for key, value in serve.items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, value)
        return cfg
