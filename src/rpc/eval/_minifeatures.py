"""TEMPORARY mini-features for the eval harness (simulation-only).

DELETE/MIGRATE when the real feature layer lands: if ``src.rpc.features``
exposes ``build_features`` (per its spec.py on main), :func:`get_feature_builder`
returns it; until then this module builds a small point-in-time feature set
with DuckDB straight from ``data/events.parquet`` (+ contact_points when present).

Point-in-time guarantee: every aggregate filters ``received_at <= as_of``.
No ground-truth or policy-log column is ever referenced here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Sequence

import duckdb
import pandas as pd

MINI_FEATURE_COLUMNS = [
    "n_attempts",
    "n_answered",
    "answer_rate",
    "n_recent_7d",
    "n_recent_30d",
    "consec_failures",
    "days_since_last_attempt",
    "days_since_last_answer",
    "avg_ring_seconds",
    "n_switched_off",
    "n_not_reachable",
    "n_does_not_exist",
    "n_no_answer",
    "n_immediate_hangup",
    "is_primary",
]


def try_real_feature_builder() -> Callable | None:
    """Return the real feature builder if the feature layer has landed, else None."""
    try:
        from src.rpc.features import build_features  # type: ignore[import-not-found]

        return build_features
    except Exception:
        return None


def get_feature_builder() -> Callable:
    real = try_real_feature_builder()
    if real is not None:
        return real
    return build_minifeatures


def _norm_events(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    for c in ("occurred_at", "received_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    if "network_response" not in df.columns and "payload" in df.columns:
        import json

        def _nr(p: object) -> str:
            try:
                d = json.loads(p) if isinstance(p, str) else (p or {})
                return str(d.get("network_response", "no_answer"))
            except Exception:
                return "no_answer"

        df["network_response"] = df["payload"].map(_nr)
    if "ring_seconds" not in df.columns and "payload" in df.columns:
        import json

        def _rs(p: object) -> float:
            try:
                d = json.loads(p) if isinstance(p, str) else (p or {})
                return float(d.get("ring_seconds", 0.0))
            except Exception:
                return 0.0

        df["ring_seconds"] = df["payload"].map(_rs)
    return df


def build_minifeatures(
    as_of: datetime,
    contact_point_refs: Sequence[str],
    events: pd.DataFrame,
    contact_points: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """PIT-correct per-contact-point features from raw events (temporary).

    Only rows with ``received_at <= as_of`` contribute. Contact points with no
    history get zero counts and NaN recency fields (callers may fill).
    """
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    refs = list(contact_point_refs)
    base = pd.DataFrame({"contact_point_ref": refs})

    ev = _norm_events(events)
    ev = ev[ev["received_at"] <= as_of_ts]
    ev = ev[ev["contact_point_ref"].isin(set(refs))].copy()
    if ev.empty:
        out = base.copy()
        for c in MINI_FEATURE_COLUMNS:
            out[c] = 0.0 if c != "is_primary" else 0.0
        out["days_since_last_attempt"] = float("nan")
        out["days_since_last_answer"] = float("nan")
        return out

    ev = ev.sort_values(["contact_point_ref", "occurred_at"])
    con = duckdb.connect()
    try:
        con.register("ev", ev)
        agg = con.execute(
            """
            SELECT contact_point_ref,
                   COUNT(*) AS n_attempts,
                   SUM(CASE WHEN network_response = 'answered' THEN 1 ELSE 0 END) AS n_answered,
                   SUM(CASE WHEN occurred_at >= $asof - INTERVAL 7 DAY THEN 1 ELSE 0 END) AS n_recent_7d,
                   SUM(CASE WHEN occurred_at >= $asof - INTERVAL 30 DAY THEN 1 ELSE 0 END) AS n_recent_30d,
                   MAX(occurred_at) AS last_attempt,
                   MAX(CASE WHEN network_response = 'answered' THEN occurred_at ELSE NULL END) AS last_answer,
                   AVG(ring_seconds) AS avg_ring_seconds,
                   SUM(CASE WHEN network_response = 'switched_off' THEN 1 ELSE 0 END) AS n_switched_off,
                   SUM(CASE WHEN network_response = 'not_reachable' THEN 1 ELSE 0 END) AS n_not_reachable,
                   SUM(CASE WHEN network_response = 'does_not_exist' THEN 1 ELSE 0 END) AS n_does_not_exist,
                   SUM(CASE WHEN network_response = 'no_answer' THEN 1 ELSE 0 END) AS n_no_answer,
                   SUM(CASE WHEN network_response = 'immediate_hangup' THEN 1 ELSE 0 END) AS n_immediate_hangup
            FROM ev GROUP BY contact_point_ref
            """,
            {"asof": as_of_ts.to_pydatetime()},
        ).fetchdf()
    finally:
        con.close()

    # Trailing consecutive-failure streak (ordered oldest -> newest per cp).
    streaks: dict[str, int] = {}
    for ref, g in ev.groupby("contact_point_ref"):
        s = 0
        for _, r in g.iterrows():
            s = 0 if r["network_response"] == "answered" else s + 1
        streaks[str(ref)] = s

    out = base.merge(agg, on="contact_point_ref", how="left")
    out["consec_failures"] = out["contact_point_ref"].map(streaks).fillna(0).astype(float)
    for c in ("n_attempts", "n_answered", "n_recent_7d", "n_recent_30d",
              "avg_ring_seconds", "n_switched_off", "n_not_reachable",
              "n_does_not_exist", "n_no_answer", "n_immediate_hangup"):
        out[c] = out[c].fillna(0.0)
    out["answer_rate"] = (out["n_answered"] / out["n_attempts"].replace(0, float("nan"))).fillna(0.0)
    out["days_since_last_attempt"] = (
        (as_of_ts - pd.to_datetime(out["last_attempt"], utc=True)).dt.total_seconds() / 86400.0
    )
    out["days_since_last_answer"] = (
        (as_of_ts - pd.to_datetime(out["last_answer"], utc=True)).dt.total_seconds() / 86400.0
    )
    if contact_points is not None and "is_primary" in contact_points.columns:
        prim = contact_points.set_index("contact_point_ref")["is_primary"]
        out["is_primary"] = out["contact_point_ref"].map(prim).fillna(False).astype(float)
    else:
        out["is_primary"] = 0.0
    return out[["contact_point_ref", *MINI_FEATURE_COLUMNS]]
