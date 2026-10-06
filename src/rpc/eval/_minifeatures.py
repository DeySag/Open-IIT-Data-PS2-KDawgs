"""TEMPORARY mini-features for the eval harness.

If ``src.rpc.features`` has landed, :func:`get_feature_builder` returns an
adapter around its ``build_features`` (same 5-argument eval signature, PIT
semantics preserved); until then this module builds a small point-in-time
feature set with DuckDB straight from ``data/events.parquet`` (+ contact_points
when present).

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
    """Return an eval-signature adapter around the real feature layer, else None.

    The real ``build_features(as_of, source, refs)`` takes an ``EventSource``;
    eval call sites use ``builder(as_of, refs, events, contact_points,
    borrowers)``. The adapter bridges the two without changing PIT semantics:
    the in-memory source filters ``received_at <= as_of`` exactly like the
    Parquet one, and only mini columns the real output lacks are backfilled
    (baselines contract: ``n_attempts``/``consec_failures``/...), never
    overwriting real columns.
    """
    try:
        from src.rpc.features import build_features as _real
        from src.rpc.features.source import (
            ALLOWED_BORROWER_COLUMNS,
            DataFrameEventSource,
        )
    except Exception:
        return None

    def _adapter(
        as_of: datetime,
        contact_point_refs: Sequence[str],
        events: pd.DataFrame,
        contact_points: pd.DataFrame | None = None,
        borrowers: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        refs = list(contact_point_refs)
        if borrowers is None:
            borrowers = pd.DataFrame({c: [] for c in ALLOWED_BORROWER_COLUMNS})
        source = DataFrameEventSource(
            events,
            _complete_contact_points(contact_points, events, as_of),
            borrowers,
        )
        real = _real(as_of, source, refs)
        mini = build_minifeatures(as_of, refs, events, contact_points)
        extra = [c for c in mini.columns if c not in real.columns]
        if extra:
            real = real.merge(
                mini[["contact_point_ref", *extra]],
                on="contact_point_ref",
                how="left",
            )
        return real

    return _adapter


def _complete_contact_points(
    contact_points: pd.DataFrame | None,
    events: pd.DataFrame,
    as_of: datetime,
) -> pd.DataFrame:
    """Supply the keys ``DataFrameEventSource.load_contact_points`` needs.

    Eval fixtures often carry only ``(contact_point_ref, account_id,
    is_primary)``. Missing ``lender_id``/``borrower_id`` are backfilled from
    the per-ref mode in events (otherwise the real layer's
    ``(lender, borrower, ref)`` universe would double-count rows); a missing
    ``created_at`` is backfilled with the earliest event time so fixture
    records are visible at any ``as_of``. Assumption: fixture contact points
    are known from the start of the event history.
    """
    cps = pd.DataFrame() if contact_points is None else contact_points.copy()
    if "contact_point_ref" not in cps.columns:
        cps["contact_point_ref"] = pd.Series(dtype=object)
    ev = events.copy() if events is not None else pd.DataFrame()
    for key in ("lender_id", "borrower_id"):
        if key not in cps.columns and key in ev.columns:
            mode = (
                ev.dropna(subset=[key])
                .groupby("contact_point_ref")[key]
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
            )
            cps[key] = cps["contact_point_ref"].map(mode)
    if "created_at" not in cps.columns:
        if not ev.empty and "occurred_at" in ev.columns:
            base = pd.to_datetime(ev["occurred_at"], utc=True).min()
        else:
            base = pd.Timestamp(as_of, tz="UTC")
        cps["created_at"] = base
    return cps


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
    borrowers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """PIT-correct per-contact-point features from raw events (temporary).

    Only rows with ``received_at <= as_of`` contribute. Contact points with no
    history get zero counts and NaN recency fields (callers may fill).
    ``borrowers`` is accepted for signature uniformity with the real-layer
    adapter and ignored (mini-features use no account context).
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
