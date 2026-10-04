"""Observed vs oracle labels. The two are always kept separate.

- Observed: ``rpc_next_7d`` among DIALLED contact points (a dial attempt exists in
  the horizon window). Undialled points get ``censored=True`` and are excluded
  from dialled-only metrics (flagged in the report).
- Oracle (eval-only, reads ground truth): true state + reachable flag. Models
  must never see this; only ``src/rpc/eval`` may read ground_truth.parquet.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd

ANSWER_LIKE = {"answered"}


def _payload_field(payload: object, key: str, default: object = None) -> object:
    try:
        d = json.loads(payload) if isinstance(payload, str) else (payload or {})
        return d.get(key, default)
    except Exception:
        return default


def observed_labels(
    events: pd.DataFrame,
    as_of: datetime,
    contact_point_refs: Sequence[str],
    horizon_days: int = 7,
    rpc_responses: Sequence[str] = ("answered",),
    rpc_dispositions: Sequence[str] = ("RPC", "promise_to_pay", "callback"),
) -> pd.DataFrame:
    """Label each ref: RPC observed in (as_of, as_of+horizon] and dialled there.

    ``rpc_next_7d`` is NaN when censored (no dial in window); ``censored`` flags it.
    """
    as_of_ts = pd.to_datetime(as_of, utc=True)
    end_ts = as_of_ts + pd.Timedelta(days=horizon_days)
    ev = events.copy()
    ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)
    win = ev[
        (ev["occurred_at"] > as_of_ts)
        & (ev["occurred_at"] <= end_ts)
        & (ev["contact_point_ref"].isin(set(contact_point_refs)))
    ].copy()
    if win.empty:
        return pd.DataFrame(
            {
                "contact_point_ref": list(contact_point_refs),
                "rpc_next_7d": float("nan"),
                "censored": True,
                "n_dials_window": 0,
            }
        )
    win["is_dial"] = win["event_type"].eq("dial_attempt") | ~win["event_type"].isin(
        ["disposition"]
    )
    # RPC evidence: answered dial OR rpc-like disposition.
    resp_set, disp_set = set(rpc_responses), set(rpc_dispositions)

    def _row_is_rpc(r: object) -> bool:
        et = r["event_type"]  # type: ignore[index]
        if et == "dial_attempt":
            return str(_payload_field(r.get("payload"), "network_response", "")) in resp_set  # type: ignore[union-attr]
        if et == "disposition":
            return str(_payload_field(r.get("payload"), "disposition", "")) in disp_set  # type: ignore[union-attr]
        return False

    win["is_rpc"] = win.apply(_row_is_rpc, axis=1).astype(bool)
    g = win.groupby("contact_point_ref").agg(
        n_dials_window=("is_dial", "sum"), rpc_next_7d=("is_rpc", "max")
    )
    out = pd.DataFrame({"contact_point_ref": list(contact_point_refs)}).merge(
        g, on="contact_point_ref", how="left"
    )
    out["n_dials_window"] = out["n_dials_window"].fillna(0).astype(int)
    dialled = out["n_dials_window"] > 0
    out["censored"] = ~dialled
    out["rpc_next_7d"] = out["rpc_next_7d"].fillna(False).astype(float)
    out.loc[~dialled, "rpc_next_7d"] = float("nan")
    return out


def oracle_labels(
    ground_truth: pd.DataFrame,
    as_of: datetime,
    contact_point_refs: Sequence[str],
    dead_states: Sequence[str] = ("recycled", "invalid", "switched_off_long"),
    reachable_states: Sequence[str] = ("valid_reachable",),
) -> pd.DataFrame:
    """True-state labels as of ``as_of`` (eval-only).

    Expects ground-truth rows with (contact_point_ref, true_state, [valid_from, valid_to]).
    If validity intervals exist, picks the row covering as_of; else the latest row.
    """
    refs = list(contact_point_refs)
    g = ground_truth[ground_truth["contact_point_ref"].isin(set(refs))].copy()
    if g.empty:
        return pd.DataFrame(
            {
                "contact_point_ref": refs,
                "true_state": None,
                "oracle_reachable": float("nan"),
                "oracle_dead": float("nan"),
            }
        )
    as_of_ts = pd.to_datetime(as_of, utc=True)
    if "valid_from" in g.columns:
        g["valid_from"] = pd.to_datetime(g["valid_from"], utc=True)
        g["valid_to"] = pd.to_datetime(g.get("valid_to"), utc=True)
        cover = g[(g["valid_from"] <= as_of_ts) & ((g["valid_to"].isna()) | (g["valid_to"] > as_of_ts))]
        if not cover.empty:
            g = cover
    g = g.sort_values("valid_from" if "valid_from" in g.columns else "true_state").drop_duplicates(
        "contact_point_ref", keep="last"
    )
    out = pd.DataFrame({"contact_point_ref": refs}).merge(
        g[["contact_point_ref", "true_state"]], on="contact_point_ref", how="left"
    )
    out["oracle_reachable"] = out["true_state"].isin(set(reachable_states)).astype(float)
    out["oracle_dead"] = out["true_state"].isin(set(dead_states)).astype(float)
    return out
