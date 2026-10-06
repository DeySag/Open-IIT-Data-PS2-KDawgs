"""Observed labels for model training.

IMPORTANT: this module is standalone. It is never imported by
``src.rpc.features.features`` or any other feature code, so no future
information can leak into features. Labels intentionally use events with
``occurred_at`` AFTER ``as_of`` (the outcome window); that is their purpose.

* ``rpc_next_7d``: any disposition with value RPC occurred in
  (as_of, as_of + rpc_next_days].
* ``was_dialled_next_7d``: any dial attempt occurred in the same window.
  Only dialled contact points have a meaningful label; keep this censoring
  flag alongside the label.

Window length comes from ``configs/features.yaml`` (labels.rpc_next_days).
"""

from __future__ import annotations

import pandas as pd

from src.rpc.features.source import EventSource, as_utc
from src.rpc.features.spec import load_feature_config

LABEL_COLUMNS = ["lender_id", "borrower_id", "contact_point_ref", "as_of",
                 "rpc_next_7d", "was_dialled_next_7d"]


def build_labels(
    as_of: str | pd.Timestamp,
    source: EventSource,
    contact_point_refs: list[str] | None = None,
) -> pd.DataFrame:
    """Observed next-window labels for contact points known at ``as_of``."""
    as_of = as_utc(as_of)
    config = load_feature_config()
    window_days: int = int(config["labels"]["rpc_next_days"])
    end = as_of + pd.Timedelta(days=window_days)

    future = source.load_events_window(as_of, end)
    universe = source.load_visible_events(as_of)
    keys = universe[["lender_id", "borrower_id", "contact_point_ref"]].drop_duplicates()
    if contact_point_refs is not None:
        keys = keys[keys["contact_point_ref"].isin(set(contact_point_refs))]
    keys = keys.reset_index(drop=True)
    uidx = pd.MultiIndex.from_frame(keys)

    if future.empty:
        rpc = pd.Series(False, index=uidx)
        dialled = pd.Series(False, index=uidx)
    else:
        disp_norm = future["disposition"].fillna("").astype(str).str.lower() \
            if "disposition" in future.columns else pd.Series("", index=future.index)
        is_rpc = (future["event_type"] == "disposition") & (disp_norm == "rpc")
        rpc_keys = set(map(tuple, future.loc[
            is_rpc, ["lender_id", "borrower_id", "contact_point_ref"]].values.tolist()))
        dial_keys = set(map(tuple, future.loc[
            future["event_type"] == "dial_attempt",
            ["lender_id", "borrower_id", "contact_point_ref"],
        ].values.tolist()))
        rpc = pd.Series([tuple(k) in rpc_keys for k in uidx], index=uidx)
        dialled = pd.Series([tuple(k) in dial_keys for k in uidx], index=uidx)

    result = pd.DataFrame({
        "lender_id": uidx.get_level_values("lender_id").astype("string"),
        "borrower_id": uidx.get_level_values("borrower_id").astype("string"),
        "contact_point_ref": uidx.get_level_values("contact_point_ref").astype("string"),
        "as_of": pd.Series(as_of, index=uidx, dtype="datetime64[ns, UTC]"),
        "rpc_next_7d": rpc.astype("boolean"),
        "was_dialled_next_7d": dialled.astype("boolean"),
    }).reset_index(drop=True)
    return result
