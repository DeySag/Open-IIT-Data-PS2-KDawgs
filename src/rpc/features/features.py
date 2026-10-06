"""Point-in-time feature builder.

One row per phone/address contact point known at ``as_of``. Point-in-time
rules (non-negotiable, enforced by tests):

1. Only events with ``received_at <= as_of`` are visible. Window membership
   uses ``occurred_at``. Late events (occurred <= as_of < received) are
   excluded until ``as_of`` passes their ``received_at``.
2. Deduplicated by ``event_id`` (earliest ``received_at``) in the source layer.
3. Contact points created/first-seen after ``as_of`` are absent.
4. Nothing after ``as_of`` is used, including days-since arithmetic.
5. Never-attempted contact points still get a row: ``has_any_attempt=False``,
   counts are 0, rates and days-since are null (never 0-imputed).
6. Slots/weekends use Asia/Kolkata (from configs).

Everything is vectorised (groupby/merge/map); no per-contact-point loops,
except a borrower-level payment-link loop that only touches borrowers having
both qualifying events and payments, and a union-find over sharing groups.
"""

from __future__ import annotations

import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.rpc.features.source import EventSource, as_utc
from src.rpc.features.spec import (
    KEY_COLUMNS,
    META_COLUMNS,
    build_registry,
    config_text_for_hash,
    load_feature_config,
    snapshot_id,
    window_suffix,
)
from src.rpc.features.text import (
    REMARK_GROUPS,
    count_cues,
    extract_switched_off_months,
)

CP_KEYS = ["lender_id", "borrower_id", "contact_point_ref"]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _days_since(as_of: pd.Timestamp, ts: pd.Series) -> pd.Series:
    """Date-based days from ts to as_of (UTC dates), nullable Int64."""
    base = as_of.floor("D")
    out = (base - ts.dt.floor("D")).dt.days
    return out.astype("Int64")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise num/den, null where den is 0/None (never 0-imputed)."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den.replace(0, np.nan)
    return out.astype("Float64")


def _sum_by_cp(values: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """Sum values per contact point, indexed by the CP MultiIndex (vectorised)."""
    mi = pd.MultiIndex.from_arrays([frame[k].tolist() for k in CP_KEYS], names=CP_KEYS)
    return values.groupby(mi, sort=False).sum()


def _max_by_cp(values: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """Max values per contact point, indexed by the CP MultiIndex (vectorised)."""
    mi = pd.MultiIndex.from_arrays([frame[k].tolist() for k in CP_KEYS], names=CP_KEYS)
    return values.groupby(mi, sort=False).max()


def _trailing_runlength(sorted_vals: pd.Series, groups: pd.Series) -> pd.Series:
    """Length of the trailing run of each row's group-last value (vectorised).

    ``sorted_vals`` must already be ordered by (group, occurred_at,
    received_at). Returns, per row, the run length for its group; callers
    take the last row per group.
    """
    last = sorted_vals.groupby(groups, sort=False).transform("last")
    same = (sorted_vals == last).fillna(False)
    pos = groups.groupby(groups, sort=False).cumcount()
    bad = pos.where(~same, -1)
    last_bad = bad.groupby(groups, sort=False).transform("max")
    size = groups.groupby(groups, sort=False).transform("size")
    return (size - 1 - last_bad).astype("int64")


def _resolve_account_id(events: pd.DataFrame, universe_idx: pd.MultiIndex) -> pd.Series:
    """Mode account_id per contact point; deterministic fallback when none."""
    if events.empty or "account_id" not in events.columns:
        mode: dict[tuple, Any] = {}
    else:
        ev = events.dropna(subset=["account_id"])
        if ev.empty:
            mode = {}
        else:
            counts = ev.groupby([*CP_KEYS, "account_id"], sort=False).size()
            counts = counts.reset_index(name="_n").sort_values(
                [*CP_KEYS, "_n", "account_id"], ascending=[True, True, True, False, True]
            )
            best = counts.drop_duplicates(subset=CP_KEYS, keep="last")
            mode = {
                tuple(r[k] for k in CP_KEYS): r["account_id"]
                for _, r in best.iterrows()
            }
    out = []
    for key in universe_idx:
        if key in mode:
            out.append(mode[key])
        else:
            borrower = key[1]
            out.append(f"ACC_{borrower.split('_', 1)[1]}" if "_" in borrower else borrower)
    return pd.Series(out, index=universe_idx, dtype="string")


def _union_find_component_sizes(
    universe: pd.DataFrame,
) -> pd.Series:
    """Lender-local borrower sharing components (union-find over sharing groups).

    Nodes are (lender_id, borrower_id); edges join borrowers sharing a
    contact_point_ref within the lender. Returns component size per universe row.
    """
    parent: dict[tuple, tuple] = {}

    def find(x: tuple) -> tuple:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple, b: tuple) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    nodes = list(
        universe[["lender_id", "borrower_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for n in nodes:
        parent[n] = n
    for (_lender, _ref), g in universe.groupby(["lender_id", "contact_point_ref"], sort=False):
        members = [(_lender, b) for b in g["borrower_id"].unique()]
        for m in members[1:]:
            union(members[0], m)
    comp: dict[tuple, int] = {}
    for n in nodes:
        r = find(n)
        comp[r] = comp.get(r, 0) + 1
    sizes = [comp[find((r.lender_id, r.borrower_id))] for r in universe.itertuples()]
    return pd.Series(sizes, index=universe.index, dtype="Int64")


# --------------------------------------------------------------------------
# main builder
# --------------------------------------------------------------------------

def build_features(
    as_of: str | pd.Timestamp,
    source: EventSource,
    contact_point_refs: list[str] | None = None,
) -> pd.DataFrame:
    """Build one feature row per contact point known at ``as_of``."""
    t0 = time.time()
    as_of = as_utc(as_of)
    config = load_feature_config()
    fcfg = config["features"]
    windows: list[int] = list(fcfg["windows_days"])
    short_ring: float = float(fcfg["short_ring_seconds"])
    confirm_days: int = int(fcfg["payment_confirmation_days"])
    ist = ZoneInfo(fcfg["timezone"])
    slot_bounds = {k: tuple(v) for k, v in fcfg["slots"].items()}
    holidays = set(fcfg.get("holidays", []) or [])
    # Dispositions counting as borrower contact (RPC evidence). The mapped
    # extracts emit the rpc_* family as promise_to_pay/callback/dispute/RPC;
    # matching the literal "rpc" alone would miss all of them.
    rpc_family = {str(v).lower() for v in fcfg.get("rpc_dispositions", ["RPC"])}

    events = source.load_visible_events(as_of)
    cps = source.load_contact_points(as_of)
    borrowers = source.load_borrowers()
    include_agent = bool(
        (events["event_type"] == "disposition").any()
        and events.loc[events["event_type"] == "disposition", "agent_id"].notna().any()
    )

    watermark = events["received_at"].max() if not events.empty else pd.NaT
    watermark_str = (
        as_utc(watermark).isoformat() if pd.notna(watermark) else as_of.isoformat()
    )
    snap = snapshot_id(config_text_for_hash(), as_of.isoformat(), watermark_str)

    # ---- universe ------------------------------------------------------
    cp_base = cps[[c for c in
                   ["lender_id", "borrower_id", "contact_point_ref", "type",
                    "source", "is_primary", "created_at"] if c in cps.columns]].copy()
    if not events.empty:
        first_seen = events.groupby(CP_KEYS, sort=False)["received_at"].min().rename("first_seen")
    else:
        first_seen = pd.Series(dtype="datetime64[ns, UTC]", name="first_seen")
    universe = pd.merge(
        cp_base, events[CP_KEYS].drop_duplicates() if not events.empty
        else pd.DataFrame(columns=CP_KEYS),
        on=CP_KEYS, how="outer",
    )
    universe = universe.merge(first_seen.rename("first_seen"), left_on=CP_KEYS,
                              right_index=True, how="left")
    if "created_at" not in universe.columns:
        universe["created_at"] = pd.NaT
    universe["created_at"] = universe["created_at"].fillna(universe["first_seen"])
    universe = universe.drop(columns=["first_seen"])
    if contact_point_refs is not None:
        universe = universe[universe["contact_point_ref"].isin(set(contact_point_refs))]
    universe = universe.reset_index(drop=True)

    if universe.empty:
        return _empty_output(as_of, watermark, snap, config, include_agent)

    uidx = pd.MultiIndex.from_frame(universe[CP_KEYS])
    universe.index = uidx

    # contact type resolution: table -> latest update -> dial/bot/field evidence
    upd_all = events[events["event_type"] == "contact_point_update"].copy() if not events.empty \
        else events.copy()
    if not upd_all.empty:
        upd_all = upd_all.sort_values(["received_at", "occurred_at"], kind="mergesort")
        last_upd_type = upd_all.drop_duplicates(subset=CP_KEYS, keep="last").set_index(CP_KEYS)[
            "contact_type"]
    else:
        last_upd_type = pd.Series(dtype=object)
    has_phone_ev = set()
    has_field_ev = set()
    if not events.empty:
        phone_ev = events[events["event_type"].isin(
            ["dial_attempt", "disposition", "bot_transcript"])]
        if not phone_ev.empty:
            has_phone_ev = set(map(tuple, phone_ev[CP_KEYS].drop_duplicates().values.tolist()))
        field_ev = events[events["event_type"] == "field_visit"]
        if not field_ev.empty:
            has_field_ev = set(map(tuple, field_ev[CP_KEYS].drop_duplicates().values.tolist()))
    cp_type = universe["type"] if "type" in universe.columns else pd.Series(None, index=uidx)
    resolved_types = []
    for key, t in zip(uidx, cp_type.tolist()):
        if isinstance(t, str) and t in ("phone", "address"):
            resolved_types.append(t)
        elif key in last_upd_type.index and last_upd_type.loc[key] in ("phone", "address"):
            resolved_types.append(last_upd_type.loc[key])
        elif key in has_phone_ev:
            resolved_types.append("phone")
        elif key in has_field_ev:
            resolved_types.append("address")
        else:
            resolved_types.append("phone")  # documented fallback
    universe["contact_point_type"] = pd.Series(resolved_types, index=uidx, dtype="string")

    out = pd.DataFrame(index=uidx)
    out["contact_point_type"] = universe["contact_point_type"]

    # ---- dial attempts --------------------------------------------------
    dial = events[events["event_type"] == "dial_attempt"].copy() if not events.empty \
        else events.copy()
    if not dial.empty:
        dial["occurred_at"] = pd.to_datetime(dial["occurred_at"], utc=True)
        dial = dial.sort_values([*CP_KEYS, "occurred_at", "received_at"], kind="mergesort")
        dial_ist = dial["occurred_at"].dt.tz_convert(ist)
        dial["ist_date"] = dial_ist.dt.floor("D")
        hour = dial_ist.dt.hour
        dial["slot"] = None
        for slot, (lo, hi) in slot_bounds.items():
            dial.loc[(hour >= lo) & (hour < hi), "slot"] = slot
        dial["weekend"] = dial_ist.dt.weekday >= 5
        dial["answered"] = (dial["network_response"] == "answered").fillna(False)
        dial["failed"] = ~dial["answered"]
        dial["resp"] = dial["network_response"].fillna("missing")
    has_attempt = pd.Series(False, index=uidx)
    if not dial.empty:
        has_attempt = uidx.isin(pd.MultiIndex.from_frame(dial[CP_KEYS].drop_duplicates()))
        has_attempt = pd.Series(has_attempt, index=uidx)
    out["has_any_attempt"] = has_attempt.astype("boolean")

    for w in windows:
        sfx = window_suffix(w)
        start = as_of - pd.Timedelta(days=w)
        dw = dial[(dial["occurred_at"] > start) & (dial["occurred_at"] <= as_of)] \
            if not dial.empty else dial
        if dw.empty:
            grp_size = pd.Series(dtype="float64")
        else:
            grp_size = dw.groupby(CP_KEYS, sort=False).size()
        n_att = grp_size.reindex(uidx).fillna(0).astype("Int64")
        out[f"n_attempts{sfx}"] = n_att
        if dw.empty:
            out[f"n_answered{sfx}"] = pd.Series(0, index=uidx, dtype="Int64")
            out[f"answer_rate{sfx}"] = pd.Series(np.nan, index=uidx, dtype="Float64")
            for resp in fcfg["response_values"]:
                out[f"n_{resp}{sfx}"] = pd.Series(0, index=uidx, dtype="Int64")
            for slot in ("morning", "afternoon", "evening"):
                out[f"n_attempts_{slot}{sfx}"] = pd.Series(0, index=uidx, dtype="Int64")
                out[f"answer_rate_{slot}{sfx}"] = pd.Series(np.nan, index=uidx, dtype="Float64")
            out[f"weekend_attempt_share{sfx}"] = pd.Series(np.nan, index=uidx, dtype="Float64")
            out[f"ring_seconds_mean{sfx}"] = pd.Series(np.nan, index=uidx, dtype="Float64")
            out[f"ring_seconds_std{sfx}"] = pd.Series(np.nan, index=uidx, dtype="Float64")
            out[f"short_ring_rate{sfx}"] = pd.Series(np.nan, index=uidx, dtype="Float64")
            continue
        g = dw.groupby(CP_KEYS, sort=False)
        n_ans = g["answered"].sum().reindex(uidx).fillna(0).astype("Int64")
        out[f"n_answered{sfx}"] = n_ans
        out[f"answer_rate{sfx}"] = _safe_div(n_ans, n_att).where(n_att > 0)
        for resp in fcfg["response_values"]:
            out[f"n_{resp}{sfx}"] = _sum_by_cp(
                (dw["resp"] == resp).astype("int64"), dw).reindex(uidx).fillna(0).astype("Int64")
        for slot in ("morning", "afternoon", "evening"):
            in_slot = dw["slot"] == slot
            slot_n = dw[in_slot].groupby(CP_KEYS, sort=False).size()
            slot_n = slot_n.reindex(uidx).fillna(0).astype("Int64")
            out[f"n_attempts_{slot}{sfx}"] = slot_n
            slot_a = dw[in_slot & (dw["answered"])].groupby(CP_KEYS, sort=False).size()
            slot_a = slot_a.reindex(uidx).fillna(0)
            out[f"answer_rate_{slot}{sfx}"] = _safe_div(slot_a, slot_n).where(slot_n > 0)
        we = g["weekend"].sum().reindex(uidx).fillna(0)
        out[f"weekend_attempt_share{sfx}"] = _safe_div(we, n_att).where(n_att > 0)
        ring = pd.to_numeric(dw["ring_seconds"], errors="coerce")
        key_tuples = [dw[k].tolist() for k in CP_KEYS]
        ring_mi = pd.MultiIndex.from_arrays(key_tuples, names=CP_KEYS)
        ring_mean = ring.groupby(ring_mi).mean()
        out[f"ring_seconds_mean{sfx}"] = ring_mean.reindex(uidx).astype("Float64")
        ring_std = ring.groupby(ring_mi).std(ddof=1)
        out[f"ring_seconds_std{sfx}"] = ring_std.reindex(uidx).astype("Float64")
        short = (ring < short_ring).groupby(ring_mi).mean()
        out[f"short_ring_rate{sfx}"] = short.reindex(uidx).astype("Float64").where(n_att > 0)

    # ---- dial scalars ------------------------------------------------------
    if dial.empty:
        out["last_response_type"] = pd.Series(None, index=uidx, dtype="string")
        for c in ("consecutive_failures", "consecutive_same_response",
                  "days_since_first_attempt", "days_since_last_attempt",
                  "days_since_last_answer", "mean_gap_between_attempts_days"):
            out[c] = pd.Series(np.nan, index=uidx, dtype="Int64" if c != "mean_gap_between_attempts_days" else "Float64")
        out["system_fail_rate_on_last_attempt_day"] = pd.Series(np.nan, index=uidx, dtype="Float64")
        last_ist_date = pd.Series(pd.NaT, index=uidx)
    else:
        gkeys = dial[CP_KEYS].apply(tuple, axis=1)
        last_resp = dial.groupby(CP_KEYS, sort=False)["resp"].last()
        out["last_response_type"] = last_resp.reindex(uidx).astype("string")
        run_same = _trailing_runlength(dial["resp"], gkeys)
        dial["_run_same"] = run_same.values
        out["consecutive_same_response"] = dial.drop_duplicates(
            subset=CP_KEYS, keep="last").set_index(CP_KEYS)["_run_same"].reindex(uidx).astype("Int64")
        fail_int = dial["failed"].astype("int64")
        # trailing failures: 0 when last attempt answered, else trailing run of failures
        last_failed = dial.groupby(CP_KEYS, sort=False)["failed"].last()
        run_fail = _trailing_runlength(fail_int.astype(str), gkeys)
        dial["_run_fail"] = run_fail.values
        run_fail_last = dial.drop_duplicates(subset=CP_KEYS, keep="last").set_index(CP_KEYS)["_run_fail"]
        run_fail_last = run_fail_last.reindex(uidx)
        last_failed_r = last_failed.reindex(uidx)
        cf = pd.Series(np.nan, index=uidx, dtype="Float64")
        cf = cf.where(last_failed_r.isna(), run_fail_last.where(
            last_failed_r.fillna(False).astype(bool), 0))
        out["consecutive_failures"] = cf.astype("Int64")
        first_occ = dial.groupby(CP_KEYS, sort=False)["occurred_at"].first().reindex(uidx)
        last_occ = dial.groupby(CP_KEYS, sort=False)["occurred_at"].last().reindex(uidx)
        out["days_since_first_attempt"] = _days_since(as_of, first_occ)
        out["days_since_last_attempt"] = _days_since(as_of, last_occ)
        last_ans = dial[dial["answered"]].groupby(CP_KEYS, sort=False)["occurred_at"].last()
        out["days_since_last_answer"] = _days_since(as_of, last_ans.reindex(uidx))
        gaps = dial.groupby(CP_KEYS, sort=False)["occurred_at"].diff().dt.total_seconds() / 86400
        mean_gap = gaps.groupby(gkeys, sort=False).mean()
        mean_gap.index = pd.MultiIndex.from_tuples(mean_gap.index.tolist(), names=CP_KEYS) \
            if len(mean_gap) else mean_gap.index
        out["mean_gap_between_attempts_days"] = mean_gap.reindex(uidx).astype("Float64")
        day_fail = dial.groupby("ist_date", sort=False)["failed"].mean()
        last_ist_date = dial.groupby(CP_KEYS, sort=False)["ist_date"].last().reindex(uidx)
        out["system_fail_rate_on_last_attempt_day"] = last_ist_date.map(day_fail).astype("Float64")
        dial.drop(columns=["_run_same", "_run_fail"], inplace=True, errors="ignore")

    # ---- dispositions + remark cues ------------------------------------------
    disp = events[events["event_type"] == "disposition"].copy() if not events.empty \
        else events.copy()
    if not disp.empty:
        disp["disp_norm"] = disp["disposition"].fillna("").astype(str).str.lower()
        disp = disp.sort_values([*CP_KEYS, "occurred_at", "received_at"], kind="mergesort")
    for d in fcfg["disposition_values"]:
        if disp.empty:
            out[f"n_{d}"] = pd.Series(0, index=uidx, dtype="Int64")
        else:
            out[f"n_{d}"] = disp[disp["disp_norm"] == d].groupby(
                CP_KEYS, sort=False).size().reindex(uidx).fillna(0).astype("Int64")
    if disp.empty:
        out["wrong_number_rate"] = pd.Series(np.nan, index=uidx, dtype="Float64")
        out["last_disposition"] = pd.Series(None, index=uidx, dtype="string")
        out["days_since_last_disposition"] = pd.Series(None, index=uidx, dtype="Int64")
        for feat in REMARK_GROUPS.values():
            out[feat] = pd.Series(0, index=uidx, dtype="Int64")
        out["switched_off_months_max"] = pd.Series(None, index=uidx, dtype="Int64")
        out["days_since_last_rpc"] = pd.Series(None, index=uidx, dtype="Int64")
    else:
        n_disp = disp.groupby(CP_KEYS, sort=False).size().reindex(uidx).fillna(0)
        out["wrong_number_rate"] = _safe_div(out["n_wrong_number"], n_disp).where(n_disp > 0)
        out["last_disposition"] = disp.drop_duplicates(
            subset=CP_KEYS, keep="last").set_index(CP_KEYS)["disposition"].reindex(uidx).astype("string")
        last_disp_t = disp.groupby(CP_KEYS, sort=False)["occurred_at"].last().reindex(uidx)
        out["days_since_last_disposition"] = _days_since(as_of, last_disp_t)
        remarks = disp["remarks"] if "remarks" in disp.columns else pd.Series(None, index=disp.index)
        for group, feat in REMARK_GROUPS.items():
            hit = count_cues(remarks, group)
            out[feat] = _sum_by_cp(hit, disp).reindex(uidx).fillna(0).astype("Int64")
        months = extract_switched_off_months(remarks)
        out["switched_off_months_max"] = _max_by_cp(months, disp).reindex(uidx).astype("Int64")
        last_rpc = disp[disp["disp_norm"].isin(rpc_family)].groupby(
            CP_KEYS, sort=False)["occurred_at"].last()
        out["days_since_last_rpc"] = _days_since(as_of, last_rpc.reindex(uidx))
    if include_agent:
        agent_rate = disp.groupby("agent_id", sort=False)["disp_norm"].apply(
            lambda s: float((s == "wrong_number").mean()))
        last_agent = disp.drop_duplicates(subset=CP_KEYS, keep="last").set_index(CP_KEYS)["agent_id"]
        out["agent_wrong_number_rate"] = last_agent.reindex(uidx).map(agent_rate).astype("Float64")

    # NOTE: no voice-bot transcript block by design. The issued extracts
    # carry no transcript table, so bot columns would be permanently null.
    # The remark-cue machinery above covers the observable text instead.

    # ---- shared contacts (lender-local) ------------------------------------------
    share = universe.reset_index(drop=True)
    grp_ref = share.groupby(["lender_id", "contact_point_ref"], sort=False)
    n_bor = grp_ref["borrower_id"].nunique()
    key_tuples = pd.MultiIndex.from_frame(share[["lender_id", "contact_point_ref"]])
    out["n_borrowers_sharing_cp"] = pd.Series(
        key_tuples.map(n_bor).astype("Int64").tolist(), index=uidx, dtype="Int64")
    out["is_shared"] = (out["n_borrowers_sharing_cp"] > 1).astype("boolean")
    # accounts: resolve account ids first (needed below anyway)
    account_ids = _resolve_account_id(events, uidx)
    phone_mask = (universe["contact_point_type"] == "phone").values
    n_phones = share[phone_mask].groupby("borrower_id", sort=False).size() \
        if phone_mask.any() else pd.Series(dtype="float64")
    out["n_phone_cps_for_borrower"] = universe["borrower_id"].map(n_phones).fillna(0).astype("Int64")
    order = share.sort_values(["borrower_id", "created_at", "contact_point_ref"], kind="mergesort")
    rank = order.groupby("borrower_id", sort=False).cumcount() + 1
    out["cp_rank_within_borrower"] = pd.Series(
        rank.sort_index().astype("Int64").tolist(), index=uidx, dtype="Int64")
    if not upd_all.empty:
        upd_last = upd_all.drop_duplicates(subset=CP_KEYS, keep="last").set_index(CP_KEYS)
        upd_primary = upd_last["is_primary"]
        upd_source = upd_last["source"] if "source" in upd_last.columns else pd.Series(dtype=object)
    else:
        upd_primary = pd.Series(dtype=object)
        upd_source = pd.Series(dtype=object)
    table_primary = universe["is_primary"] if "is_primary" in universe.columns \
        else pd.Series(None, index=uidx)
    is_prim = pd.Series(uidx.map(
        lambda k: (None if k not in upd_primary.index or upd_primary.loc[k] is None
                   else bool(upd_primary.loc[k]))), index=uidx)
    table_primary_r = pd.Series(table_primary.values, index=uidx)
    out["is_primary"] = is_prim.fillna(table_primary_r).fillna(False).astype("boolean")
    out["connected_component_size"] = pd.Series(
        _union_find_component_sizes(
            share[["lender_id", "borrower_id", "contact_point_ref"]]
        ).astype("Int64").tolist(), index=uidx, dtype="Int64")

    # ---- record history --------------------------------------------------------------
    table_source = universe["source"] if "source" in universe.columns \
        else pd.Series(None, index=uidx)
    if not upd_all.empty:
        n_upd = upd_all.groupby(CP_KEYS, sort=False).size().reindex(uidx).fillna(0).astype("Int64")
        last_upd_t = upd_all.groupby(CP_KEYS, sort=False)["occurred_at"].last().reindex(uidx)
        upd_src_last = upd_source.reindex(uidx)
    else:
        n_upd = pd.Series(0, index=uidx, dtype="Int64")
        last_upd_t = pd.Series(pd.NaT, index=uidx, dtype="datetime64[ns, UTC]")
        upd_src_last = pd.Series(None, index=uidx)
    out["n_updates"] = n_upd
    out["days_since_last_update"] = _days_since(as_of, last_upd_t)
    src_final = upd_src_last.where(upd_src_last.notna(),
                                   pd.Series(table_source.values, index=uidx)).fillna("unknown")
    out["source"] = src_final.astype("string")
    created_ts = pd.to_datetime(universe["created_at"], utc=True)
    created_ts = pd.Series(created_ts.array, index=uidx)
    out["record_age_days"] = _days_since(as_of, created_ts)

    # confirmed by payment: payment within confirm_days after an answered/RPC event
    pay = events[events["event_type"] == "payment"].copy() if not events.empty else events.copy()
    if not pay.empty:
        pay["occurred_at"] = pd.to_datetime(pay["occurred_at"], utc=True)
        pay = pay.sort_values(["borrower_id", "occurred_at"], kind="mergesort")
    qual_frames = []
    if not dial.empty:
        q = dial[dial["answered"]][["borrower_id", "occurred_at"]].copy()
        q["cp_tuple"] = list(map(tuple, dial.loc[dial["answered"], CP_KEYS].values.tolist()))
        qual_frames.append(q[["borrower_id", "cp_tuple", "occurred_at"]])
    if not disp.empty:
        rpc_mask = disp["disp_norm"].isin(rpc_family)
        q2 = disp[rpc_mask][["borrower_id", "occurred_at"]].copy()
        q2["cp_tuple"] = list(map(tuple, disp.loc[rpc_mask, CP_KEYS].values.tolist()))
        qual_frames.append(q2[["borrower_id", "cp_tuple", "occurred_at"]])
    confirmed = pd.Series(False, index=uidx)
    days_conf = pd.Series(pd.NaT, index=uidx, dtype="datetime64[ns, UTC]")
    if qual_frames and not pay.empty:
        qual = pd.concat(qual_frames, ignore_index=True)
        # only borrowers having both qualifying events and payments
        common = set(qual["borrower_id"].unique()) & set(pay["borrower_id"].unique())
        for b in common:
            qt = qual[qual["borrower_id"] == b].sort_values("occurred_at")
            pt = pay[pay["borrower_id"] == b][["occurred_at"]].sort_values("occurred_at")
            qtimes = qt["occurred_at"].values.astype("datetime64[ns]")
            ptimes = pt["occurred_at"].values.astype("datetime64[ns]")
            pos = np.searchsorted(qtimes, ptimes, side="right") - 1
            ok = pos >= 0
            dt_days = np.full(len(ptimes), np.inf)
            dt_days[ok] = (ptimes[ok] - qtimes[pos[ok]]).astype("timedelta64[D]").astype(float)
            good = dt_days <= confirm_days
            if good.any():
                good_pt = pt.iloc[np.flatnonzero(good)]
                # attribute to every cp of this borrower whose qual time qualifies
                for cp_t in qt["cp_tuple"].unique():
                    ct = qt[qt["cp_tuple"] == cp_t]["occurred_at"].values.astype("datetime64[ns]")
                    pm = good_pt["occurred_at"].values.astype("datetime64[ns]")[:, None] - ct[None, :]
                    pm_days = pm.astype("timedelta64[D]").astype(float)
                    hit = (pm_days >= 0) & (pm_days <= confirm_days)
                    if hit.any():
                        key = tuple(cp_t)
                        if key in confirmed.index:
                            confirmed.loc[key] = True
                            best = as_utc(good_pt["occurred_at"].max())
                            if pd.isna(days_conf.loc[key]) or best > days_conf.loc[key]:
                                days_conf.loc[key] = best
    out["confirmed_by_payment"] = confirmed.astype("boolean")
    out["days_since_confirmed"] = _days_since(as_of, days_conf)

    # ---- cross-line ----------------------------------------------------------------------
    bor_keys = universe["borrower_id"]
    if not dial.empty:
        for w in windows:
            sfx = window_suffix(w)
            start = as_of - pd.Timedelta(days=w)
            dw = dial[(dial["occurred_at"] > start) & (dial["occurred_at"] <= as_of)]
            b_att = dw.groupby("borrower_id", sort=False).size()
            b_ans = dw[dw["answered"]].groupby("borrower_id", sort=False).size()
            own_att = dw.groupby(CP_KEYS, sort=False).size().reindex(uidx).fillna(0)
            own_ans = dw[dw["answered"]].groupby(CP_KEYS, sort=False).size().reindex(uidx).fillna(0)
            o_att = (bor_keys.map(b_att).fillna(0) - own_att.values).clip(lower=0).astype("Int64")
            o_ans = (bor_keys.map(b_ans).fillna(0) - own_ans.values).clip(lower=0).astype("Int64")
            # exclude non-phone rows' own counts already phone-only? dial events only
            # exist for phones; address rows: other = all borrower phone attempts.
            out[f"other_lines_attempts{sfx}"] = pd.Series(o_att.values, index=uidx, dtype="Int64")
            out[f"other_lines_answered{sfx}"] = pd.Series(o_ans.values, index=uidx, dtype="Int64")
            out[f"other_lines_answer_rate{sfx}"] = _safe_div(
                pd.Series(o_ans.values, index=uidx, dtype="Float64"),
                pd.Series(o_att.values, index=uidx, dtype="Float64"),
            ).where(np.asarray(o_att.fillna(0) > 0))
    else:
        for w in windows:
            sfx = window_suffix(w)
            out[f"other_lines_attempts{sfx}"] = pd.Series(0, index=uidx, dtype="Int64")
            out[f"other_lines_answered{sfx}"] = pd.Series(0, index=uidx, dtype="Int64")
            out[f"other_lines_answer_rate{sfx}"] = pd.Series(np.nan, index=uidx, dtype="Float64")
    if not pay.empty:
        for w in windows:
            sfx = window_suffix(w)
            start = as_of - pd.Timedelta(days=w)
            pw = pay[(pay["occurred_at"] > start) & (pay["occurred_at"] <= as_of)]
            out[f"n_payments{sfx}"] = bor_keys.map(
                pw.groupby("borrower_id", sort=False).size()).fillna(0).astype("Int64").values
        last_pay = pay.groupby("borrower_id", sort=False)["occurred_at"].last()
        out["days_since_last_payment"] = _days_since(as_of, bor_keys.map(last_pay))
    else:
        for w in windows:
            out[f"n_payments{window_suffix(w)}"] = pd.Series(0, index=uidx, dtype="Int64")
        out["days_since_last_payment"] = pd.Series(None, index=uidx, dtype="Int64")
    # days since last other-line answer: per-borrower top-2 trick
    if not dial.empty and dial["answered"].any():
        ans = dial[dial["answered"]].groupby(CP_KEYS, sort=False)["occurred_at"].max().reset_index()
        ans["borrower_id"] = ans["borrower_id"].astype(str)
        ans = ans.sort_values(["borrower_id", "occurred_at"], ascending=[True, False], kind="mergesort")
        top2 = ans.groupby("borrower_id", sort=False).head(2)
        first = top2.groupby("borrower_id", sort=False).first()
        second = top2.groupby("borrower_id", sort=False).nth(1)
        other_t = []
        for key, b in zip(uidx, bor_keys.tolist()):
            if b not in first.index:
                other_t.append(pd.NaT)
                continue
            top_cp = (first.loc[b, "lender_id"], b, first.loc[b, "contact_point_ref"])
            if tuple(key) == tuple(top_cp) and b in second.index:
                other_t.append(second.loc[b, "occurred_at"])
            elif tuple(key) == tuple(top_cp):
                other_t.append(pd.NaT)
            else:
                other_t.append(first.loc[b, "occurred_at"])
        out["days_since_last_other_line_answer"] = _days_since(
            as_of, pd.Series(pd.to_datetime(other_t, utc=True), index=uidx))
    else:
        out["days_since_last_other_line_answer"] = pd.Series(None, index=uidx, dtype="Int64")

    # ---- field visits (null for phone rows) ------------------------------------------------
    fv = events[events["event_type"] == "field_visit"].copy() if not events.empty else events.copy()
    if not fv.empty:
        fv["occurred_at"] = pd.to_datetime(fv["occurred_at"], utc=True)
        vt = pd.to_datetime(fv["visit_time"], utc=True, errors="coerce").fillna(fv["occurred_at"])
        fv["visit_hour_ist"] = vt.dt.tz_convert(ist).dt.hour + \
            vt.dt.tz_convert(ist).dt.minute / 60.0
        fv = fv.sort_values([*CP_KEYS, "occurred_at", "received_at"], kind="mergesort")
    out["n_visits"] = (fv.groupby(CP_KEYS, sort=False).size().reindex(uidx).fillna(0).astype("Int64")
                       if not fv.empty else pd.Series(0, index=uidx, dtype="Int64"))
    for oc in fcfg["visit_outcomes"]:
        col = f"n_visits_{oc}"
        out[col] = (fv[fv["outcome"] == oc].groupby(CP_KEYS, sort=False).size()
                    .reindex(uidx).fillna(0).astype("Int64")
                    if not fv.empty else pd.Series(0, index=uidx, dtype="Int64"))
    if not fv.empty:
        out["last_visit_outcome"] = fv.drop_duplicates(
            subset=CP_KEYS, keep="last").set_index(CP_KEYS)["outcome"].reindex(uidx).astype("string")
        out["days_since_last_visit"] = _days_since(
            as_of, fv.groupby(CP_KEYS, sort=False)["occurred_at"].last().reindex(uidx))
        dwell = pd.to_numeric(fv["dwell_seconds"], errors="coerce")
        out["gps_dwell_mean_seconds"] = dwell.groupby(
            pd.MultiIndex.from_arrays([fv[k].tolist() for k in CP_KEYS],
                                      names=CP_KEYS), sort=False).mean().reindex(uidx).astype("Float64")
        out["visit_hour_mean"] = fv["visit_hour_ist"].groupby(
            pd.MultiIndex.from_arrays([fv[k].tolist() for k in CP_KEYS],
                                      names=CP_KEYS), sort=False).mean().reindex(uidx).astype("Float64")
    else:
        out["last_visit_outcome"] = pd.Series(None, index=uidx, dtype="string")
        out["days_since_last_visit"] = pd.Series(None, index=uidx, dtype="Int64")
        out["gps_dwell_mean_seconds"] = pd.Series(np.nan, index=uidx, dtype="Float64")
        out["visit_hour_mean"] = pd.Series(np.nan, index=uidx, dtype="Float64")
    phone_rows = out["contact_point_type"] == "phone"
    for col in (["n_visits", *[f"n_visits_{o}" for o in fcfg["visit_outcomes"]],
                 "last_visit_outcome", "days_since_last_visit",
                 "gps_dwell_mean_seconds", "visit_hour_mean"]):
        if out[col].dtype == "string":
            out.loc[phone_rows, col] = pd.Series(
                [None] * int(phone_rows.sum()), dtype="string").values
        elif str(out[col].dtype) == "Int64":
            out.loc[phone_rows, col] = pd.Series(
                [None] * int(phone_rows.sum()), dtype="Int64").values
        else:
            out.loc[phone_rows, col] = np.nan

    # ---- account context ----------------------------------------------------------------------
    # Generic passthroughs from the borrowers table (or the official
    # accounts.csv fallback columns per configs/features.yaml
    # account_passthroughs). No secured_flag: no secured column exists and the
    # portfolio->secured mapping is unknown. Snapshot fields are quarantined
    # upstream until their as-of is confirmed; the passthroughs below are the
    # sanctioned set.
    def _coerce_passthrough(values: pd.Series, dtype: str) -> pd.Series:
        if dtype == "boolean":
            lowered = values.astype("string").str.strip().str.lower()
            mapped = lowered.map({"true": True, "false": False, "1": True, "0": False})
            if pd.api.types.is_bool_dtype(values.dtype):
                mapped = mapped.fillna(values.astype("boolean"))
            return mapped.astype("boolean")
        if dtype in ("Int64", "Float64"):
            return pd.to_numeric(values, errors="coerce").astype(dtype)
        return values.astype("string")

    if not borrowers.empty:
        bor = borrowers.drop_duplicates(subset="borrower_id", keep="first").set_index("borrower_id")
        bor_keys = universe["borrower_id"]
        for spec_entry in fcfg.get("account_passthroughs", []):
            col = next((c for c in spec_entry["columns"] if c in bor.columns), None)
            vals = bor[col].reindex(bor_keys.values).values if col is not None \
                else [None] * len(uidx)
            out[spec_entry["feature"]] = _coerce_passthrough(
                pd.Series(vals, index=uidx), str(spec_entry["dtype"]))
    else:
        for spec_entry in fcfg.get("account_passthroughs", []):
            out[spec_entry["feature"]] = _coerce_passthrough(
                pd.Series([None] * len(uidx), index=uidx), str(spec_entry["dtype"]))

    # ---- calendar ----------------------------------------------------------------------------------
    asof_ist = as_of.tz_convert(ist)
    out["asof_weekday"] = pd.Series(asof_ist.weekday(), index=uidx, dtype="Int64")
    out["asof_day_of_month"] = pd.Series(asof_ist.day, index=uidx, dtype="Int64")
    out["is_holiday"] = pd.Series(asof_ist.date().isoformat() in holidays,
                                  index=uidx, dtype="boolean")

    # ---- keys, dtypes, metadata -----------------------------------------------------------------------
    result = pd.DataFrame(index=uidx)
    result["lender_id"] = uidx.get_level_values("lender_id").astype("string")
    result["borrower_id"] = uidx.get_level_values("borrower_id").astype("string")
    result["account_id"] = pd.Series(account_ids.values, index=uidx, dtype="string")
    result["contact_point_ref"] = uidx.get_level_values("contact_point_ref").astype("string")
    result["as_of"] = pd.Series(as_of, index=uidx, dtype="datetime64[ns, UTC]")

    registry = build_registry(config, include_agent_feature=include_agent)
    for f in registry:
        col = out[f.name] if f.name in out.columns else pd.Series(np.nan, index=uidx)
        try:
            if f.dtype == "Int64":
                col = pd.to_numeric(col, errors="coerce").astype("Int64")
            elif f.dtype == "Float64":
                col = pd.to_numeric(col, errors="coerce").astype("Float64")
            elif f.dtype == "boolean":
                col = col.astype("boolean")
            elif f.dtype == "string":
                col = col.astype("string")
        except (TypeError, ValueError):
            col = pd.Series(np.nan, index=uidx).astype(f.dtype)
        result[f.name] = col
    result["feature_snapshot_id"] = pd.Series(snap, index=result.index, dtype="string")
    result["event_watermark"] = pd.Series(
        pd.Timestamp(watermark_str, tz="UTC"), index=result.index, dtype="datetime64[ns, UTC]")
    result = result.reset_index(drop=True)
    result = result.sort_values(
        ["lender_id", "borrower_id", "account_id", "contact_point_ref"], kind="mergesort"
    ).reset_index(drop=True)
    elapsed = time.time() - t0
    result.attrs["build_seconds"] = elapsed
    result.attrs["include_agent_feature"] = include_agent
    return result


def _empty_output(
    as_of: pd.Timestamp,
    watermark: Any,
    snap: str,
    config: dict,
    include_agent: bool,
) -> pd.DataFrame:
    registry = build_registry(config, include_agent_feature=include_agent)
    result = pd.DataFrame(columns=[*KEY_COLUMNS, *[f.name for f in registry], *META_COLUMNS])
    result["as_of"] = pd.Series(dtype="datetime64[ns, UTC]")
    result["event_watermark"] = pd.Series(dtype="datetime64[ns, UTC]")
    result.attrs["build_seconds"] = 0.0
    result.attrs["include_agent_feature"] = include_agent
    return result


def build_training_table(
    as_of_dates: list[str | pd.Timestamp],
    source: EventSource,
    contact_point_refs: list[str] | None = None,
) -> pd.DataFrame:
    """Stack ``build_features`` over several as_of dates (same code path)."""
    frames = [build_features(d, source, contact_point_refs) for d in as_of_dates]
    if not frames:
        ref = build_features(pd.Timestamp.now(tz="UTC"), source, [])
        return ref.iloc[0:0]
    return pd.concat(frames, ignore_index=True)
