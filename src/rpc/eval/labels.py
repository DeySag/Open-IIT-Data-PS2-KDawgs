"""Sanctioned RPC labels with censoring (P2).

Per-attempt RPC = ``answered AND disposition LIKE 'rpc_*'`` over the
sanctioned 7-variant set (coordinator sign-off 2026-10-07, docs/decision_log.md).
Raw ``answered`` alone is NOT a label (37% non-RPC inside it on the issued
extracts). ``language_barrier`` stays ambiguous (excluded, never RPC).

Grain is (account, phone): ``account_id`` rides on every label row because
20 phone_ids were dialled under >1 account with different outcomes, so a
per-``phone_id`` label without account context mixes borrowers. Undialled
links/addresses are censored (NaN, never negative).

Two input modes (no frozen-contract change):
- raw extracts: ``dial_attempts`` rows with ``network_response`` + ``disposition``
  columns — full fidelity; strict variant exact.
- canonical events: ``dial_attempt`` + ``disposition`` companion rows paired by
  (account_id, contact_point_ref, occurred_at) — same attempt_ts yields both
  rows at ingest. Canonical ``RPC`` collapses refused/hung_up/hardship/
  claims_paid, so canonical-strict = {promise_to_pay, callback, dispute}
  (documented limitation); raw-strict drops only hung_up/refused.

Verified rows (250) are holdout gold only — never label sources for training,
and even membership is leakage. Payment-anchored labels are weak supervision
only; post-cutoff payments never features. Trace results/new IDs are VOI-only,
never predictors.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Sequence

import pandas as pd

# --- Sanctioned sets (raw names as in dial_attempts.csv) ---------------------

RAW_RPC_DISPOSITIONS: frozenset[str] = frozenset(
    {
        "rpc_ptp",
        "rpc_call_back",
        "rpc_hung_up",
        "rpc_refused",
        "rpc_hardship",
        "rpc_dispute",
        "rpc_claims_paid",
    }
)

# Raw strict sensitivity drops only contact-without-content variants.
RAW_STRICT_EXCLUDE: frozenset[str] = frozenset({"rpc_hung_up", "rpc_refused"})

# Canonical names as emitted by configs/field_mappings/cn_dial_dispositions.yaml.
CANONICAL_RPC_DISPOSITIONS: frozenset[str] = frozenset(
    {"RPC", "promise_to_pay", "callback", "dispute"}
)

# Canonical-strict: "RPC" collapses refused/hung_up/hardship/claims_paid, so the
# canonical envelope cannot isolate hung_up/refused. Excluding "RPC" is the
# conservative canonical analogue (also drops hardship/claims_paid, n=372 —
# documented in docs/decision_log.md).
CANONICAL_STRICT_RPC: frozenset[str] = frozenset(
    {"promise_to_pay", "callback", "dispute"}
)

ANSWERED: frozenset[str] = frozenset({"answered"})

# Ambiguous: contact-without-content, never RPC (audit §5).
AMBIGUOUS_DISPOSITIONS: frozenset[str] = frozenset({"language_barrier"})

# Raw disposition -> canonical (mirrors cn_dial_dispositions.yaml map).
RAW_TO_CANONICAL: dict[str, str] = {
    "rpc_ptp": "promise_to_pay",
    "rpc_call_back": "callback",
    "rpc_dispute": "dispute",
    "rpc_refused": "RPC",
    "rpc_hung_up": "RPC",
    "rpc_hardship": "RPC",
    "rpc_claims_paid": "RPC",
}


def _payload_field(payload: object, key: str, default: object = None) -> object:
    try:
        d = json.loads(payload) if isinstance(payload, str) else (payload or {})
        return d.get(key, default)
    except Exception:
        return default


def _is_raw_dials_frame(df: pd.DataFrame) -> bool:
    """Raw dial_attempts.csv rows carry both columns; canonical events do not."""
    return "network_response" in df.columns and "disposition" in df.columns


def attempt_is_rpc_raw(
    network_response: object, disposition: object, *, strict: bool = False
) -> bool:
    """Per-attempt RPC on raw extract values: answered AND sanctioned rpc_*."""
    if str(network_response) not in ANSWERED:
        return False  # quarantines the 24 non-answered rpc_ptp rows
    disp = str(disposition)
    if disp in AMBIGUOUS_DISPOSITIONS:
        return False
    allowed = RAW_RPC_DISPOSITIONS - RAW_STRICT_EXCLUDE if strict else RAW_RPC_DISPOSITIONS
    return disp in allowed


def attempt_is_rpc_canonical(
    network_response: object, disposition: object, *, strict: bool = False
) -> bool:
    """Per-attempt RPC on canonical envelope values (paired dial+disposition)."""
    if str(network_response) not in ANSWERED:
        return False
    disp = str(disposition)
    if disp in AMBIGUOUS_DISPOSITIONS:
        return False
    # Accept raw names too (defensive: some pipelines pass raw through).
    if disp in RAW_RPC_DISPOSITIONS:
        allowed_raw = RAW_RPC_DISPOSITIONS - RAW_STRICT_EXCLUDE if strict else RAW_RPC_DISPOSITIONS
        return disp in allowed_raw
    allowed = CANONICAL_STRICT_RPC if strict else CANONICAL_RPC_DISPOSITIONS
    return disp in allowed


def attempt_table_from_raw(
    dials: pd.DataFrame, *, strict: bool = False, strict_col: bool = True
) -> pd.DataFrame:
    """One row per raw attempt with AND-joined RPC flags.

    Expects ``account_id``, ``phone_id``, ``attempt_ts`` columns. ``is_rpc``
    is the sanctioned primary; ``is_rpc_strict`` the sensitivity.
    """
    df = dials.copy()
    df["occurred_at"] = pd.to_datetime(df["attempt_ts"], utc=True)
    df["is_rpc"] = [
        attempt_is_rpc_raw(nr, d, strict=False)
        for nr, d in zip(df["network_response"], df["disposition"])
    ]
    if strict_col or strict:
        df["is_rpc_strict"] = [
            attempt_is_rpc_raw(nr, d, strict=True)
            for nr, d in zip(df["network_response"], df["disposition"])
        ]
    # 24 pattern-breaking rows: rpc_* on non-answered networks.
    df["quarantined"] = df["disposition"].isin(RAW_RPC_DISPOSITIONS) & ~df[
        "network_response"
    ].isin(ANSWERED)
    return df


def attempt_table_from_events(events: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct per-attempt rows from canonical dial+disposition companions.

    Pairs by (account_id, contact_point_ref, occurred_at): one source row
    yields both companions with identical keys. ``is_rpc`` requires an
    answered dial AND a sanctioned disposition in the same group — never OR.
    Groups with an rpc-like disposition but no answered dial are quarantined.
    """
    ev = events.copy()
    ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)
    if ev.empty:
        return pd.DataFrame(
            columns=[
                "account_id",
                "contact_point_ref",
                "occurred_at",
                "n_events",
                "has_dial",
                "answered",
                "disposition",
                "is_rpc",
                "is_rpc_strict",
                "quarantined",
            ]
        )

    def _nr(r: pd.Series) -> str | None:
        if r["event_type"] == "dial_attempt":
            return str(_payload_field(r.get("payload"), "network_response", ""))
        return None

    def _disp(r: pd.Series) -> str | None:
        if r["event_type"] == "disposition":
            return str(_payload_field(r.get("payload"), "disposition", ""))
        return None

    ev["_nr"] = ev.apply(_nr, axis=1)
    ev["_disp"] = ev.apply(_disp, axis=1)
    group_keys = ["account_id", "contact_point_ref", "occurred_at"]
    for k in group_keys:
        if k not in ev.columns:
            ev[k] = "UNK"

    rows: list[dict] = []
    for keys, g in ev.groupby(group_keys, dropna=False):
        account_id, ref, ts = keys
        nrs = [v for v in g["_nr"].tolist() if v is not None]
        disps = [v for v in g["_disp"].tolist() if v is not None]
        answered = any(v in ANSWERED for v in nrs)
        has_dial = len(nrs) > 0
        disp = disps[0] if disps else None
        is_rpc = bool(
            answered
            and disp is not None
            and attempt_is_rpc_canonical("answered", disp, strict=False)
        )
        is_strict = bool(
            answered
            and disp is not None
            and attempt_is_rpc_canonical("answered", disp, strict=True)
        )
        rpc_like = disp in CANONICAL_RPC_DISPOSITIONS or disp in RAW_RPC_DISPOSITIONS
        rows.append(
            {
                "account_id": account_id,
                "contact_point_ref": ref,
                "occurred_at": ts,
                "n_events": len(g),
                "has_dial": has_dial,
                "answered": answered,
                "disposition": disp,
                "is_rpc": is_rpc,
                "is_rpc_strict": is_strict,
                "quarantined": bool(rpc_like and not answered),
            }
        )
    return pd.DataFrame(rows)


def _normalise_keys(
    keys: pd.DataFrame | Sequence[str] | Sequence[tuple[str, str]] | None,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Return DataFrame[account_id, contact_point_ref] for the label universe.

    Plain refs (legacy callers) resolve account_id via per-ref mode in events
    with an explicit ``account_resolved`` flag — shared ids dialled under >1
    account resolve to their mode and are flagged via ``account_ambiguous`` so
    callers can enforce per-(account, phone) attribution instead.
    """
    if isinstance(keys, pd.DataFrame) and {
        "account_id",
        "contact_point_ref",
    } <= set(keys.columns):
        out = keys[["account_id", "contact_point_ref"]].copy()
        out["account_resolved"] = False
        out["account_ambiguous"] = False
        return out.drop_duplicates().reset_index(drop=True)
    refs: list[str]
    if keys is None:
        refs = sorted(events["contact_point_ref"].dropna().unique().tolist())
    elif len(keys) > 0 and isinstance(keys[0], (tuple, list)):
        pairs = [(str(a), str(r)) for a, r in keys]  # type: ignore[misc]
        out = pd.DataFrame(pairs, columns=["account_id", "contact_point_ref"])
        out["account_resolved"] = False
        out["account_ambiguous"] = False
        return out.drop_duplicates().reset_index(drop=True)
    else:
        refs = [str(r) for r in keys]  # type: ignore[union-attr]
    base = pd.DataFrame({"contact_point_ref": refs})
    if "account_id" in events.columns and not events.empty:
        mode = (
            events.dropna(subset=["account_id"])
            .groupby("contact_point_ref")["account_id"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "UNK")
        )
        nunq = events.groupby("contact_point_ref")["account_id"].nunique()
        base["account_id"] = base["contact_point_ref"].map(mode).fillna("UNK")
        base["account_ambiguous"] = (
            base["contact_point_ref"].map(nunq).fillna(0).astype(int) > 1
        )
    else:
        base["account_id"] = "UNK"
        base["account_ambiguous"] = False
    base["account_resolved"] = True
    return base[["account_id", "contact_point_ref", "account_resolved", "account_ambiguous"]]


def observed_labels(
    events: pd.DataFrame,
    as_of: datetime,
    contact_point_refs: Sequence[str] | pd.DataFrame | Sequence[tuple[str, str]] | None = None,
    horizon_days: int = 7,
    rpc_responses: Sequence[str] = ("answered",),
    rpc_dispositions: Sequence[str] = ("RPC", "promise_to_pay", "callback", "dispute"),
    *,
    keys: pd.DataFrame | Sequence[tuple[str, str]] | None = None,
    strict: bool = False,
    verified_keys: pd.DataFrame | set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Label each (account, phone) key: RPC observed in (as_of, as_of+horizon].

    ``rpc_next_7d`` is NaN when censored (no dial in window); ``censored``
    flags it. Undialled links are censored, never negative. ``rpc_next_7d_strict``
    carries the minus-hung_up/refused sensitivity. ``verified_holdout`` flags
    keys in the verified set (eval-only; training must drop them — even
    membership is leakage).

    ``rpc_responses``/``rpc_dispositions`` overrides are honoured only when they
    narrow the sanctioned set; the AND-join and ambiguous exclusions always apply.
    """
    _ = rpc_responses  # answered-AND is non-negotiable per sign-off; kept for compat
    universe = _normalise_keys(
        keys if keys is not None else contact_point_refs, events
    )
    as_of_ts = pd.to_datetime(as_of, utc=True)
    end_ts = as_of_ts + pd.Timedelta(days=horizon_days)

    if _is_raw_dials_frame(events):
        at = attempt_table_from_raw(events)
        win = at[(at["occurred_at"] > as_of_ts) & (at["occurred_at"] <= end_ts)].copy()
        if win.empty:
            return _empty_labels(universe, horizon_days)
        win["contact_point_ref"] = win["phone_id"].astype(str)
        val_col = "is_rpc_strict" if strict else "is_rpc"
        g = win.groupby(["account_id", "contact_point_ref"]).agg(
            n_dials_window=("attempt_ts", "size"),
            _rpc=(val_col, "max"),
            _strict=("is_rpc_strict", "max"),
        )
    else:
        at = attempt_table_from_events(events)
        win = at[(at["occurred_at"] > as_of_ts) & (at["occurred_at"] <= end_ts)].copy()
        if win.empty:
            return _empty_labels(universe, horizon_days)
        # Custom disposition allowlist: intersect with sanctioned (narrow-only).
        if set(rpc_dispositions) != set(CANONICAL_RPC_DISPOSITIONS):
            allow = set(rpc_dispositions) & (
                set(CANONICAL_RPC_DISPOSITIONS) | set(RAW_RPC_DISPOSITIONS)
            )
            win["is_rpc"] = win.apply(
                lambda r: bool(
                    r["answered"]
                    and r["disposition"] in allow
                    and str(r["disposition"]) not in AMBIGUOUS_DISPOSITIONS
                ),
                axis=1,
            )
            if strict:
                allow_s = allow & (set(CANONICAL_STRICT_RPC) | (RAW_RPC_DISPOSITIONS - RAW_STRICT_EXCLUDE))
                win["is_rpc_strict"] = win.apply(
                    lambda r: bool(r["answered"] and r["disposition"] in allow_s),
                    axis=1,
                )
        val_col = "is_rpc_strict" if strict else "is_rpc"
        g = win.groupby(["account_id", "contact_point_ref"]).agg(
            n_dials_window=("is_rpc", "size"),
            _rpc=(val_col, "max"),
            _strict=("is_rpc_strict", "max"),
        )

    out = universe.merge(g, on=["account_id", "contact_point_ref"], how="left")
    out["n_dials_window"] = out["n_dials_window"].fillna(0).astype(int)
    dialled = out["n_dials_window"] > 0
    out["censored"] = ~dialled
    label_col = f"rpc_next_{horizon_days}d"
    out[label_col] = _boolmax_to_float(out["_rpc"])
    out.loc[~dialled, label_col] = float("nan")
    out[f"{label_col}_strict"] = _boolmax_to_float(out["_strict"])
    out.loc[~dialled, f"{label_col}_strict"] = float("nan")
    # Legacy alias (harness expects rpc_next_7d when horizon=7).
    if label_col != "rpc_next_7d":
        out["rpc_next_7d"] = out[label_col]
    else:
        out["rpc_next_7d"] = out[label_col]
    out = _apply_verified_holdout(out, verified_keys)
    return out.drop(columns=["_rpc", "_strict"])


def _empty_labels(universe: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    out = universe.copy()
    label_col = f"rpc_next_{horizon_days}d"
    out["n_dials_window"] = 0
    out["censored"] = True
    out[label_col] = float("nan")
    out[f"{label_col}_strict"] = float("nan")
    out["rpc_next_7d"] = float("nan")
    out["verified_holdout"] = False
    return out


def _boolmax_to_float(s: pd.Series) -> pd.Series:
    return s.map(lambda v: float("nan") if pd.isna(v) else float(bool(v)))


def _apply_verified_holdout(
    labels: pd.DataFrame, verified_keys: pd.DataFrame | set[tuple[str, str]] | None
) -> pd.DataFrame:
    out = labels.copy()
    out["verified_holdout"] = False
    if verified_keys is None:
        return out
    if isinstance(verified_keys, pd.DataFrame):
        vset = set(
            zip(
                verified_keys["account_id"].astype(str),
                verified_keys["contact_point_ref"].astype(str),
            )
        )
    else:
        vset = {(str(a), str(r)) for a, r in verified_keys}
    mask = list(
        zip(
            out["account_id"].astype(str),
            out["contact_point_ref"].astype(str),
        )
    )
    out["verified_holdout"] = [m in vset for m in mask]
    return out


def train_labels_only(
    labels: pd.DataFrame,
    *,
    drop_verified: bool = True,
    dialled_only: bool = True,
) -> pd.DataFrame:
    """Enforce the DATA RULE: training never sees verified holdout or censored rows.

    Verified keys are dropped (not just masked — membership itself is leakage)
    and censored (undialled) rows are dropped, never treated as negatives.
    """
    out = labels.copy()
    if drop_verified and "verified_holdout" in out.columns:
        out = out[~out["verified_holdout"]].copy()
    if dialled_only and "censored" in out.columns:
        out = out[~out["censored"]].copy()
    return out.reset_index(drop=True)


def ever_rpc_labels(
    events: pd.DataFrame,
    as_of: datetime,
    keys: pd.DataFrame | Sequence[str] | Sequence[tuple[str, str]] | None = None,
    *,
    verified_keys: pd.DataFrame | set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Per-(account,phone) ever-RPC over history with occurred_at <= as_of."""
    universe = _normalise_keys(keys, events)
    as_of_ts = pd.to_datetime(as_of, utc=True)
    if _is_raw_dials_frame(events):
        at = attempt_table_from_raw(events)
        at["contact_point_ref"] = at["phone_id"].astype(str)
        hist = at[at["occurred_at"] <= as_of_ts]
        g = hist.groupby(["account_id", "contact_point_ref"]).agg(
            n_dials_hist=("attempt_ts", "size"),
            ever_rpc=("is_rpc", "max"),
            ever_rpc_strict=("is_rpc_strict", "max"),
        )
    else:
        at = attempt_table_from_events(events)
        hist = at[at["occurred_at"] <= as_of_ts]
        if hist.empty:
            out = universe.copy()
            out["n_dials_hist"] = 0
            out["ever_rpc"] = float("nan")
            out["ever_rpc_strict"] = float("nan")
            out["censored"] = True
            return _apply_verified_holdout(out, verified_keys)
        g = hist.groupby(["account_id", "contact_point_ref"]).agg(
            n_dials_hist=("is_rpc", "size"),
            ever_rpc=("is_rpc", "max"),
            ever_rpc_strict=("is_rpc_strict", "max"),
        )
    out = universe.merge(g, on=["account_id", "contact_point_ref"], how="left")
    out["n_dials_hist"] = out["n_dials_hist"].fillna(0).astype(int)
    dialled = out["n_dials_hist"] > 0
    out["censored"] = ~dialled
    out["ever_rpc"] = _boolmax_to_float(out["ever_rpc"])
    out.loc[~dialled, "ever_rpc"] = float("nan")
    out["ever_rpc_strict"] = _boolmax_to_float(out["ever_rpc_strict"])
    out.loc[~dialled, "ever_rpc_strict"] = float("nan")
    return _apply_verified_holdout(out, verified_keys)


def payment_weak_labels(
    payments: pd.DataFrame,
    dials_or_attempts: pd.DataFrame,
    as_of: datetime,
    keys: pd.DataFrame | Sequence[tuple[str, str]] | None = None,
    *,
    confirmation_days: int = 7,
) -> pd.DataFrame:
    """Weak-supervision payment anchor: NEVER a primary label, NEVER a feature.

    Flags (account, phone) keys whose account paid within ``confirmation_days``
    after an RPC attempt at-or-before ``as_of``. Post-``as_of`` payments are
    ignored (they would leak). The pay/RPC mismatch on issued extracts breaks
    any equivalence — callers must treat this as weak supervision only.
    """
    as_of_ts = pd.to_datetime(as_of, utc=True)
    if isinstance(keys, pd.DataFrame):
        universe = keys[["account_id", "contact_point_ref"]].copy().drop_duplicates()
    elif keys is not None:
        universe = pd.DataFrame(
            [(str(a), str(r)) for a, r in keys],
            columns=["account_id", "contact_point_ref"],
        )
    else:
        accts = payments["account_id"].astype(str).unique().tolist()
        universe = pd.DataFrame(
            {"account_id": accts, "contact_point_ref": "__account__"}
        )
    pay = payments.copy()
    pay["payment_ts"] = pd.to_datetime(pay["payment_ts"], utc=True)
    pay = pay[pay["payment_ts"] <= as_of_ts]  # post-cutoff payments never features
    if _is_raw_dials_frame(dials_or_attempts):
        at = attempt_table_from_raw(dials_or_attempts)
        rpc = at[at["is_rpc"] & (at["occurred_at"] <= as_of_ts)][
            ["account_id", "occurred_at"]
        ]
    else:
        at = attempt_table_from_events(dials_or_attempts)
        rpc = at[at["is_rpc"] & (at["occurred_at"] <= as_of_ts)][
            ["account_id", "occurred_at"]
        ]
    out = universe.copy()
    out["pay_weak"] = False
    if rpc.empty or pay.empty:
        return out
    for idx, row in out.iterrows():
        acct = str(row["account_id"])
        rpc_ts = rpc.loc[rpc["account_id"].astype(str) == acct, "occurred_at"]
        pay_ts = pay.loc[pay["account_id"].astype(str) == acct, "payment_ts"]
        hit = False
        for rt in rpc_ts:
            delta = (pay_ts - rt).dt.total_seconds() / 86400.0
            if ((delta >= 0) & (delta <= confirmation_days)).any():
                hit = True
                break
        out.at[idx, "pay_weak"] = hit
    return out


def load_verified_keys(path: str, ref_col: str = "contact_point_ref") -> pd.DataFrame:
    """Load verified holdout keys as (account_id, contact_point_ref).

    Maps ``phone_id`` -> ``contact_point_ref`` (same string when labels are
    built on raw extracts; hash-matched by callers on canonical stores).
    """
    v = pd.read_csv(path, dtype="string")
    phone_col = "phone_id" if "phone_id" in v.columns else ref_col
    out = pd.DataFrame(
        {
            "account_id": v["account_id"].astype(str),
            "contact_point_ref": v[phone_col].astype(str),
        }
    )
    return out.drop_duplicates().reset_index(drop=True)
