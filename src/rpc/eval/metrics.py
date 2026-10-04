"""Metrics: discrimination, calibration, rare-event, decision, avoiding-vs-invalid.

All functions take plain pandas/numpy inputs so tests use tiny hand fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc as _sk_auc
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def _clean(y: np.ndarray, p: np.ndarray, clip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~(np.isnan(y) | np.isnan(p))
    if clip:
        return y[m], np.clip(p[m], 1e-9, 1 - 1e-9)
    return y[m], p[m]


def roc_auc(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    y, p = _clean(np.asarray(y), np.asarray(p))
    if len(np.unique(y)) < 2 or len(y) == 0:
        return float("nan")
    return float(roc_auc_score(y, p))


def pr_auc(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    y, p = _clean(np.asarray(y), np.asarray(p))
    if len(np.unique(y)) < 2 or len(y) == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def brier(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    y, p = _clean(np.asarray(y), np.asarray(p))
    if len(y) == 0:
        return float("nan")
    return float(brier_score_loss(y, np.clip(p, 0.0, 1.0)))


def logloss(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    y, p = _clean(np.asarray(y), np.asarray(p), clip=True)
    if len(y) == 0:
        return float("nan")
    return float(log_loss(y, p, labels=[0, 1]))


def reliability_table(
    y: np.ndarray | pd.Series, p: np.ndarray | pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Per-bin predicted mean vs observed rate + count (calibration curve data)."""
    y, p = _clean(np.asarray(y), np.asarray(p))
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1] + 1e-12)
        if i == 0:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        rows.append(
            {
                "bin": f"({edges[i]:.1f},{edges[i+1]:.1f}]",
                "n": int(m.sum()),
                "mean_pred": float(p[m].mean()) if m.any() else float("nan"),
                "obs_rate": float(y[m].mean()) if m.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def ece(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series, n_bins: int = 10) -> float:
    """Expected calibration error = count-weighted mean |obs - pred| over bins."""
    tab = reliability_table(y, p, n_bins).dropna()
    if tab["n"].sum() == 0:
        return float("nan")
    return float((tab["n"] * (tab["obs_rate"] - tab["mean_pred"]).abs()).sum() / tab["n"].sum())


def ece_by_segment(
    df: pd.DataFrame, y_col: str, p_col: str, segment_col: str, n_bins: int = 10
) -> pd.DataFrame:
    rows = []
    for seg, g in df.groupby(segment_col):
        rows.append({"segment": seg, "n": len(g), "ece": ece(g[y_col].to_numpy(), g[p_col].to_numpy(), n_bins)})
    return pd.DataFrame(rows)


# --- Rare-event (recycled) metrics ---

def pr_at_thresholds(
    y: np.ndarray | pd.Series, scores: np.ndarray | pd.Series, thresholds: list[float]
) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    s = np.asarray(scores, dtype=float)
    rows = []
    for t in thresholds:
        pred = s >= t
        tp = int(((pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        rows.append(
            {
                "threshold": t,
                "n_flagged": tp + fp,
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def cost_weighted_loss(
    y: np.ndarray | pd.Series, p: np.ndarray | pd.Series, cost_ratio: float
) -> float:
    """Mean loss with missed positives costing cost_ratio x a false alarm.

    cost_ratio = cost(missed recycled) / cost(false suppression), from guardrails.
    """
    y, p = _clean(np.asarray(y), np.asarray(p), clip=True)
    if len(y) == 0:
        return float("nan")
    w = np.where(y == 1, cost_ratio, 1.0)
    eps = 1e-9
    pc = np.clip(p, eps, 1 - eps)
    return float(-(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).mean())


# --- Decision metrics ---

def rpc_per_1000(y_ranked: np.ndarray | pd.Series, top_n: int = 1000) -> float:
    """RPCs per 1,000 dials if dialled in descending score order (ranked labels)."""
    y = np.asarray(y_ranked, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return float("nan")
    k = min(top_n, len(y))
    return float(y[:k].sum() / k * 1000.0)


def wasted_attempts(
    events: pd.DataFrame,
    oracle_dead_refs: set[str],
    action_times: pd.DataFrame | None = None,
    action_col: str = "first_action_at",
) -> pd.DataFrame:
    """Attempts on truly-dead lines before the policy first acted (non-continue).

    Without action_times: all attempts on dead lines count as wasted (upper bound,
    flagged via the ``note`` field by callers).
    """
    ev = events.copy()
    ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)
    dead = ev[ev["contact_point_ref"].isin(oracle_dead_refs)].copy()
    if action_times is not None and not action_times.empty:
        at = action_times.set_index("contact_point_ref")[action_col]
        dead["cutoff"] = dead["contact_point_ref"].map(at)
        dead = dead[dead["cutoff"].isna() | (dead["occurred_at"] < pd.to_datetime(dead["cutoff"], utc=True))]
    per_cp = dead.groupby("contact_point_ref").size().rename("wasted_attempts").reset_index()
    return per_cp


def detection_within_k(
    per_cp_wasted: pd.DataFrame,
    dead_refs: set[str],
    detected_refs: set[str],
) -> dict[int, float]:
    """Share of dead points detected (policy acted) -- k variants use wasted-attempt budgets.

    ``detected_refs`` = dead refs the policy acted on at all; the k-cut is
    approximated by wasted_attempts <= k (acted within k wasted dials).
    """
    out: dict[int, float] = {}
    w = dict(zip(per_cp_wasted["contact_point_ref"], per_cp_wasted["wasted_attempts"])) if not per_cp_wasted.empty else {}
    dead = [r for r in dead_refs]
    for k in (1, 3, 6):
        hit = sum(1 for r in dead if r in detected_refs and w.get(r, 0) <= k)
        out[k] = hit / len(dead) if dead else float("nan")
    return out


def coverage(
    accounts: pd.DataFrame,
    scores: pd.DataFrame,
    dead_refs: set[str],
    trace_refs: set[str] | None = None,
    p_threshold: float = 0.5,
) -> dict[str, float]:
    """Account coverage + orphaned share.

    Covered = account has >=1 non-dead cp with p_rpc >= threshold.
    Orphaned = no viable cp AND no trace queued for the account.
    """
    sc = scores.merge(accounts[["contact_point_ref", "account_id"]], on="contact_point_ref", how="left")
    sc["viable"] = (~sc["contact_point_ref"].isin(dead_refs)) & (sc["p_rpc"] >= p_threshold)
    by_acc = sc.groupby("account_id")["viable"].max()
    out = {"coverage": float(by_acc.mean()) if len(by_acc) else float("nan"), "n_accounts": int(len(by_acc))}
    if trace_refs is not None:
        traced_accs = set(accounts[accounts["contact_point_ref"].isin(trace_refs)]["account_id"])
        orphan = [a for a, v in by_acc.items() if (not v) and a not in traced_accs]
        out["orphaned_share"] = len(orphan) / len(by_acc) if len(by_acc) else float("nan")
    return out


# --- Avoiding vs invalid ---

def avoiding_vs_invalid(
    df: pd.DataFrame,
    true_state_col: str = "true_state",
    avoid_score_col: str = "avoiding",
    dead_states: tuple[str, ...] = ("recycled", "invalid", "switched_off_long"),
) -> dict[str, object]:
    """Among silent lines, separate truly-avoiding from truly-dead lines.

    Positive class = avoiding; negative = dead. Score = posterior mass on avoiding
    (vs dead mass renormalised if an ``invalid``-family column set is present).
    Returns confusion counts at 0.5, AUC, and n.
    """
    sub = df[df[true_state_col].isin(["avoiding", *dead_states])].copy()
    if sub.empty or avoid_score_col not in sub.columns:
        return {"n": 0, "auc": float("nan"), "note": "no silent-line subset or no posteriors"}
    sub["y_avoid"] = (sub[true_state_col] == "avoiding").astype(float)
    auc = roc_auc(sub["y_avoid"].to_numpy(), sub[avoid_score_col].to_numpy())
    pred = (sub[avoid_score_col] >= 0.5).astype(int)
    tn = int(((pred == 0) & (sub["y_avoid"] == 0)).sum())
    fp = int(((pred == 1) & (sub["y_avoid"] == 0)).sum())
    fn = int(((pred == 0) & (sub["y_avoid"] == 1)).sum())
    tp = int(((pred == 1) & (sub["y_avoid"] == 1)).sum())
    return {"n": int(len(sub)), "auc": float(auc), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def cross_line_subset(
    events: pd.DataFrame,
    oracle: pd.DataFrame,
    rpc_responses: tuple[str, ...] = ("answered",),
) -> pd.DataFrame:
    """Borrower has >=2 lines and an RPC observed on one line while another stayed silent.

    Returns the silent-side rows (borrower_id, silent_ref, rpc_ref) for the subset report.
    """
    import json as _json

    ev = events.copy()
    ev["occurred_at"] = pd.to_datetime(ev["occurred_at"], utc=True)

    def _nr(p: object) -> str:
        try:
            d = _json.loads(p) if isinstance(p, str) else (p or {})
            return str(d.get("network_response", ""))
        except Exception:
            return ""

    ev["nr"] = ev["payload"].map(_nr) if "payload" in ev.columns else ""
    rpc_refs = set(ev[ev["nr"].isin(set(rpc_responses))]["contact_point_ref"])
    acct = ev[["borrower_id", "contact_point_ref"]].drop_duplicates()
    n_lines = acct.groupby("borrower_id")["contact_point_ref"].nunique()
    multi = set(n_lines[n_lines >= 2].index)
    rows = []
    for b in multi:
        refs = set(acct[acct["borrower_id"] == b]["contact_point_ref"])
        hit = refs & rpc_refs
        silent = refs - rpc_refs
        for s in silent:
            if hit:
                rows.append({"borrower_id": b, "silent_ref": s, "rpc_ref": sorted(hit)[0]})
    return pd.DataFrame(rows)


# --- Uncertainty ---

def bootstrap_ci(
    fn: object,
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap CI for metric fn(y, p)."""
    assert callable(fn)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    point = float(fn(y, p)) if len(y) else float("nan")  # type: ignore[operator]
    if len(y) == 0:
        return (point, float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            v = float(fn(y[idx], p[idx]))  # type: ignore[operator]
        except Exception:
            v = float("nan")
        vals.append(v)
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (point, float("nan"), float("nan"))
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return (point, float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q)))
