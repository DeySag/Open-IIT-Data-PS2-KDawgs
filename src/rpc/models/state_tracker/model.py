"""Contact-point state tracker with a borrower-level avoidance latent.

Model (see docs/state_tracker.md for full justification):
- Per phone line k: hidden state S in {valid, temp_unreachable,
  switched_off_long, recycled, third_party, invalid}, daily transitions.
- Per borrower: hidden avoidance A in {0, 1} with daily transitions,
  optionally scaled by DPD bucket.
- Each line is filtered as a JOINT 12-state chain over (A, S) with factorised
  transitions P(a'|a) * P(s'|s) and emissions depending on (a, s).
- Cross-line sharing: borrower-level resets (rpc answers, payments) are
  applied to every sibling line's filter, and at scoring time the A marginals
  of sibling lines are pooled (naive-Bayes with prior correction) so that a
  silent line with an answering sibling is pushed toward dead while all-silent
  lines point to avoidance. With ``use_borrower_latent=False`` lines are
  scored independently (ablation).
- Reported 7-key posterior: avoiding = P(S=valid, A=1),
  valid_reachable = P(S=valid, A=0), other states marginalised over A.
- Parameters are learned by EM (forward-backward per line) on events only.
  No ground-truth labels are ever read here.

Point-in-time: every method filters events with received_at <= as_of.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.rpc.models.types import STATE_KEYS, ContactPointScore

# ---------------------------------------------------------------------------
# Index maps
# ---------------------------------------------------------------------------

S_NAMES = ["valid", "temp_unreachable", "switched_off_long", "recycled", "third_party", "invalid"]
S_IDX = {n: i for i, n in enumerate(S_NAMES)}
R_NAMES = ["answered", "no_answer", "busy", "switched_off", "not_reachable", "does_not_exist", "immediate_hangup"]
R_IDX = {n: i for i, n in enumerate(R_NAMES)}
D_NAMES = ["rpc", "wrong_number", "third_party", "switched_off", "not_reachable", "promise_to_pay", "dispute", "callback"]
D_IDX = {n: i for i, n in enumerate(D_NAMES)}

N_S, N_A, N_J = 6, 2, 12  # joint index j = a * 6 + s

# Observation kinds
K_NET, K_DISP, K_PAY, K_CUE = 0, 1, 2, 3

# 7-key report order
REPORT_S = ["valid", "temp_unreachable", "switched_off_long", "recycled", "third_party", "invalid"]

NET_ALIASES = {
    "not reachable": "not_reachable",
    "notreachable": "not_reachable",
    "switchedoff": "switched_off",
    "switch_off": "switched_off",
    "doesnt_exist": "does_not_exist",
    "doesnotexist": "does_not_exist",
    "does_not_exits": "does_not_exist",
    "hangup": "immediate_hangup",
    "immediate hangup": "immediate_hangup",
}

STRONG_RPC_DISP = {"rpc", "promise_to_pay"}  # trigger borrower-level resets


def _canon_response(v: str) -> str:
    v = str(v).strip().lower()
    if v in R_IDX:
        return v
    if v in NET_ALIASES:
        return NET_ALIASES[v]
    raise ValueError(f"unknown network_response {v!r}")


def _canon_disp(v: str) -> str:
    v = str(v).strip().lower()
    if v in D_IDX:
        return v
    raise ValueError(f"unknown disposition {v!r}")


def _ensure_tz(dt: Any) -> datetime:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Config / params
# ---------------------------------------------------------------------------


def load_config(path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path, dict):
        return path
    with open(path) as f:
        return yaml.safe_load(f)


def _matrix_from_nested(nested: dict[str, dict[str, float]], rows: list[str], cols: list[str]) -> np.ndarray:
    m = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            m[i, j] = nested[r][c]
    return m / m.sum(axis=1, keepdims=True)


def config_to_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Coarse priors from config -> numeric parameter dict (leakage-free)."""
    pi_s = np.array([cfg["initial_line_dist"][s] for s in S_NAMES], dtype=float)
    pi_s /= pi_s.sum()
    pi_a0 = float(cfg["initial_avoid_prob"])
    t_s = _matrix_from_nested(cfg["transition_line"], S_NAMES, S_NAMES)
    t_a = np.array(cfg["transition_avoid"], dtype=float)
    t_a /= t_a.sum(axis=1, keepdims=True)
    e_net = np.zeros((N_S, N_A, len(R_NAMES)))
    e_disp = np.zeros((N_S, N_A, len(D_NAMES)))
    for s in S_NAMES:
        for a_i, a in enumerate(["calm", "avoiding"]):
            row = cfg["emission_network"][s][a]
            e_net[S_IDX[s], a_i, :] = [row[r] for r in R_NAMES]
            drow = cfg["emission_disposition"][s][a]
            e_disp[S_IDX[s], a_i, :] = [drow[d] for d in D_NAMES]
    e_net /= e_net.sum(axis=2, keepdims=True)
    e_disp /= e_disp.sum(axis=2, keepdims=True)
    floor = float(cfg["em"].get("emission_floor", 1e-4))
    e_net = np.maximum(e_net, floor)
    e_disp = np.maximum(e_disp, floor)
    e_net /= e_net.sum(axis=2, keepdims=True)
    e_disp /= e_disp.sum(axis=2, keepdims=True)
    return {
        "pi_s": pi_s,
        "pi_a0": pi_a0,
        "T_S": t_s,
        "T_A": t_a,
        "E_net": e_net,
        "E_disp": e_disp,
        "e_pay": np.array([float(cfg["payment_emission"]["a0"]), float(cfg["payment_emission"]["a1"])]),
        "ans": float(cfg.get("answer_given_valid_calm", 0.35)),
    }


# ---------------------------------------------------------------------------
# Event parsing (slim frame)
# ---------------------------------------------------------------------------

_AVOID_CUES: tuple[str, ...] = ()
_WRONGNUM_CUES: tuple[str, ...] = ()


def _payload_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            d = json.loads(payload)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def parse_events(
    events_df: pd.DataFrame,
    avoid_cues: tuple[str, ...] = (),
    wrongnum_cues: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Flatten the canonical envelope into a slim observation frame.

    Returns columns: borrower_id, cp_ref, received_at, day (placeholder -1),
    kind, val, is_reset, dpd (if present), src (if present), ctype (if present).
    Never reads ground-truth columns even if present (they are dropped).
    """
    df = events_df.copy()
    drop = [c for c in df.columns if c.lower().startswith(("truth", "true_", "label", "ground_truth"))]
    df = df.drop(columns=drop, errors="ignore")
    req = {"borrower_id", "contact_point_ref", "event_type", "received_at"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"events_df missing columns {sorted(missing)}")
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    et = df["event_type"].astype(str).str.strip().str.lower()

    kinds = np.full(len(df), -1)
    vals = np.full(len(df), -1)
    is_reset = np.zeros(len(df), dtype=bool)
    src = df["source"] if "source" in df.columns else pd.Series(["KYC"] * len(df), index=df.index)
    ctype = df["cp_type"] if "cp_type" in df.columns else (
        df["type"] if "type" in df.columns else pd.Series(["phone"] * len(df), index=df.index)
    )
    dpd = df["dpd_bucket"] if "dpd_bucket" in df.columns else pd.Series([None] * len(df), index=df.index)

    for i, (_, row) in enumerate(df.iterrows()):
        t = et.iloc[i]
        p = _payload_dict(row["payload"] if "payload" in df.columns else {})
        if t == "dial_attempt":
            kinds[i] = K_NET
            vals[i] = R_IDX[_canon_response(p.get("network_response", "no_answer"))]
        elif t == "disposition":
            d = _canon_disp(p.get("disposition", "callback"))
            kinds[i] = K_DISP
            vals[i] = D_IDX[d]
            is_reset[i] = d in STRONG_RPC_DISP
        elif t == "payment":
            kinds[i] = K_PAY
            vals[i] = 0
            is_reset[i] = True
        elif t == "bot_transcript":
            text = str(p.get("transcript", "")).lower() + " " + " ".join(
                str(x).lower() for x in (p.get("extracted_phrases", []) or [])
            )
            has_avoid = any(c in text for c in avoid_cues)
            has_wn = any(c in text for c in wrongnum_cues)
            if has_avoid or has_wn:
                kinds[i] = K_CUE
                vals[i] = (1 if has_avoid else 0) + (2 if has_wn else 0)  # bitmask 1..3
        elif t in ("contact_point_update", "field_visit"):
            kinds[i] = -1  # no observation (source recorded separately)
            upd_src = p.get("source")
            if t == "contact_point_update" and upd_src:
                src.iloc[i] = upd_src
            ct = p.get("contact_type")
            if ct in ("phone", "address"):
                ctype.iloc[i] = ct
        else:
            kinds[i] = -1

    out = pd.DataFrame(
        {
            "borrower_id": df["borrower_id"].astype(str).values,
            "cp_ref": df["contact_point_ref"].astype(str).values,
            "received_at": df["received_at"].values,
            "kind": kinds,
            "val": vals,
            "is_reset": is_reset,
            "src": src.astype(str).values,
            "ctype": ctype.astype(str).values,
            "dpd": [None if (d is None or (isinstance(d, float) and math.isnan(d))) else str(d) for d in dpd.values],
        }
    )
    out = out.sort_values(["borrower_id", "received_at"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# StateTracker
# ---------------------------------------------------------------------------


class StateTracker:
    """HMM state tracker with borrower avoidance latent. See module docstring."""

    version = "state_tracker_v0.1.0"

    def __init__(self, config: str | Path | dict[str, Any] | None = None) -> None:
        cfg = load_config(config) if config is not None else {}
        self.cfg = cfg
        default_path = Path("configs/state_tracker.yaml")
        if not cfg and default_path.exists():
            with open(default_path) as f:
                self.cfg = yaml.safe_load(f)
        self.use_latent: bool = bool(self.cfg.get("use_borrower_latent", True))
        self.params: dict[str, Any] = config_to_params(self.cfg) if self.cfg else {}
        self.t0: pd.Timestamp | None = None
        self.slim: pd.DataFrame | None = None
        self.line_meta: dict[str, dict[str, Any]] = {}  # cp_ref -> {borrower_id, type, source}
        self.borrower_dpd: dict[str, str | None] = {}
        self.fitted_: bool = False

    # -- internals ---------------------------------------------------------
    def _joint_T(self, dpd: str | None) -> np.ndarray:
        T_A = self.params["T_A"].copy()
        mult = (self.cfg.get("avoid_entry_multiplier_by_dpd") or {}).get(dpd or "", None)
        if mult is not None:
            e = min(0.99, T_A[0, 1] * float(mult))
            T_A[0, 0], T_A[0, 1] = 1.0 - e, e
        T_S = self.params["T_S"]
        J = np.zeros((N_J, N_J))
        for a in range(N_A):
            for ap in range(N_A):
                J[a * N_S : (a + 1) * N_S, ap * N_S : (ap + 1) * N_S] = T_A[a, ap] * T_S
        # invalid is absorbing at line level already via T_S row; keep as is.
        return J

    def _initial_joint(self, source: str) -> np.ndarray:
        pi_s = self.params["pi_s"].copy()
        mult = (self.cfg.get("source_initial_dead_multiplier") or {}).get(source, 1.0)
        dead = [S_IDX[s] for s in ("recycled", "invalid")]
        pi_s[dead] = pi_s[dead] * float(mult)
        pi_s /= pi_s.sum()
        pa0 = float(self.params["pi_a0"])
        j = np.zeros(N_J)
        j[0 * N_S : 1 * N_S] = (1 - pa0) * pi_s
        j[1 * N_S : 2 * N_S] = pa0 * pi_s
        return j / j.sum()

    def _emission_vec(self, kind: int, val: int) -> np.ndarray:
        """Likelihood vector over 12 joint states for one observation."""
        e = np.ones(N_J)
        if kind == K_NET:
            for a in range(N_A):
                e[a * N_S : (a + 1) * N_S] = self.params["E_net"][:, a, val]
        elif kind == K_DISP:
            for a in range(N_A):
                e[a * N_S : (a + 1) * N_S] = self.params["E_disp"][:, a, val]
        elif kind == K_PAY:
            ep = self.params["e_pay"]
            e[0 * N_S : 1 * N_S] = ep[0]
            e[1 * N_S : 2 * N_S] = ep[1]
        elif kind == K_CUE:
            odds = float((self.cfg.get("transcript_cues") or {}).get("avoid_odds_ratio", 3.0))
            if val & 1:
                e[1 * N_S : 2 * N_S] = odds  # avoid cue: A=1 more likely
            # wrong-number cue handled as dead-shift override at filter time
        return e

    def _apply_overrides(self, alpha: np.ndarray, kind: int, val: int) -> np.ndarray:
        """Hard-evidence rules applied BEFORE the probabilistic update."""
        ov = self.cfg.get("overrides", {}) or {}
        a = alpha.copy()
        if kind == K_NET and val == R_IDX["does_not_exist"]:
            rule = ov.get("does_not_exist", {})
            a = self._shift_to_dead(a, float(rule.get("mass_shift", 0.55)), float(rule.get("to_invalid", 0.60)))
        elif kind == K_DISP and val == D_IDX["wrong_number"]:
            rule = ov.get("wrong_number", {})
            a = self._shift_to_dead(a, float(rule.get("mass_shift", 0.45)), float(rule.get("to_invalid", 0.45)))
        elif kind == K_CUE and (val & 2):
            rule = {"mass_shift": float((self.cfg.get("transcript_cues") or {}).get("wrong_number_dead_shift", 0.30))}
            a = self._shift_to_dead(a, rule["mass_shift"], 0.5)
        elif kind == K_DISP and val in (D_IDX["rpc"], D_IDX["promise_to_pay"]):
            # Own-line collapse to valid (keep A distribution); cross-line
            # reset handled via the borrower's reset schedule.
            tot = a.sum()
            if tot > 0:
                pa = np.array([a[0:6].sum(), a[6:12].sum()]) / tot
                a = np.zeros(N_J)
                a[0 * N_S + S_IDX["valid"]] = pa[0]
                a[1 * N_S + S_IDX["valid"]] = pa[1]
        return a

    @staticmethod
    def _shift_to_dead(alpha: np.ndarray, shift: float, to_invalid: float) -> np.ndarray:
        live = [S_IDX[s] for s in ("valid", "temp_unreachable", "switched_off_long", "third_party")]
        dead_r, dead_i = S_IDX["recycled"], S_IDX["invalid"]
        a = alpha.copy()
        for sl in (slice(0, 6), slice(6, 12)):
            block = a[sl]
            live_mass = sum(block[live])
            move = live_mass * shift
            if move <= 0:
                continue
            for s in live:
                block[s] *= 1.0 - shift
            block[dead_i] += move * to_invalid
            block[dead_r] += move * (1.0 - to_invalid)
        return a

    @staticmethod
    def _apply_reset(alpha: np.ndarray, rho: float) -> np.ndarray:
        """Move fraction rho of A=1 mass to A=0, preserving S|A distribution."""
        a = alpha.copy()
        m1 = a[6:12].sum()
        if m1 > 0 and rho > 0:
            move = a[6:12] * rho
            a[6:12] -= move
            # redistribute proportionally to A=0's S distribution (or uniform if empty)
            base = a[0:6]
            bs = base.sum()
            w = base / bs if bs > 0 else np.full(6, 1 / 6)
            a[0:6] += move.sum() * w
        return a

    def _event_grid(
        self, days: np.ndarray, kinds: np.ndarray, vals: np.ndarray, resets: set[int], end_day: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Collapse obs onto event days, injecting reset-only days.

        Returns (steps, first_day); each step has day, gap (days since the
        previous step, 0 for the first), obs list, reset flag. This guarantees
        resets on days without observations are still applied.
        """
        obs_by_day: dict[int, list[tuple[int, int]]] = {}
        for t in range(len(days)):
            obs_by_day.setdefault(int(days[t]), []).append((int(kinds[t]), int(vals[t])))
        first = int(days[0]) if len(days) else end_day
        rset = {int(r) for r in resets if first <= int(r) <= end_day}
        steps: list[dict[str, Any]] = []
        prev = first
        for d in sorted(set(obs_by_day) | rset):
            steps.append({"day": d, "gap": d - prev, "obs": obs_by_day.get(d, []), "reset": d in rset})
            prev = d
        return steps, first

    def _forward_steps(
        self,
        steps: list[dict[str, Any]],
        J: np.ndarray,
        init: np.ndarray,
        end_day: int,
        rho_use: float,
        gap_cap: int,
        last_day: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scaled forward pass over event steps. Returns (alphas, scales, final)."""
        E = len(steps)
        alphas = np.zeros((E + 1, N_J))
        scales = np.ones(E + 1)
        alpha = init.copy()
        alphas[0] = alpha
        for e, st in enumerate(steps):
            g = min(max(int(st["gap"]), 0), gap_cap)
            if g:
                alpha = alpha @ np.linalg.matrix_power(J, g)
            if st["reset"]:
                alpha = self._apply_reset(alpha, rho_use)
            for k, v in st["obs"]:
                alpha = self._apply_overrides(alpha, k, v)
                alpha = alpha * self._emission_vec(k, v)
                s = alpha.sum()
                alpha = alpha / s if s > 0 else init.copy()
            s = alpha.sum()
            alpha = alpha / s if s > 0 else init.copy()
            alphas[e + 1] = alpha
            # scale bookkeeping happens in caller (EM) via explicit c
            scales[e + 1] = 1.0
        tail = min(max(end_day - last_day, 0), gap_cap)
        final = alpha
        if tail:
            final = final @ np.linalg.matrix_power(J, tail)
            s = final.sum()
            final = final / s if s > 0 else init.copy()
        return alphas, scales, final

    def _filter_line(
        self,
        days: np.ndarray,
        kinds: np.ndarray,
        vals: np.ndarray,
        resets: set[int],
        J: np.ndarray,
        init: np.ndarray,
        end_day: int,
        rho_rpc: float,
        collect: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Forward filter for one line. Returns final alpha (or full path)."""
        steps, first = self._event_grid(days, kinds, vals, resets, end_day)
        last_day = steps[-1]["day"] if steps else first
        _, _, final = self._forward_steps(steps, J, init, end_day, rho_rpc, 366, last_day)
        if collect:
            # per-step path for diagnostics
            alphas, _, _ = self._forward_steps(steps, J, init, end_day, rho_rpc, 366, last_day)
            return final, alphas
        return final

    # -- fit ---------------------------------------------------------------
    def fit(self, events_df: pd.DataFrame, config: str | Path | dict[str, Any] | None = None) -> "StateTracker":
        """Fit parameters by EM on events only (no labels)."""
        if config is not None:
            self.cfg = load_config(config)
            self.use_latent = bool(self.cfg.get("use_borrower_latent", True))
        if not self.cfg:
            raise ValueError("no config: pass config to __init__ or fit")
        self.params = config_to_params(self.cfg)
        cues = self.cfg.get("transcript_cues") or {}
        slim = parse_events(
            events_df, tuple(cues.get("avoid_phrases", []) or []), tuple(cues.get("wrong_number_phrases", []) or [])
        )
        # fit subsample cap
        max_b = int(self.cfg.get("em", {}).get("max_fit_borrowers", 5000))
        seed = int(self.cfg.get("em", {}).get("fit_seed", 7))
        borrowers = sorted(slim["borrower_id"].unique())
        if len(borrowers) > max_b:
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(borrowers, size=max_b, replace=False))
            slim = slim[slim["borrower_id"].isin(keep)].reset_index(drop=True)
        self.t0 = slim["received_at"].min()
        slim = slim.copy()
        slim["day"] = ((slim["received_at"] - self.t0).dt.total_seconds() // 86400).astype(int)
        self._set_state(slim)
        self._run_em()
        self.fitted_ = True
        return self

    def _set_state(self, slim: pd.DataFrame) -> None:
        """Attach a parsed event frame (t0, line meta, borrower DPD).

        Used by fit() and by point-in-time tests to swap the event history
        while holding parameters fixed.
        """
        self.t0 = slim["received_at"].min()
        if "day" not in slim.columns:
            slim = slim.copy()
            slim["day"] = ((slim["received_at"] - self.t0).dt.total_seconds() // 86400).astype(int)
        self.slim = slim
        # meta (payment-only pseudo-lines contribute resets, not lines)
        self.line_meta = {}
        for cp, g in slim.groupby("cp_ref"):
            g = g.sort_values("received_at")
            if bool(((g["kind"] == K_PAY).all())):
                continue
            srcs = g[g["kind"] == -1]["src"]
            ctype = str(g["ctype"].iloc[-1])
            self.line_meta[cp] = {
                "borrower_id": str(g["borrower_id"].iloc[0]),
                "type": ctype,
                "source": str(srcs.iloc[-1]) if len(srcs) else str(g["src"].iloc[-1]),
            }
        self.borrower_dpd = {
            b: (None if g["dpd"].isna().all() else str(g["dpd"].dropna().iloc[-1]))
            for b, g in slim.groupby("borrower_id")
        }

    def _line_sequences(self, end_day: int | None = None) -> list[dict[str, Any]]:
        assert self.slim is not None
        seqs = []
        for (b, cp), g in self.slim.groupby(["borrower_id", "cp_ref"]):
            meta = self.line_meta.get(cp, {})
            if meta.get("type", "phone") != "phone":
                continue
            g = g.sort_values("received_at")
            obs = g[g["kind"] >= 0]
            resets = set(g[g["is_reset"]]["day"].astype(int).tolist())
            seqs.append(
                {
                    "borrower": b,
                    "cp": cp,
                    "dpd": self.borrower_dpd.get(b),
                    "source": meta.get("source", "KYC"),
                    "days": obs["day"].to_numpy(dtype=int),
                    "kinds": obs["kind"].to_numpy(dtype=int),
                    "vals": obs["val"].to_numpy(dtype=int),
                    "resets": resets,
                }
            )
        return seqs

    def _run_em(self) -> None:
        # Baum-Welch on per-line joint (A, S) chains. Approximations, all
        # documented in docs/state_tracker.md: (a) lines treated as
        # independent sequences in the E-step (no sibling A-pooling while
        # learning); (b) hard overrides/resets are applied in the forward
        # pass but their non-linearity is ignored in the backward pass;
        # (c) day gaps folded into an effective transition J^gap (gap capped
        # at 60: beyond that the chain is ~stationary).
        em = self.cfg.get("em", {})
        n_iter = int(em.get("n_iter", 12))
        tol = float(em.get("tol", 1e-4))
        s_tr = float(em.get("prior_strength_transition", 20.0))
        s_em = float(em.get("prior_strength_emission", 10.0))
        s_pi = float(em.get("prior_strength_initial", 5.0))
        cfg0 = config_to_params(self.cfg)  # fixed Dirichlet prior means
        rho = float((self.cfg.get("overrides", {}) or {}).get("rpc_answer", {}).get("avoid_reset", 0.9))
        rho_pay = float((self.cfg.get("overrides", {}) or {}).get("payment", {}).get("avoid_reset", 0.95))
        rho_use = max(rho, rho_pay)
        prev_ll = -np.inf
        for _ in range(n_iter):
            xi = np.zeros((N_J, N_J))
            cnt_net = np.zeros((N_S, N_A, len(R_NAMES)))
            cnt_disp = np.zeros((N_S, N_A, len(D_NAMES)))
            cnt_pi_s = np.zeros(N_S)
            cnt_pi_a = np.zeros(N_A)
            ll = 0.0
            Jcache: dict[str, np.ndarray] = {}
            Jeffcache: dict[tuple[str, int], np.ndarray] = {}
            for sq in self._line_sequences():
                dpd = sq["dpd"] or ""
                if dpd not in Jcache:
                    Jcache[dpd] = self._joint_T(sq["dpd"])
                J = Jcache[dpd]
                init = self._initial_joint(sq["source"])
                T = len(sq["days"])
                if T == 0:
                    continue
                steps, _first = self._event_grid(
                    sq["days"], sq["kinds"], sq["vals"], sq["resets"], int(sq["days"].max())
                )
                E = len(steps)
                if E == 0:
                    continue

                def jeff(gap: int) -> np.ndarray:
                    g = min(max(int(gap), 0), 60)
                    key = (dpd, g)
                    if key not in Jeffcache:
                        Jeffcache[key] = np.linalg.matrix_power(J, g) if g > 0 else np.eye(N_J)
                    return Jeffcache[key]

                emis = np.zeros((E, N_J))
                for e, st in enumerate(steps):
                    vec = np.ones(N_J)
                    for k, v in st["obs"]:
                        vec = vec * self._emission_vec(k, v)
                    emis[e] = vec
                # forward (scaled) with overrides/resets
                alphas = np.zeros((E + 1, N_J))
                scales = np.zeros(E + 1)
                scales[0] = 1.0
                alpha = init.copy()
                for d in sorted(sq["resets"]):
                    if d < int(sq["days"][0]):
                        alpha = self._apply_reset(alpha, rho_use)
                alphas[0] = alpha
                ok = True
                for e, st in enumerate(steps):
                    alpha = alpha @ jeff(st["gap"])
                    if st["reset"]:
                        # (Resets before the first obs day were applied to
                        # init above and are excluded from steps by
                        # _event_grid, so no double application.)
                        alpha = self._apply_reset(alpha, rho_use)
                    for k, v in st["obs"]:
                        alpha = self._apply_overrides(alpha, k, v)
                        alpha = alpha * self._emission_vec(k, v)
                    s = alpha.sum()
                    if s <= 0 or not np.isfinite(s):
                        ok = False
                        break
                    alpha = alpha / s
                    alphas[e + 1] = alpha
                    scales[e + 1] = s
                if not ok:
                    continue
                ll += float(np.log(scales[1:][scales[1:] > 0]).sum())
                # backward (scaled): beta_E = 1,
                # beta_e = Jeff_e @ (emis_e . beta_{e+1}) / c_{e+1}
                betas = np.zeros((E + 1, N_J))
                betas[E] = np.ones(N_J)
                for e in range(E - 1, -1, -1):
                    b = jeff(steps[e]["gap"]) @ (emis[e] * betas[e + 1])
                    c = scales[e + 1]
                    betas[e] = b / c if c > 0 else np.ones(N_J) / N_J
                gamma = alphas[1:] * betas[1:]
                gamma /= gamma.sum(axis=1, keepdims=True).clip(min=1e-300)
                gamma_init = alphas[0] * betas[0]
                gamma_init /= max(gamma_init.sum(), 1e-300)
                for e in range(E):
                    num = (alphas[e][:, None] * jeff(steps[e]["gap"])) * (emis[e] * betas[e + 1])[None, :]
                    xi += num / max(num.sum(), 1e-300)
                cnt_pi_s += gamma_init.reshape(N_A, N_S).sum(axis=0)
                cnt_pi_a += gamma_init.reshape(N_A, N_S).sum(axis=1)
                # attribute step posteriors to each obs of that day
                for e, st in enumerate(steps):
                    if not st["obs"]:
                        continue
                    w = 1.0 / len(st["obs"])
                    for k, v in st["obs"]:
                        g = gamma[e].reshape(N_A, N_S).T * w  # (S, A)
                        if k == K_NET:
                            cnt_net[:, :, v] += g
                        elif k == K_DISP:
                            cnt_disp[:, :, v] += g
            # M-step with fixed Dirichlet priors
            nA = xi.reshape(N_A, N_S, N_A, N_S).sum(axis=(1, 3))
            pA = cfg0["T_A"]
            self.params["T_A"] = (nA + s_tr * pA) / (nA + s_tr * pA).sum(axis=1, keepdims=True)
            nS = xi.reshape(N_A, N_S, N_A, N_S).sum(axis=(0, 2))
            pS = cfg0["T_S"]
            row = nS + s_tr * pS
            self.params["T_S"] = row / row.sum(axis=1, keepdims=True)
            self.params["E_net"] = (cnt_net + s_em * cfg0["E_net"]) / (
                cnt_net + s_em * cfg0["E_net"]
            ).sum(axis=2, keepdims=True)
            self.params["E_disp"] = (cnt_disp + s_em * cfg0["E_disp"]) / (
                cnt_disp + s_em * cfg0["E_disp"]
            ).sum(axis=2, keepdims=True)
            pi_s_new = cnt_pi_s + s_pi * cfg0["pi_s"]
            self.params["pi_s"] = pi_s_new / pi_s_new.sum()
            pi_a_new = cnt_pi_a + s_pi * np.array([1 - cfg0["pi_a0"], cfg0["pi_a0"]])
            self.params["pi_a0"] = float(pi_a_new[1] / pi_a_new.sum())
            if abs(ll - prev_ll) < tol * max(1.0, abs(ll)):
                prev_ll = ll
                break
            prev_ll = ll

    # -- score -------------------------------------------------------------
    def _borrower_groups(self, as_of_day: int) -> dict[str, dict[str, Any]]:
        assert self.slim is not None and self.t0 is not None
        groups: dict[str, dict[str, Any]] = {}
        for b, g in self.slim.groupby("borrower_id"):
            g = g[g["day"] <= as_of_day]
            if len(g) == 0:
                continue
            lines: dict[str, Any] = {}
            for cp, lg in g.groupby("cp_ref"):
                lg = lg.sort_values("received_at")
                obs = lg[lg["kind"] >= 0]
                if len(obs) == 0:
                    continue
                nonpay = obs[obs["kind"] != K_PAY]
                if len(nonpay) == 0:
                    continue  # payment-only pseudo-line: contributes resets, not a line
                strong_here = set(
                    nonpay[
                        (nonpay["kind"] == K_DISP)
                        & (nonpay["val"].isin([D_IDX["rpc"], D_IDX["promise_to_pay"]]))
                    ]["day"].astype(int).tolist()
                )
                lines[cp] = {
                    "days": nonpay["day"].to_numpy(dtype=int),
                    "kinds": nonpay["kind"].to_numpy(dtype=int),
                    "vals": nonpay["val"].to_numpy(dtype=int),
                    "resets": set(lg[lg["is_reset"]]["day"].astype(int).tolist()),
                    "own_rpc_days": strong_here,
                    "last_info_day": int(nonpay["day"].max()) if len(nonpay) else None,
                }
            reset_days = set(g[g["is_reset"]]["day"].astype(int).tolist())
            pay_days = set(g[g["kind"] == K_PAY]["day"].astype(int).tolist())
            for ld in lines.values():
                if ld["last_info_day"] is None and pay_days:
                    ld["last_info_day"] = max(pay_days)
                elif ld["last_info_day"] is not None and pay_days:
                    ld["last_info_day"] = max(ld["last_info_day"], max(pay_days))
            groups[b] = {"lines": lines, "resets": reset_days, "pay_days": pay_days, "dpd": self.borrower_dpd.get(b)}
        return groups

    def score(
        self, as_of: str | datetime, contact_point_refs: list[str] | None = None
    ) -> list[ContactPointScore]:
        """Score contact points point-in-time (only received_at <= as_of)."""
        if not self.fitted_:
            raise RuntimeError("StateTracker not fitted: call fit() first")
        assert self.slim is not None and self.t0 is not None
        as_of_dt = _ensure_tz(as_of)
        as_of_ts = pd.Timestamp(as_of_dt)
        if as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        t0 = self.t0.tz_convert("UTC") if self.t0.tzinfo is not None else self.t0.tz_localize("UTC")
        as_of_day = int((as_of_ts - t0).total_seconds() // 86400)
        groups = self._borrower_groups(as_of_day)
        rho = float((self.cfg.get("overrides", {}) or {}).get("rpc_answer", {}).get("avoid_reset", 0.9))
        rho_pay = float((self.cfg.get("overrides", {}) or {}).get("payment", {}).get("avoid_reset", 0.95))
        rho_use = max(rho, rho_pay)
        tau = float(self.cfg.get("confidence_tau_days", 30.0))
        ans = float(self.params["ans"]) * float(self.cfg.get("slot_multiplier_default", 1.0))

        wanted = set(contact_point_refs) if contact_point_refs else None
        out: list[ContactPointScore] = []
        # map cp -> borrower for quick lookup
        cp2b = {cp: m["borrower_id"] for cp, m in self.line_meta.items()}
        targets = list(wanted) if wanted else list(self.line_meta.keys())
        for cp in targets:
            meta = self.line_meta.get(cp, {"borrower_id": None, "type": "phone", "source": "KYC"})
            ctype = meta.get("type", "phone")
            if ctype == "address":
                out.append(self._address_score(cp, as_of_dt))
                continue
            b = cp2b.get(cp)
            grp = groups.get(b) if b else None
            if grp is None or cp not in grp["lines"]:
                out.append(self._prior_score(cp, as_of_dt, meta))
                continue
            J = self._joint_T(grp["dpd"])
            # per-line filters
            finals: dict[str, np.ndarray] = {}
            for lcp, ld in grp["lines"].items():
                lmeta = self.line_meta.get(lcp, {"source": "KYC"})
                init = self._initial_joint(lmeta.get("source", "KYC"))
                if self.use_latent:
                    line_resets = set(grp["resets"])
                else:
                    # Ablation: lines independent; only own rpc resets and
                    # borrower-level payments (observed facts) are shared.
                    line_resets = set(ld["own_rpc_days"]) | set(grp["pay_days"])
                # Resets strictly before the first obs shift the initial mass;
                # resets on/after it are applied inside _filter_line.
                resets = {d for d in line_resets if (len(ld["days"]) == 0 or d < int(ld["days"][0]))}
                alpha = init.copy()
                for d in sorted(resets):
                    alpha = self._apply_reset(alpha, rho_use)
                if len(ld["days"]):
                    alpha = self._filter_line(
                        ld["days"], ld["kinds"], ld["vals"], line_resets, J, alpha, as_of_day, rho_use
                    )
                else:
                    for _ in range(min(max(as_of_day, 0), 366)):
                        alpha = alpha @ J
                    alpha /= alpha.sum()
                assert isinstance(alpha, np.ndarray)
                finals[lcp] = alpha
            # pool avoidance across sibling lines
            pooled = self._pool_avoidance(
                [finals[l] for l in grp["lines"]], self.params["pi_a0"]
            )
            alpha = finals[cp]
            pa = np.array([alpha[0:6].sum(), alpha[6:12].sum()])
            pa = pa / pa.sum() if pa.sum() > 0 else np.array([0.5, 0.5])
            joint = np.zeros((N_A, N_S))
            for a in range(N_A):
                cond = alpha[a * 6 : (a + 1) * 6] / pa[a] if pa[a] > 0 else np.full(6, 1 / 6)
                joint[a] = cond * pooled[a]
            if self.use_latent:
                joint = self._apply_silence_shift(joint, grp["lines"], cp, grp["resets"])
            out.append(self._joint_to_score(cp, joint, as_of_dt, grp["lines"][cp]["last_info_day"], as_of_day, tau, ans))
        return out

    def _apply_silence_shift(
        self, joint: np.ndarray, lines: dict[str, Any], target: str, reset_days: set[int]
    ) -> np.ndarray:
        """Explained-away shift: silent dials on borrower-active days move
        valid mass to dead states. See configs/state_tracker.yaml rationale."""
        w_strong = float(self.cfg.get("silence_strong_weight", 1.0))
        w_weak = float(self.cfg.get("silence_weak_weight", 0.5))
        kappa = float(self.cfg.get("silence_shift_kappa", 3.0))
        day_w: dict[int, float] = {}
        for lcp, ld in lines.items():
            for t in range(len(ld["days"])):
                if int(ld["kinds"][t]) == K_NET and int(ld["vals"][t]) == R_IDX["answered"]:
                    d = int(ld["days"][t])
                    day_w[d] = max(day_w.get(d, 0.0), w_weak)
        for d in reset_days:
            day_w[d] = max(day_w.get(d, 0.0), w_strong)
        ld = lines[target]
        dialed: dict[int, bool] = {}
        answered: dict[int, bool] = {}
        for t in range(len(ld["days"])):
            if int(ld["kinds"][t]) == K_NET:
                d = int(ld["days"][t])
                dialed[d] = True
                if int(ld["vals"][t]) == R_IDX["answered"]:
                    answered[d] = True
        W = sum(w for d, w in day_w.items() if dialed.get(d) and not answered.get(d))
        sigma = 1.0 - math.exp(-W / kappa) if kappa > 0 else 0.0
        if sigma <= 0 or W <= 0:
            return joint
        j = joint.copy()
        v = S_IDX["valid"]
        move = (j[0, v] + j[1, v]) * sigma
        if move <= 0:
            return joint
        j[:, v] *= 1.0 - sigma
        dead = [s for s in range(N_S) if s != v]
        pi = np.asarray(self.params["pi_s"], dtype=float)
        for a in range(N_A):
            w = j[a, dead] + 0.1 * pi[dead] * max(j[a].sum(), 1e-9)
            w = w / w.sum() if w.sum() > 0 else np.full(len(dead), 1 / len(dead))
            share = move * (j[a].sum() / max(j.sum(), 1e-300))
            for s, ww in zip(dead, w):
                j[a, s] += share * ww
        s = j.sum()
        return j / s if s > 0 else joint

    def _pool_avoidance(self, alphas: list[np.ndarray], pi_a0: float) -> np.ndarray:
        if not self.use_latent or len(alphas) <= 1:
            a = alphas[0] if len(alphas) == 1 else sum(alphas)
            pa = np.array([a[0:6].sum(), a[6:12].sum()])
            return pa / pa.sum() if pa.sum() > 0 else np.array([0.5, 0.5])
        prior = np.array([1 - pi_a0, pi_a0]).clip(min=1e-9)
        logp = np.zeros(2)
        for al in alphas:
            pa = np.array([al[0:6].sum(), al[6:12].sum()]).clip(min=1e-300)
            logp += np.log(pa / pa.sum())
        logp -= (len(alphas) - 1) * np.log(prior)
        p = np.exp(logp - logp.max())
        return p / p.sum()

    def _joint_to_score(
        self,
        cp: str,
        joint: np.ndarray,
        as_of: datetime,
        last_info_day: int | None,
        as_of_day: int,
        tau: float,
        ans: float,
    ) -> ContactPointScore:
        posterior = {
            "valid_reachable": float(joint[0, S_IDX["valid"]]),
            "avoiding": float(joint[1, S_IDX["valid"]]),
            "temp_unreachable": float(joint[0, S_IDX["temp_unreachable"]] + joint[1, S_IDX["temp_unreachable"]]),
            "switched_off_long": float(joint[0, S_IDX["switched_off_long"]] + joint[1, S_IDX["switched_off_long"]]),
            "recycled": float(joint[0, S_IDX["recycled"]] + joint[1, S_IDX["recycled"]]),
            "third_party": float(joint[0, S_IDX["third_party"]] + joint[1, S_IDX["third_party"]]),
            "invalid": float(joint[0, S_IDX["invalid"]] + joint[1, S_IDX["invalid"]]),
        }
        tot = sum(posterior.values())
        posterior = {k: v / tot for k, v in posterior.items()}
        p_rpc = min(1.0, posterior["valid_reachable"] * ans)
        h = -sum(v * math.log(v) for v in posterior.values() if v > 0)
        hmax = math.log(len(STATE_KEYS))
        if last_info_day is None:
            conf = 0.0
        else:
            gap = max(0, as_of_day - last_info_day)
            conf = (1.0 - h / hmax) * math.exp(-gap / tau)
        return ContactPointScore(
            contact_point_ref=cp,
            type="phone",
            as_of=as_of,
            state_posterior=posterior,
            p_rpc=float(min(1.0, max(0.0, p_rpc))),
            recycled_risk=float(posterior["recycled"]),
            confidence=float(min(1.0, max(0.0, conf))),
        )

    def _prior_score(self, cp: str, as_of: datetime, meta: dict[str, Any]) -> ContactPointScore:
        init = self._initial_joint(meta.get("source", "KYC"))
        joint = np.zeros((N_A, N_S))
        joint[0] = init[0:6]
        joint[1] = init[6:12]
        tau = float(self.cfg.get("confidence_tau_days", 30.0))
        ans = float(self.params["ans"]) * float(self.cfg.get("slot_multiplier_default", 1.0))
        return self._joint_to_score(cp, joint, as_of, None, 0, tau, ans)

    def _address_score(self, cp: str, as_of: datetime) -> ContactPointScore:
        """Thin address stub (phase 2): prior-only posterior, low confidence."""
        init = self._initial_joint("KYC")
        joint = np.zeros((N_A, N_S))
        joint[0] = init[0:6]
        joint[1] = init[6:12]
        s = self._joint_to_score(cp, joint, as_of, None, 0, 30.0, 0.1)
        return ContactPointScore(
            contact_point_ref=cp,
            type="address",
            as_of=as_of,
            state_posterior=s.state_posterior,
            p_rpc=s.p_rpc,
            recycled_risk=s.recycled_risk,
            confidence=0.0,
        )

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p / "params.npz",
            **{k: np.asarray(v, dtype=float) for k, v in self.params.items()},
        )
        assert self.slim is not None and self.t0 is not None
        try:
            self.slim.to_parquet(p / "events.parquet", index=False)
            events_fmt = "parquet"
        except ImportError:
            self.slim.to_pickle(p / "events.pkl")
            events_fmt = "pickle"
        meta = {
            "version": self.version,
            "use_latent": self.use_latent,
            "t0": str(self.t0),
            "line_meta": self.line_meta,
            "borrower_dpd": self.borrower_dpd,
            "fitted": self.fitted_,
            "events_fmt": events_fmt,
        }
        (p / "meta.json").write_text(json.dumps(meta))
        (p / "config.yaml").write_text(yaml.safe_dump(self.cfg))

    @classmethod
    def load(cls, path: str | Path) -> "StateTracker":
        p = Path(path)
        obj = cls(config=yaml.safe_load((p / "config.yaml").read_text()))
        z = np.load(p / "params.npz")
        obj.params = {k: z[k] for k in z.files}
        meta = json.loads((p / "meta.json").read_text())
        if meta.get("events_fmt", "parquet") == "parquet":
            obj.slim = pd.read_parquet(p / "events.parquet")
        else:
            obj.slim = pd.read_pickle(p / "events.pkl")
        obj.t0 = pd.Timestamp(meta["t0"])
        obj.line_meta = meta["line_meta"]
        obj.borrower_dpd = meta["borrower_dpd"]
        obj.use_latent = bool(meta["use_latent"])
        obj.fitted_ = bool(meta["fitted"])
        return obj


# ---------------------------------------------------------------------------
# Scorer adapter (eval-harness DataFrame form)
# ---------------------------------------------------------------------------


class StateTrackerScorer:
    """Adapter returning the DataFrame form expected by the eval harness."""

    def __init__(self, tracker: StateTracker) -> None:
        self.tracker = tracker

    def score_df(self, as_of: str | datetime, contact_point_refs: list[str] | None = None) -> pd.DataFrame:
        scores = self.tracker.score(as_of, contact_point_refs)
        rows = []
        for s in scores:
            row = {
                "contact_point_ref": s.contact_point_ref,
                "p_rpc": s.p_rpc,
                "state_posterior": dict(s.state_posterior),
                "recycled_risk": s.recycled_risk,
                "confidence": s.confidence,
            }
            for k in STATE_KEYS:
                row[f"sp_{k}"] = s.state_posterior[k]
            rows.append(row)
        cols = ["contact_point_ref", "p_rpc", "state_posterior", "recycled_risk", "confidence"] + [
            f"sp_{k}" for k in STATE_KEYS
        ]
        return pd.DataFrame(rows, columns=cols)


def try_register_eval(scorer: StateTrackerScorer) -> bool:
    """Register with the eval-harness registry if it exists on main."""
    try:
        from src.rpc.eval.registry import register  # type: ignore[import-not-found]

        register("state_tracker", scorer)
        return True
    except Exception:
        return False
