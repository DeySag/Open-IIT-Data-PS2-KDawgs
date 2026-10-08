"""Tests for the contact-point state tracker + borrower avoidance latent.

Model-quality assertions run on invented fixture data plus a self-contained
mini generator. No test reads hidden-truth files except the slow eval test,
which goes through the eval harness (or skips when issued extracts with
annotations are absent).
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.rpc.models.state_tracker import StateTracker
from src.rpc.models.types import STATE_KEYS

BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)
CFG_PATH = Path("configs/state_tracker.yaml")


def load_cfg(**overrides):
    cfg = yaml.safe_load(CFG_PATH.read_text())
    cfg["em"]["n_iter"] = 1
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def ev(borrower, cp, day, etype, payload):
    ts = (BASE + timedelta(days=day)).isoformat()
    return {
        "event_id": "e",
        "event_type": etype,
        "lender_id": "L",
        "borrower_id": borrower,
        "account_id": "A",
        "contact_point_ref": cp,
        "occurred_at": ts,
        "received_at": ts,
        "payload": payload,
    }


def dials(borrower, cp, days, response, ring=30.0):
    return [ev(borrower, cp, d, "dial_attempt", {"network_response": response, "ring_seconds": ring}) for d in days]


def dead_mass(score) -> float:
    p = score.state_posterior
    return 1.0 - p["valid_reachable"] - p["avoiding"]


@pytest.fixture(scope="module")
def fixture_df():
    rows = []
    # B1: silent L1 + answering sibling L2 (with rpc) -> L1 should look dead.
    rows += dials("B1", "L1", range(6), "no_answer")
    rows += dials("B1", "L2", range(6), "answered", ring=3.0)
    rows.append(ev("B1", "L2", 6, "disposition", {"disposition": "rpc"}))
    # B2: both lines silent -> avoidance.
    rows += dials("B2", "M1", range(6), "no_answer")
    rows += dials("B2", "M2", range(6), "no_answer")
    # B3: repeated does_not_exist -> invalid.
    rows += dials("B3", "D1", range(4), "does_not_exist", ring=0.2)
    # B4: silent then payment -> avoidance lowered.
    rows += dials("B4", "P1", range(5), "no_answer")
    rows.append(
        ev(
            "B4",
            "P1",
            5,
            "payment",
            {"amount": 1000.0, "payment_mode": "upi", "received_at": (BASE + timedelta(days=5)).isoformat()},
        )
    )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tracker(fixture_df):
    tr = StateTracker(load_cfg())
    tr.fit(fixture_df)
    return tr


ASOF = BASE + timedelta(days=7)


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------


def test_posterior_shape_sums_no_nan(tracker):
    scores = tracker.score(ASOF)
    assert len(scores) > 0
    for s in scores:
        assert set(s.state_posterior.keys()) == set(STATE_KEYS)
        vals = list(s.state_posterior.values())
        assert all(v == v for v in vals), "NaN in posterior"
        assert all(v >= 0.0 for v in vals)
        assert abs(sum(vals) - 1.0) < 1e-9
        assert 0.0 <= s.p_rpc <= 1.0
        assert 0.0 <= s.recycled_risk <= 1.0
        assert 0.0 <= s.confidence <= 1.0
        assert s.recycled_risk == pytest.approx(s.state_posterior["recycled"])


def test_point_in_time_ignores_future_events(fixture_df, tracker):
    import copy as _copy

    before = {s.contact_point_ref: s for s in tracker.score(ASOF)}
    # Same parameters, but the event history now includes future events:
    # score at ASOF must be bit-identical (only received_at <= ASOF is used).
    extra = pd.DataFrame([ev("B1", "L1", 30, "dial_attempt", {"network_response": "answered", "ring_seconds": 2.0})])
    full = pd.concat([fixture_df, extra], ignore_index=True)
    tr2 = StateTracker(load_cfg())
    tr2.params = _copy.deepcopy(tracker.params)
    tr2.cfg = tracker.cfg
    tr2.use_latent = tracker.use_latent
    cues = tracker.cfg.get("transcript_cues") or {}
    from src.rpc.models.state_tracker.model import parse_events

    slim = parse_events(
        full, tuple(cues.get("avoid_phrases", []) or []), tuple(cues.get("wrong_number_phrases", []) or [])
    )
    slim["day"] = ((slim["received_at"] - tracker.t0).dt.total_seconds() // 86400).astype(int)
    tr2._set_state(slim)
    assert tr2.t0 == tracker.t0  # extra events are strictly later; day grid coincides
    tr2.fitted_ = True
    after = {s.contact_point_ref: s for s in tr2.score(ASOF)}
    for cp, s in before.items():
        for k in STATE_KEYS:
            assert after[cp].state_posterior[k] == pytest.approx(s.state_posterior[k], abs=1e-12)
    # And the future event DOES change the later score (sanity: it is used).
    later = {s.contact_point_ref: s for s in tr2.score(BASE + timedelta(days=31))}
    assert abs(later["L1"].state_posterior["valid_reachable"] - before["L1"].state_posterior["valid_reachable"]) > 0.01


# ---------------------------------------------------------------------------
# Micro-fixtures
# ---------------------------------------------------------------------------


def test_does_not_exist_gives_invalid_mass(tracker):
    s = {x.contact_point_ref: x for x in tracker.score(ASOF)}["D1"]
    assert s.state_posterior["invalid"] > 0.8


def test_silent_line_with_answering_sibling_looks_dead(tracker):
    by_ref = {x.contact_point_ref: x for x in tracker.score(ASOF)}
    assert dead_mass(by_ref["L1"]) - dead_mass(by_ref["M1"]) > 0.30


def test_all_silent_lines_point_to_avoidance(tracker):
    by_ref = {x.contact_point_ref: x for x in tracker.score(ASOF)}
    assert by_ref["M1"].state_posterior["avoiding"] > 0.5


def test_payment_lowers_avoidance(tracker):
    by_ref = {x.contact_point_ref: x for x in tracker.score(ASOF)}
    assert by_ref["P1"].state_posterior["avoiding"] < 0.30
    assert by_ref["P1"].state_posterior["avoiding"] < by_ref["M1"].state_posterior["avoiding"]


def test_evidence_gap_increases_entropy(tracker):
    early = {x.contact_point_ref: x for x in tracker.score(BASE + timedelta(days=7))}["M1"]
    late = {x.contact_point_ref: x for x in tracker.score(BASE + timedelta(days=60))}["M1"]
    assert late.confidence < early.confidence


# ---------------------------------------------------------------------------
# Ablation: borrower latent off
# ---------------------------------------------------------------------------


def test_no_latent_ablation_degrades_cross_line(fixture_df):
    cfg = load_cfg()
    cfg["use_borrower_latent"] = False
    tr = StateTracker(cfg)
    tr.use_latent = False
    tr.fit(fixture_df)
    by_ref = {x.contact_point_ref: x for x in tr.score(ASOF)}
    assert abs(dead_mass(by_ref["L1"]) - dead_mass(by_ref["M1"])) < 0.05


# ---------------------------------------------------------------------------
# Save / load + scorer adapter
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tracker, tmp_path):
    tracker.save(tmp_path / "m")
    tr2 = StateTracker.load(tmp_path / "m")
    a = {s.contact_point_ref: s for s in tracker.score(ASOF)}
    b = {s.contact_point_ref: s for s in tr2.score(ASOF)}
    for cp in a:
        for k in STATE_KEYS:
            assert b[cp].state_posterior[k] == pytest.approx(a[cp].state_posterior[k], abs=1e-9)


def test_scorer_adapter_columns(tracker):
    from src.rpc.models.state_tracker import StateTrackerScorer

    df = StateTrackerScorer(tracker).score_df(ASOF)
    for col in ("contact_point_ref", "p_rpc", "state_posterior", "recycled_risk", "confidence"):
        assert col in df.columns
    for k in STATE_KEYS:
        assert f"sp_{k}" in df.columns
    assert (df[[f"sp_{k}" for k in STATE_KEYS]].sum(axis=1) - 1.0).abs().max() < 1e-9


def test_scorer_eval_protocol_and_registry(tracker):
    """Scorer conforms to src.rpc.eval.protocol.Scorer and registers."""
    from src.rpc.eval.protocol import STATE_COLUMNS, Scorer
    from src.rpc.eval.registry import get_scorer
    from src.rpc.models.state_tracker import StateTrackerScorer, try_register_eval
    from src.rpc.models.state_tracker.model import StateTrackerScorer as STS

    assert try_register_eval() is True
    sc = get_scorer("state_tracker")
    assert isinstance(sc, Scorer)
    assert sc.name == "state_tracker"
    refs = [s.contact_point_ref for s in tracker.score(ASOF)][:3]
    sc_fitted = STS(tracker)
    df = sc_fitted.score(ASOF, refs)
    assert list(df["contact_point_ref"]) == refs
    for col in ("p_rpc", "recycled_risk", "confidence", *STATE_COLUMNS):
        assert col in df.columns
    assert ((df["p_rpc"] >= 0.0) & (df["p_rpc"] <= 1.0)).all()
    assert (df[list(STATE_COLUMNS)].sum(axis=1) - 1.0).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# Parameter recovery on a self-contained mini generator
# ---------------------------------------------------------------------------


def _mini_generate(seed=11, n_lines=400, n_days=12):
    """Sample (A, S) trajectories + obs from known parameters.

    True params are deliberately different from the config priors so recovery
    is a real test. Returns (events_df, true_labels, true_params).
    """
    rng = np.random.default_rng(seed)
    states = ["valid", "temp_unreachable", "switched_off_long", "recycled", "third_party", "invalid"]
    resps = ["answered", "no_answer", "busy", "switched_off", "not_reachable", "does_not_exist", "immediate_hangup"]
    T_S = np.array(
        [
            [0.90, 0.04, 0.02, 0.005, 0.005, 0.02],
            [0.25, 0.60, 0.10, 0.01, 0.02, 0.02],
            [0.08, 0.08, 0.70, 0.06, 0.02, 0.06],
            [0.00, 0.00, 0.00, 0.95, 0.00, 0.05],
            [0.05, 0.05, 0.02, 0.00, 0.82, 0.06],
            [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
        ]
    )
    T_A = np.array([[0.96, 0.04], [0.06, 0.94]])
    # Sharp, well-separated emissions (rows S, cols responses), A shifts mass
    # between answered/no_answer/immediate_hangup for the valid state.
    E0 = np.array(
        [
            [0.45, 0.35, 0.06, 0.05, 0.05, 0.005, 0.035],
            [0.12, 0.30, 0.10, 0.25, 0.18, 0.005, 0.045],
            [0.02, 0.10, 0.03, 0.55, 0.20, 0.05, 0.05],
            [0.06, 0.20, 0.05, 0.10, 0.14, 0.40, 0.05],
            [0.45, 0.28, 0.06, 0.08, 0.08, 0.005, 0.045],
            [0.005, 0.10, 0.02, 0.06, 0.12, 0.65, 0.045],
        ]
    )
    E1 = E0.copy()
    E1[0] = np.array([0.03, 0.55, 0.05, 0.15, 0.10, 0.005, 0.115])
    rows, labels = [], []

    def pick(n, p):
        p = np.asarray(p, dtype=float)
        return int(rng.choice(n, p=p / p.sum()))

    for li in range(n_lines):
        b = f"G{li // 2}"
        cp = f"GC{li}"
        a = int(rng.random() < 0.25)
        s = pick(6, [0.55, 0.12, 0.10, 0.06, 0.07, 0.10])
        for d in range(n_days):
            E = E1 if a else E0
            r = resps[pick(7, E[s])]
            rows.append(ev(b, cp, d, "dial_attempt", {"network_response": r, "ring_seconds": 5.0}))
            labels.append((cp, d, a, s))
            a = int(rng.random() < T_A[a, 1])
            s = pick(6, T_S[s])
    return pd.DataFrame(rows), labels, {"T_S": T_S, "T_A": T_A, "E0": E0, "E1": E1}


def test_em_parameter_recovery():
    df, labels, true = _mini_generate()
    cfg = load_cfg()
    cfg["em"]["n_iter"] = 15
    tr = StateTracker(cfg)
    tr.fit(df)
    assert abs(tr.params["T_S"] - true["T_S"]).mean() < 0.15
    assert abs(tr.params["T_A"] - true["T_A"]).mean() < 0.10
    rec0 = tr.params["E_net"][:, 0, :]
    assert abs(rec0 - true["E0"]).mean() < 0.15
    # Posterior accuracy above chance (chance 1/7 ≈ 0.143 for the 7-key map).
    asof = BASE + timedelta(days=11)
    by_ref = {s.contact_point_ref: s for s in tr.score(asof)}
    key_of = {
        (0, 0): "valid_reachable",
        (1, 0): "avoiding",
        (0, 1): "temp_unreachable",
        (1, 1): "temp_unreachable",
        (0, 2): "switched_off_long",
        (1, 2): "switched_off_long",
        (0, 3): "recycled",
        (1, 3): "recycled",
        (0, 4): "third_party",
        (1, 4): "third_party",
        (0, 5): "invalid",
        (1, 5): "invalid",
    }
    # accuracy on final-day states only
    last = {}
    for cp, d, a, s in labels:
        last[cp] = (a, s)
    hits, n = 0, 0
    for cp, (a, s) in last.items():
        pred = max(by_ref[cp].state_posterior, key=by_ref[cp].state_posterior.get)
        hits += pred == key_of[(a, s)]
        n += 1
    acc = hits / n
    assert acc > 1 / 7 + 0.25, f"accuracy {acc:.3f} not above chance by margin"


# ---------------------------------------------------------------------------
# P4 fix: avoidance-prior split + slot-multiplier hook
# ---------------------------------------------------------------------------


def test_slot_hook_defaults_to_neutral(fixture_df, tracker):
    plain = {s.contact_point_ref: s.p_rpc for s in tracker.score(ASOF)}
    tracker.attach_slot_multipliers({})
    same = {s.contact_point_ref: s.p_rpc for s in tracker.score(ASOF)}
    assert plain == same


def test_slot_multiplier_scales_head_only(fixture_df, tracker):
    before = {s.contact_point_ref: s for s in tracker.score(ASOF)}
    tracker.attach_slot_multipliers({"L2": 2.0})
    try:
        after = {s.contact_point_ref: s for s in tracker.score(ASOF)}
    finally:
        tracker.attach_slot_multipliers({})
    assert after["L2"].p_rpc == pytest.approx(min(1.0, before["L2"].p_rpc * 2.0))
    assert after["L1"].p_rpc == pytest.approx(before["L1"].p_rpc)
    # posteriors untouched — the hook scales the head, never the states
    assert after["L2"].state_posterior == pytest.approx(before["L2"].state_posterior)


def test_avoidance_strengths_fall_back_to_shared():
    from src.rpc.models.state_tracker.model import StateTracker as _ST

    cfg = load_cfg()
    del cfg["em"]["prior_strength_avoidance"]
    del cfg["em"]["prior_strength_avoidance_init"]
    tr = _ST(cfg)
    assert tr.cfg["em"].get("prior_strength_avoidance", 20.0) == 20.0


def test_stronger_avoidance_prior_holds_pi_a0(fixture_df):
    from src.rpc.models.state_tracker.model import StateTracker as _ST

    weak = load_cfg()
    weak["em"]["prior_strength_avoidance"] = 1e-9
    weak["em"]["prior_strength_avoidance_init"] = 1e-9
    strong = load_cfg()
    strong["em"]["prior_strength_avoidance"] = 1e9
    strong["em"]["prior_strength_avoidance_init"] = 1e9
    tr_w, tr_s = _ST(weak), _ST(strong)
    tr_w.fit(fixture_df)
    tr_s.fit(fixture_df)
    prior_a0 = float(_ST(load_cfg()).params["pi_a0"])
    assert abs(tr_s.params["pi_a0"] - prior_a0) < abs(tr_w.params["pi_a0"] - prior_a0)
    assert tr_s.params["pi_a0"] == pytest.approx(prior_a0, rel=0.05)
