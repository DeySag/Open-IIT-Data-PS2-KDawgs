"""Eval harness tests (test fixtures throughout)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.rpc.eval import metrics as M
from src.rpc.eval.labels import observed_labels
from src.rpc.eval.report import generate_report
from src.rpc.eval.splits import check_splits, make_rolling_splits
from src.rpc.models.baselines.account_gbm import AccountGBMScorer

ROOT = Path(__file__).resolve().parents[1]


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


# --- Splits ---

def test_splits_embargo_and_no_overlap() -> None:
    splits = make_rolling_splits(utc(2026, 1, 15), 3, 7, 14, 3, 7)
    check_splits(splits, 3)
    for s in splits:
        assert s.test_start >= s.train_end + timedelta(days=3)
        assert s.test_start == s.embargo_end
    for a, b in zip(splits, splits[1:]):
        assert b.as_of > a.as_of
        assert b.as_of >= a.as_of + timedelta(days=7)


# --- Metric correctness on hand fixtures ---

def test_auc_hand_computed() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert M.roc_auc(y, p) == pytest.approx(0.75)


def test_brier_hand_computed() -> None:
    assert M.brier([1, 0], [1, 0]) == pytest.approx(0.0)
    assert M.brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_ece_perfect_is_zero() -> None:
    assert M.ece([0, 0, 1, 1], [0, 0, 1, 1], n_bins=2) == pytest.approx(0.0)


def test_wasted_attempts_and_detection_within_k() -> None:
    ev = pd.DataFrame(
        {
            "contact_point_ref": ["dead1"] * 4 + ["dead2"] * 2 + ["live1"],
            "occurred_at": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-10",
                 "2026-01-01", "2026-01-02", "2026-01-01"]
            ),
        }
    )
    per_cp = M.wasted_attempts(ev, {"dead1", "dead2"})
    assert int(per_cp.set_index("contact_point_ref").loc["dead1", "wasted_attempts"]) == 4
    # Policy acted on dead1 only, after 4 wasted dials; dead2 never acted on.
    det = M.detection_within_k(per_cp, {"dead1", "dead2"}, {"dead1"})
    assert det[6] == pytest.approx(0.5)   # dead1 acted within 6, dead2 not at all
    assert det[3] == pytest.approx(0.0)   # dead1 needed 4 > 3
    # With action cutoff: attempts after first action do not count.
    cut = pd.DataFrame({"contact_point_ref": ["dead1"],
                        "first_action_at": [pd.Timestamp("2026-01-03", tz="UTC")]})
    per_cp2 = M.wasted_attempts(ev, {"dead1"}, cut)
    assert int(per_cp2.set_index("contact_point_ref").loc["dead1", "wasted_attempts"]) == 2


# --- Leakage guard: baselines must never touch ground truth / policy log ---

FORBIDDEN = ("ground_truth", "policy_log")


def test_baselines_never_read_restricted_tables() -> None:
    pkg = ROOT / "src" / "rpc" / "models" / "baselines"
    assert pkg.exists()
    for f in pkg.glob("*.py"):
        src = f.read_text()
        for token in FORBIDDEN:
            assert token not in src, f"{f.name} references {token}"
    # Import-level: no such attribute may exist on baseline modules.
    import importlib

    for mod in ("incumbent", "account_gbm", "contact_gbm"):
        m = importlib.import_module(f"src.rpc.models.baselines.{mod}")
        assert not any(t in dir(m) for t in FORBIDDEN)


def test_eval_is_only_reader_of_restricted_tables() -> None:
    readers: list[str] = []
    for f in (ROOT / "src").rglob("*.py"):
        txt = f.read_text()
        if "ground_truth.parquet" in txt or "policy_log.parquet" in txt:
            readers.append(str(f.relative_to(ROOT)).replace("\\", "/"))
    assert readers, "expected eval to reference the restricted tables"
    assert all(r.startswith("src/rpc/eval/") for r in readers), readers


# --- Report end to end on dev-scale test data ---

def _tiny_events(n_cp: int = 40, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = utc(2026, 1, 1)
    for i in range(n_cp):
        ref = f"tcp{i:03d}"
        for k in range(rng.integers(2, 8)):
            t = base + timedelta(days=int(rng.integers(0, 28)), hours=int(rng.integers(0, 24)))
            nr = "answered" if rng.random() < (0.5 if i % 2 == 0 else 0.05) else "no_answer"
            rows.append({"event_id": f"e{i}-{k}", "event_type": "dial_attempt",
                         "lender_id": "L1", "borrower_id": f"b{i % 10}", "account_id": f"a{i % 10}",
                         "contact_point_ref": ref, "occurred_at": t.isoformat(),
                         "received_at": (t + timedelta(minutes=5)).isoformat(),
                         "payload": json.dumps({"network_response": nr, "ring_seconds": 5.0})})
    return pd.DataFrame(rows)


def test_report_end_to_end(tmp_path: Path) -> None:
    ev = _tiny_events()
    ev_path = tmp_path / "events.parquet"
    ev.to_parquet(ev_path)
    as_of = utc(2026, 1, 20)
    refs = sorted(ev["contact_point_ref"].unique().tolist())
    lab = observed_labels(ev, as_of, refs, horizon_days=7)
    assert set(lab.columns) >= {"contact_point_ref", "rpc_next_7d", "censored"}
    assert lab["censored"].any()  # some refs undialled in window -> flagged censored
    y = lab[~lab["censored"]]["rpc_next_7d"].to_numpy(float)
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, len(y))
    results = {
        "label": "test",
        "config_summary": "test",
        "data_summary": "tiny test data",
        "notes": ["dialled-only"],
        "tables": {
            "discrimination": [{"model": "m", "n": len(y), "auc": M.roc_auc(y, p)}],
            "calibration": [{"model": "m", "n": len(y), "ece": M.ece(y, p)}],
            "rare_event": [],
            "decision": [{"model": "m", "rpc_per_1000_dials": 100.0}],
            "avoiding_vs_invalid": [],
            "propensity": [],
            "reliability": {"m": M.reliability_table(y, p, 5).to_dict("records")},
        },
    }
    md, js = generate_report(results, tmp_path, "test", timestamp="TESTSTAMP")
    assert md.exists() and js.exists()
    text = md.read_text()
    assert "test" in text
    for heading in ("Discrimination", "Calibration", "Rare-event", "Decision",
                    "Avoiding vs invalid", "second view"):
        assert heading in text
    assert text.count("test") >= 7
    assert json.loads(js.read_text())["label"] == "test"


def test_run_cli_end_to_end_on_fixture_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    d = tmp_path / "data"
    d.mkdir()
    _tiny_events(120).to_parquet(d / "events.parquet")
    pd.DataFrame({"contact_point_ref": [f"tcp{i:03d}" for i in range(120)],
                  "account_id": [f"a{i % 12}" for i in range(120)],
                  "is_primary": [i % 3 == 0 for i in range(120)]}).to_parquet(d / "contact_points.parquet")
    cfg = {"splits": {"step_days": 5, "n_splits": 2, "train_days": 7, "embargo_days": 1, "test_days": 5, "seed": 42},
           "labels": {"horizon_days": 5, "rpc_network_responses": ["answered"],
                      "rpc_dispositions": ["RPC"], "dead_states": ["recycled", "invalid", "switched_off_long"],
                      "reachable_states": ["valid_reachable"]},
           "incumbent": {"k_consecutive_failures": 2, "max_attempts_trace": 4, "primary_first": True, "base_score": 0.5},
           "baselines": {"account_gbm": {"n_estimators": 10, "seed": 42},
                         "contact_gbm": {"n_estimators": 10, "seed": 42}},
           "metrics": {"n_bootstrap": 20, "ci_level": 0.95, "seed": 42, "reliability_bins": 5,
                       "recycled_cost_ratio": 100, "recycled_thresholds": [0.5], "detection_k": [1, 3, 6], "rpc_top_n": 1000},
           "propensity": {"enabled": True, "min_prob": 0.05, "max_prob": 0.95},
           "report": {"label": "test", "out_dir": str(tmp_path / "reports")},
           "data": {"events": str(d / "events.parquet"), "contact_points": str(d / "contact_points.parquet"),
                    "borrowers": str(d / "borrowers.parquet"),
                    "policy_log": str(d / "policy_log.parquet")}}
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["run", "--config", str(cfg_path)])
    from src.rpc.eval import run as runmod
    runmod.main()
    out = list((tmp_path / "reports").glob("eval_*.md"))
    assert out, "report markdown not produced"
    assert "test" in out[0].read_text()


# --- Account GBM aggregation contract (mean + best line + choice-set size) ---

_BEST_RATE = 0.9
_N_ACCOUNT_CONTACTS = 3


def test_account_aggregation_carries_best_line_and_count() -> None:
    agg = AccountGBMScorer._aggregate(
        pd.DataFrame({"answer_rate": [0.1, 0.1, _BEST_RATE], "n_attempts": [5, 5, 5]}),
        pd.Series(["A0", "A0", "A0"]),
    )
    assert agg.loc["A0", "answer_rate_max"] == _BEST_RATE
    assert agg.loc["A0", "answer_rate_mean"] < _BEST_RATE
    assert agg.loc["A0", "n_contacts"] == _N_ACCOUNT_CONTACTS


def test_account_gbm_unseen_path_uses_same_aggregation() -> None:
    refs = [f"R{a}{c}" for a in range(6) for c in range(3)]
    accs = pd.Series([f"A{a}" for a in range(6) for _ in range(3)])
    feats = pd.DataFrame({"contact_point_ref": refs, "answer_rate": [0.5] * 18,
                          "n_attempts": [5] * 18})
    labels = pd.Series([1.0 if a == 0 else 0.0 for a in range(6) for _ in range(3)])
    sc = AccountGBMScorer({"n_estimators": 10, "seed": 42}).fit(feats, labels, accs)
    assert set(sc._agg_columns) >= {"answer_rate_max", "n_contacts"}
    sc.attach_context(
        pd.DataFrame({"contact_point_ref": feats["contact_point_ref"], "account_id": accs}),
        feats,
    )
    got = sc.score(utc(2026, 5, 1), feats["contact_point_ref"].tolist())
    assert got["p_rpc"].between(0.0, 1.0).all()
    assert len(got) == len(feats)
