"""Train driver for the add-on models (slot, third-party, trace-outcome).

Fits all three on TRAIN-window evidence only (``--fit-cap``, default the P4
cap 2026-05-26) and persists configs + params + fit records under
``artifacts/<model>/`` (P4 pattern; ``/models/`` is gitignored). No eval,
no decision wiring — integration adapters live with each module.

Outputs per model: ``config.yaml`` (effective params), ``params.json``
(fitted rates/priors/columns), ``fit_record.json`` (window, counts,
base rates). Deterministic: fixed seeds, no sampling.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.rpc.models.slot import SlotRPCModel, load_slot_params
from src.rpc.models.third_party import ThirdPartyRiskScorer, load_third_party_params
from src.rpc.models.uplift import TraceOutcomeModel, load_uplift_params, trace_outcome_labels

TP_DISPOSITIONS = ("third_party_contact", "third_party_ptp")


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def train_slot(dials: pd.DataFrame, cap: pd.Timestamp, out: Path) -> dict:
    frame = dials[dials["ts"] <= cap].copy()
    frame = frame.rename(columns={"attempt_ts": "occurred_at"})
    model = SlotRPCModel().fit(frame, cap)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("configs/slot.yaml", out / "config.yaml")
    _dump(out / "params.json", model.to_dict())
    record = {"window_end": str(cap), "n_attempts": len(frame),
              "global_rate": model.to_dict()["global_rate"],
              "note": "dialled attempts only; IST bins pending CN ask #1"}
    _dump(out / "fit_record.json", record)
    return record


def train_third_party(dials: pd.DataFrame, phones: pd.DataFrame, cap: pd.Timestamp,
                      out: Path) -> dict:
    past = dials[dials["ts"] <= cap].copy()
    past["tp"] = past["disposition"].isin(TP_DISPOSITIONS).astype(int)
    link = past.groupby(["account_id", "phone_id"])["tp"].max().reset_index()
    link = link.rename(columns={"tp": "tp_ever"})
    src = phones[["account_id", "phone_id", "source"]].drop_duplicates()
    link = link.merge(src, on=["account_id", "phone_id"], how="left")
    link["source"] = link["source"].fillna("KYC")
    model = ThirdPartyRiskScorer().fit(link)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("configs/third_party.yaml", out / "config.yaml")
    _dump(out / "params.json", model.to_dict())
    record = {"window_end": str(cap), "n_links": len(link),
              "global_prior": model.to_dict()["global_prior"],
              "note": "weak labels (tp disposition in window); true status UNKNOWN"}
    _dump(out / "fit_record.json", record)
    return record


def _pre_trace_features(dials: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    dials = dials.sort_values("ts")
    rows = []
    for r in traces.itertuples():
        past = dials[(dials["account_id"] == r.account_id) & (dials["ts"] <= r.trace_date)]
        ans = (past["network_response"] == "answered").sum()
        n = len(past)
        rows.append({"trace_id": r.trace_id, "n_attempts": n, "n_answered": ans,
                     "answer_rate": float(ans / n) if n else 0.0,
                     "consec_failures": int((past["network_response"] != "answered")[::-1]
                                            .cumprod().sum()) if n else 0})
    return pd.DataFrame(rows)


def train_uplift(traces: pd.DataFrame, payments: pd.DataFrame, dials: pd.DataFrame,
                 cap: pd.Timestamp, out: Path) -> dict:
    window_days = load_uplift_params()["window_days"]
    tr = traces[traces["trace_date"] <= cap].copy()
    labels = trace_outcome_labels(tr, payments, window_days=window_days)
    feats = _pre_trace_features(dials, tr)
    model = TraceOutcomeModel().fit(feats, labels)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("configs/uplift.yaml", out / "config.yaml")
    _dump(out / "params.json", model.to_dict())
    unc = labels[~labels["censored"]]
    record = {"window_end": str(cap), "n_traces": len(tr),
              "n_uncensored": len(unc),
              "paid_rate": float(unc["paid"].mean()) if len(unc) else 0.0,
              "note": "trace-conditional only; NOT causal uplift (no control arm)"}
    _dump(out / "fit_record.json", record)
    # Persist the fitted booster (LightGBM text format; sklearn fallback stays in-memory only).
    try:
        booster = model._model.booster_
        booster.save_model(str(out / "model.txt"))
        record["booster"] = "model.txt"
        _dump(out / "fit_record.json", record)
    except Exception:
        record["booster"] = "in-memory only (fallback estimator)"
        _dump(out / "fit_record.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--fit-cap", default="2026-05-26")
    args = ap.parse_args(argv)
    ds, root = Path(args.datasets), Path(args.out)
    cap = pd.Timestamp(args.fit_cap, tz="UTC")

    dials = pd.read_csv(ds / "dial_attempts.csv", dtype="string")
    dials["ts"] = pd.to_datetime(dials["attempt_ts"], utc=True)
    phones = pd.read_csv(ds / "phones.csv", dtype="string")
    traces = pd.read_csv(ds / "skip_traces.csv", dtype="string")
    traces["trace_date"] = pd.to_datetime(traces["trace_date"], utc=True)
    payments = pd.read_csv(ds / "payments.csv", dtype="string")

    _ = load_slot_params(), load_third_party_params(), load_uplift_params()
    s = train_slot(dials, cap, root / "slot")
    t = train_third_party(dials, phones, cap, root / "third_party")
    u = train_uplift(traces, payments, dials, cap, root / "trace_outcome")
    sys.stdout.write(json.dumps({"slot": s, "third_party": t, "trace_outcome": u},
                                indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
