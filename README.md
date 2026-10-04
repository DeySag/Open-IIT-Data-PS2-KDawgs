# PS2 RPC: Right-Party Contact Prediction and Skip-Trace Prioritisation

> **SIMULATION-ONLY.** There is no real CreditNirvana data and no usable public
> dataset. Everything in this repo trains and validates on a synthetic
> simulator. Every number derived from simulated data is labelled as such —
> nothing here implies real-world lift. See `docs/assumptions.md`.

For every phone number and address on file, predict how likely it is to reach
the borrower (right-party contact), then decide per account whether to keep
trying, switch contact point, switch channel, or trigger a skip-trace — with
skip-traces ranked by expected incremental recovery net of cost. Compliance
(RBI Fair Practices Code, DPDP) is enforced as hard guardrails, not model
output.

## Checkpoint status

All 8 workstream PRs are merged into `main`. Full suite: **139 passed,
1 skipped, 0 failed** (140 tests), plus the end-to-end smoke test
(event in → decision out) passes.

| Layer | State |
|---|---|
| Contracts | Frozen (`src/rpc/contracts/__init__.py`); change needs coordinator approval |
| Simulator | v1 with hidden states, avoidance latent, incumbent policy; dev 5k borrowers / full 100k |
| Ingestion | Mapping-driven adapter + DuckDB event store, idempotent, PII-hashing |
| Features | 182+1 point-in-time features, `EventSource` abstraction, registry == output enforced |
| Models | Joint 12-state HMM state tracker + borrower latent; 3 baselines (incumbent, account GBM, contact GBM) |
| Decision | Guardrails-first, 4 actions, VOI-ranked trace queue |
| Serving | FastAPI `/v1/*`, compliance fast path, stale fallback, suppression list |
| Eval | Rolling-origin harness, bootstrap CIs, calibration, leakage tripwires |

**Known gaps (not hidden):** full-scale (5M-event) ingest is projected (~4.3 min), not yet measured; `make train-baselines` / `make train-models` reference trainer modules that don't exist yet — baselines are trained inside the eval harness instead; `configs/serve.yaml` doesn't exist so serving runs on in-code defaults; root `.gitignore` patterns `eval/` and `models/` also match `src/rpc/eval` and `src/rpc/models` (files were force-added; scoping them to `/eval/` `/models/` needs coordinator approval).

## Architecture

```
CN extracts ──► 1. Ingest (mapping YAMLs → canonical InputEvent → DuckDB event store)
                        │
                        ▼
              2. Features (EventSource, received_at <= as_of, 182+1 PIT columns)
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   3a. State tracker (HMM+latent)  3b. Baselines (incumbent / GBMs)
            │                       │
            └───────────┬───────────┘
                        ▼ ContactPointScore {p_rpc, 7-state posterior, recycled_risk, confidence}
              4. Decision (guardrails restrict → action ×4 + reason code → VOI trace rank)
                        │
                        ▼
   5. Serve (dialer lists │ campaign decisions │ trace queue │ suppression │ visit planner)
                        │
                        ▼
              6. Eval (rolling-origin, calibration, leakage guards) + feedback/monitoring (stub)
```

Repository layout:

```
.
├── configs/                  # All tunable parameters (see Configuration)
│   └── field_mappings/       # Per-source CN→canonical mappings (cn_dialer_csv, cn_disposition_ndjson, simulator)
├── data/                     # Gitignored generated datasets (events/borrowers/contact_points/policy_log/ground_truth.parquet)
├── docs/                     # assumptions, decision_log, api, decision, eval, features, ingest, state_tracker, diagnosis_report
├── src/rpc/
│   ├── contracts/            # FROZEN pydantic schemas (single __init__.py)
│   ├── sim/                  # generate.py — the simulator / mock CN
│   ├── ingest/               # adapter.py, mapping.py, normalize.py, store.py
│   ├── features/             # build.py, features.py, spec.py, source.py, text.py, labels.py
│   ├── models/               # state_tracker/, baselines/, types.py (frozen ContactPointScore)
│   ├── decision/             # guardrails.py, actions.py, voi.py, reason_codes.py, engine.py, types.py, stubs.py
│   ├── serve/                # app.py, interfaces.py, fast_path.py, suppression.py, config.py, smoke_test.py
│   └── eval/                 # run.py, splits.py, labels.py, metrics.py, protocol.py, registry.py, report.py, ...
└── tests/                    # test_contracts/decision/eval/features/ingest/serve/state_tracker (140 tests)
```

## Key domain concepts

- **Right-party contact (RPC):** reaching the borrower, not someone else. The core split is *avoiding vs invalid* — a borrower who won't answer and a dead number look identical ("no answer") but need opposite actions (switch channel vs trace). A borrower-level reachability latent shared across that borrower's contact points separates them.
- **Exactly four actions:** `continue` | `switch_contact_point` | `switch_channel` | `trace`. Retry-with-backoff and change-visit-time are *parameters* of `continue`. Recycled numbers resolve via the suppression list, never an action.
- **Guardrails first, restrict-only.** Contact hours (8–19 IST), frequency caps, consent/DND/disputes/deceased flags, and the suppression list are evaluated before any model output. Models can only narrow the action set, never widen it.
- **Trace VOI (incremental, not gross):** `VOI = P(find) × [P(recovery|reached) − P(recovery|¬reached)] × recoverable − trace_cost − collection − goodwill`. Ranked by VOI per rupee under a portfolio budget. Self-cure accounts gain nothing from a trace.
- **Point-in-time (PIT) correctness:** no feature may use information after the score timestamp (`received_at <= as_of`); splits are rolling-origin with embargo, never random k-fold.
- **Hidden ground truth is never a feature.** True contact-point state lives in a separate table used only for evaluation (and its columns are dropped on ingestion if present).
- **Suppression is one-way:** recycled/third-party additions are immediate (fast path: within the intake request); removals need CN sign-off. No delete path exists.

## Quick start (verified commands)

```bash
pip install -e ".[dev]"

# Generate dev data (~5k borrowers, 30 days, ~414k events, ~33 MB)
python -m src.rpc.sim.generate --config configs/sim.yaml --scale dev --output data/dev.parquet
# NOTE: artefacts land beside --output as data/events.parquet, borrowers.parquet,
# contact_points.parquet, policy_log.parquet, ground_truth.parquet

# Build point-in-time features
python -m src.rpc.features.build --scale dev --out data/features_dev.parquet
# or: make features-dev

# Run evaluation (rolling-origin, baselines, bootstrap CIs, markdown+JSON report)
python -m src.rpc.eval.run --config configs/eval.yaml
# or: make eval

# End-to-end smoke test (event in, decision out)
python -m src.rpc.serve.smoke_test
# or: make smoke

# Serve the API locally (port 8000, in-code defaults; no configs/serve.yaml yet)
python -m uvicorn src.rpc.serve.app:app --host 0.0.0.0 --port 8000
# or: make serve

# Full test suite / lint / typecheck
pytest tests -q        # or: make test   (140 tests, ~1-2 min)
ruff check src tests   # or: make lint
mypy src               # or: make typecheck
```

`make train-baselines` and `make train-models` are wired in the Makefile but their
trainer modules (`src/rpc/models/train*.py`) are not implemented yet — baseline
training currently happens inside `src/rpc/eval/run.py`.

## Module guide

### Contracts — `src/rpc/contracts/__init__.py` (FROZEN, single file)
Pydantic schemas with `extra="forbid"`. **Input envelope** `InputEvent`
(`event_id`, `event_type` ×6, `lender/borrower/account_id`,
`contact_point_ref` hash, `occurred_at`/`received_at`, typed `payload` ×6:
dial attempt, disposition, bot transcript, field visit, contact-point update,
payment). **Output** `OutputDecision` (4 actions, 11 reason codes,
ranked contact points with state posteriors, trace VOI, flags) and
`SuppressionEntry` (recycled/third-party, evidence, `removal_requires:
cn_signoff`). Change only with coordinator approval.

### Simulator — `src/rpc/sim/generate.py`
Generative world with 6-state phone chains + borrower avoidance overlay,
emission/disposition/ring matrices, recycling dormancy, shared contacts,
and a rule-based incumbent dialer policy that censors event generation.
`python -m src.rpc.sim.generate --config configs/sim.yaml --scale {dev|full} --output <path>`
(`dev` forces 30 days; full = 100k borrowers / 180 days / ~5M events).
Writes `events.parquet` (canonical envelope) plus `borrowers`,
`contact_points`, `policy_log`, and the separate hidden `ground_truth.parquet`.
All parameters in `configs/sim.yaml` are assumptions for CN/SMEs to confirm.

### Ingestion — `src/rpc/ingest/`
Because CN's real format is unknown, translation is data, not code: YAML
mappings in `configs/field_mappings/` (`cn_dialer_csv`, `cn_disposition_ndjson`,
`simulator`) define field renames, nested paths, value maps, timestamp
formats/timezones, and validation knobs.
`ingest(batch_or_path, source) -> {accepted, duplicate, rejected, dirty_marked}`;
`read_events(received_before, event_types, lender_id, contact_point_refs)` is
the downstream (feature-store) entry point; `replay(path, source)` re-ingests
idempotently. Phones/addresses are normalised and sha256-hashed (16 hex);
raw values never reach the store, dead-letter rows, or logs. DuckDB store:
`events` (dedupe on `event_id`, keep earliest `received_at`), `dead_letter`
(redacted, `row_hash`-keyed), `dirty_contact_points` (late vs scoring
watermarks), `watermarks`. Measured: 414k dev rows in ~21 s (~19.5k rows/s).
Details: `docs/ingest.md`.

### Features — `src/rpc/features/`
182 (+1 conditional) point-in-time features via `build_features(as_of, source)`
over the `EventSource` abstraction (`ParquetEventSource` / `DataFrameEventSource`;
only `received_at <= as_of` is visible). Groups: telephony per window
(1/3/7/14/30d) and slot, dispositions, text cues (v0 keywords, Hinglish),
bot transcripts, shared-contact graph counts (lender-local union-find, no GNN),
record history, borrower cross-line signals, field/address, account/VOI inputs,
calendar. Registry in `spec.py` is the single source of truth (registry ==
output is test-enforced; `docs/features.md` is partly auto-generated).
`configs/features.yaml` + `configs/text_patterns.yaml`. Details: `docs/features.md`.

### Models — `src/rpc/models/`
- `state_tracker/`: joint 12-state (line-state × avoidance) HMM filter with
  borrower sharing (resets, naive-Bayes pooling, silence explained-away),
  hard-evidence overrides, events-only EM, calibrated outputs as frozen
  `ContactPointScore` (`types.py`). Priors in `configs/state_tracker.yaml`
  (deliberately not copied from `sim.yaml` — leakage guard).
- `baselines/`: (a) incumbent attempt-count rules, (b) account-level GBM,
  (c) per-contact-point GBM — every model must beat all three.
Details: `docs/state_tracker.md`.

### Decision — `src/rpc/decision/`
`evaluate_guardrails` (restrict-only) → exclusions (recycled-risk cutoff from
cost ratio 1:100, third-party restrict) → state-driven action mapping →
`compute_voi` → threshold/deferral-gated `trace`, else parked `continue` variant
— never a dead end. Reason codes: 11 frozen contract codes plus internal
`GUARDRAIL_*`/deferral codes coerced at the boundary (contracts untouched).
`configs/guardrails.yaml` + `configs/costs.yaml` (all ₹ values are assumptions).
Details: `docs/decision.md` (includes a worked VOI example).

### Serving — `src/rpc/serve/` (all `/v1`, see `docs/api.md`)

| Endpoint | Purpose |
|---|---|
| `POST /v1/events` | Batch intake → ingest counts + synchronous recycled/third-party fast-path suppression (before the response returns) |
| `GET /v1/dial-lists` | Per-account contact points ordered by health; all-excluded accounts still get a fallback action |
| `GET /v1/decisions`, `POST /v1/score` | Batch (paginated) / single-account decisions with reason codes and expiry |
| `GET /v1/trace-queue` | VOI-per-rupee ranked skip-trace queue under a budget |
| `GET /v1/suppression`, `POST /v1/suppression/removal-requests` | Append-only suppression list with version diffs; removals are `pending_cn_signoff` requests — no delete |
| `GET /v1/visit-candidates` | Address health from field-visit outcomes (design stub depth) |
| `GET /v1/health` | Liveness + data freshness/staleness state |

Built on `EventStore`/`Scorer`/`Decider` Protocols with in-memory stubs
(`create_app(...)` injection); per-lender isolation via `X-Lender-Id`
(optional `X-API-Key`); stale scores decay toward conservative defaults and
never block the dialer. The remaining integration step (flagged by the serve
team): adapt the DataFrame-shaped real event store behind the
`ingest(batch, source)` Protocol.

### Evaluation — `src/rpc/eval/`
`python -m src.rpc.eval.run --config configs/eval.yaml`: rolling-origin
splits (train 14d, embargo 3d, test 7d — never random k-fold), observed labels
(dialled-only, censored excluded) vs oracle labels (hidden truth, harness-only),
AUC/PR/calibration-by-segment/ECE, rare-event precision-recall at operating
points with cost-weighted loss, RPC-per-1000, avoiding-vs-invalid slices,
propensity-weighted second view, bootstrap CIs. Reports to
`reports/eval_<timestamp>.md/.json`, stamped SIMULATION-ONLY. Only `eval/` may
read ground truth and policy logs (test-enforced tripwire). Details: `docs/eval.md`.

## Configuration

| File | Owns |
|---|---|
| `configs/sim.yaml` | Simulator world (transitions, emissions, policy, noise, seed=42) — all assumptions |
| `configs/features.yaml`, `configs/text_patterns.yaml` | Feature windows, slots, thresholds, Hinglish cue patterns |
| `configs/state_tracker.yaml` | HMM priors/transitions/emissions, EM settings (leakage-free by construction) |
| `configs/guardrails.yaml` | Contact hours, frequency caps, suppression rules, decision thresholds |
| `configs/costs.yaml` | Channel/trace costs, recovery curves, VOI budget (all ₹ values assumed) |
| `configs/eval.yaml` | Splits, horizons, bootstrap, incumbent params |
| `configs/field_mappings/*.yaml` | Per-source CN→canonical translation + validation knobs |

## Testing

`pytest tests -q` — 140 tests:

| File | n | Scope |
|---|---|---|
| `test_contracts.py` | 6 | Schema validation/serialization, reason-code completeness |
| `test_decision.py` | 30 | Guardrails restrict-only, hard-blocks, action mapping, hand-checked VOI, IST hours, caps |
| `test_eval.py` | 10 | Split embargo, hand-computed metrics, leakage guards, CLI end-to-end |
| `test_features.py` | 32 | PIT leakage, late arrival, dedupe, micro-fixture values, lender-local graph, determinism |
| `test_ingest.py` | 21 | Mappings, idempotent replay, dedupe-earliest, dirty/late, rejections, hashing/PII, tenancy |
| `test_serve.py` | 28 | Endpoint contracts, fast-path rules, stale fallback, suppression one-way, tenancy, smoke |
| `test_state_tracker.py` | 13 | Posteriors, PIT, overrides, cross-line effects, EM, save/load, registry |

Plus `make smoke` (live FastAPI end-to-end) and property-style tests for
idempotent replay and out-of-order events in `test_ingest.py`.

## Documentation

- `docs/assumptions.md` — A1–A6 working assumptions + simulator parameters awaiting CN/SME confirmation
- `docs/decision_log.md` — dated decisions with reasons and rejected alternatives, per workstream
- `docs/api.md` — serving contract: endpoints, envelopes, fast-path rules, stale policy
- `docs/decision.md` — decision pipeline, reason-code table, worked VOI example
- `docs/eval.md` — harness, metrics, leakage rules, current data status
- `docs/features.md` — pipeline semantics, null conventions, auto-generated registry
- `docs/ingest.md` — mappings, validation, store schema, measured performance
- `docs/state_tracker.md` — HMM design, sharing mechanisms, verification
- `docs/diagnosis_report.md` — simulator-output bug diagnosis (informational)
- `System Prompt.md` — standing team brief (local-only, gitignored; not in the repo)

## Contributing (day-1 rules)

- Contracts, the four-action set, and guardrail semantics are frozen — coordinator approval required.
- Config over constants; deterministic by default (fixed seeds, versioned configs).
- Small commits, `area: what and why`. No notebooks as source of truth; data/models stay out of git.
- Never present simulated numbers as business impact; label everything simulation-only.
- No PII beyond need: hash contact points, redact third-party names, per-lender isolation.
