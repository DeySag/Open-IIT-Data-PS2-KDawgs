# Build Guidelines — completing the PS2 pipeline

Read this file before implementing any pipeline work. It translates the
standing brief (`System Prompt.md`), the data reality
(`docs/dataset_audit.md`), and the current repo state into a build order,
hard rules, and a definition of done. If any instruction here conflicts with
the brief, the brief wins — stop and ask the coordinator (Sagnik).

## 1. The target

A complete pipeline means one clean chain on the **issued extracts**
(`Dataset/`, see audit §1 for inventory):

```
make data  ->  features  ->  train  ->  eval  ->  smoke  ->  daily batches
(mapped        (PIT,         (baselines +   (full report   (event in,
 canonical     registry-     state-tracker  on official +  decision out)
 events)       governed)     EM + cal.)     purged splits)
```

"Sensible outputs" means the five serving consumers each get contract-valid
output (`System Prompt.md` §8) with sane content:

- **Dialer:** contact points ordered by health with score + confidence.
- **Campaign engine:** exactly one of `continue | switch_contact_point |
  switch_channel | trace`, plus params, a stable reason code, and expiry.
- **Skip-trace queue:** ranked by VOI per rupee with cost (observed
  `cost_inr`, never the old flat-cost assumption).
- **Compliance:** suppression additions (recycled/third-party) with
  provenance; removals need CN sign-off.
- **Visit planner:** address health (thin stub acceptable per scope).

Content sanity (assert these, don't eyeball them): scores in [0,1];
probabilities calibrated per segment before they touch VOI; action mix has no
degenerate collapse (e.g. everything `trace`); trace ranking correlates with
incremental value, not raw outstanding; every number labelled with its data
grade (issued-extract results are NOT real-world performance — the README
states the data is invented).

## 2. Build order (thin end-to-end first)

Do stages in order. Each stage's exit criterion gates the next. Prefer a
working crude path over a polished isolated component.

| # | Stage | Exit criterion |
|---|---|---|
| 1 | **Mapping** — wire the `configs/field_mappings/cn_*.yaml` tables through ingest per audit §13: dials → `dial_attempt` + `disposition` companions; phones/addresses → `contact_point_update`; payments → `payment`; visits → `field_visit`; traces → trace history for VOI only (never a predictor); verified rows → held-out gold only | `make data` loads; enum coverage validated; 24 pattern-breaking rows quarantined; unknown enums reject, never crash |
| 2 | **Labels** — implement the sanctioned target set (audit §5): per-attempt RPC = `answered AND disposition LIKE 'rpc_*'` (strict variant minus hung_up/refused as sensitivity); per-(account,phone) aggregation, fixed horizon + embargo + censoring | Label module + leakage test green; untested links censored, never negative; coordinator sign-off on the rpc_* set recorded in `docs/decision_log.md` |
| 3 | **Baselines on real extracts** — retrain incumbent / account GBM / contact GBM via `run.py` on official + purged splits | Baselines score end-to-end; the bar is set before any real model claims anything |
| 4 | **State tracker refit** — EM on issued extracts at **account grain** (no borrower_id exists), register via `try_register_eval`, beat the baselines on record | Tracker in the comparison tables; avoiding-vs-invalid tables populated; priors-vs-data influence documented |
| 5 | **Calibration + propensity** — per-segment calibration on a later split (no naive time-pooling; regime is non-stationary); validate 1/k on the random arm before any IPS weight is trusted | ECE down per segment; IPW second view computed, not skipped |
| 6 | **Text + graph features** — re-validate cue lists on issued remarks (Hindi/Kannada/English; never reuse older patterns); redact third-party names *before* featuring; graph by **id-degree, never mask-degree**; high-precision third-party classifier (risk + audit trail, never auto-decision) | Feature registry == output; remark pipeline redaction-tested |
| 7 | **Decision on real outputs** — guardrails first (with `lender_overrides`), per-lender costs, VOI with observed costs + calibrated probabilities; fast-path suppression wiring; suppression reversible | Action mix sane; VOI ranking validated against trace outcomes; no dead-end accounts |
| 8 | **Serve + smoke** — FastAPI app, daily-batch outputs per contract, fail-safe defaults (scores decay, last suppression list holds, never block the dialer) | `make smoke` green on real extracts |
| 9 | **Report + design doc** — full `make eval` report on official splits; record assumptions, ask-CN answers, sensitivity/ablations | Demo-ready; every table labelled with data grade |

Blocked-by-design items (do NOT build around them silently — log and escalate):
slot model waits on the timezone ask (IST vs UTC shifts every slot feature);
bot-transcript features wait on transcript content; consent/DND guardrails
stay unevaluable until source columns exist.

## 3. Hard rules (violations fail review)

**Compliance and safety**
- Debt never disclosed to a third party. Recycled/third-party risk resolves
  toward suppression, always.
- Guardrails are hard rules evaluated first. Models narrow the action set,
  never widen it. Exactly four actions; backoff/visit-time are *parameters*
  of `continue`, not actions.
- Suppression is one-way: immediate add, removal needs evidence + reason +
  CN sign-off. Never explore on suppressed/recycled-risk/third-party numbers.
- Recycled module never auto-decides: no threshold, no `decide` method. The
  cutoff lives in the decision layer's cost-ratio rule, and stays reversible
  (recycled status is unsupervised — proxies only).
- No PII beyond need: contact ref = **hash of `phone_id`/`address_id`**
  (masks are unhashable; number normalisation reserved for full-number
  extracts). Redact names from remarks before featuring. Per-lender isolation
  by default. `datasets/` and `reports/` are gitignored — never commit them,
  never paste raw rows into prompts, chats, fixtures, or tickets.

**Data and modelling integrity**
- Point-in-time, always: `received_at <= as_of` (here `received == occurred`
  is a *flagged assumption* — late-event logic is unit-tested synthetically,
  not exercised on this data). Every feature pipeline change ships with its
  leakage test green.
- Hidden/verified data is never a feature: `verified_status`, membership in
  the verified set, post-window payments, trace results/new IDs, same-row
  outcomes, remark future-PTP substrings. Eval-only, under embargo.
- Account snapshot numerics (`dpd/bucket/emi/overdue/outstanding/salary/
  ability/prev_ptp/paid_other/bounce`) are **quarantined**: pipeline must run
  with AND without them until their as-of is confirmed. Per-event `agent_id`
  excluded (policy confound). `ptp_id` dangles — ignore, never join.
- Splits: official file = secondary sanity only. Primary validation is
  time-purged + embargoed + group-contained at **account grain**; random arm
  is the unbiased slice. Never random k-fold.
- Calibration over AUC alone; report per segment/bucket/time. Rare states
  (invalid n=88, verified-invalid n=3) get weak supervision + cost weighting,
  never resampling that breaks PIT. Dispositions are noisy evidence, never
  ground truth. Track coverage/orphaned accounts and net recovery per rupee
  alongside RPC (avoid Goodhart).
- Selection bias is first-class: rule arm propensity is 1.0 by design;
  random-arm propensities validate 1/k; within-account concentration (median
  max-phone share 0.92) and underexposed reference/employer/bureau links mean
  thin histories are policy artefacts, not validity signals.

**Service behaviour**
- Idempotent ingestion on `event_id`; state rebuildable by replay; every
  output carries `model_version` + `feature_snapshot_id` with an audit log.
- Fail-safe: stale/down service degrades to conservative defaults, never
  blocks the dialer. No dead ends: exhausted phone routes to channel, field,
  or trace — never no-action.

## 4. Per-task recipe

1. **Read before writing.** Inspect the module, its configs, its tests, and
   `docs/decision_log.md` first. Respect workstream boundaries — touch another
   module only if the integration forces it, and say so in the log.
2. **Plan briefly, then build.** State files-to-touch + tests-to-add up front.
   Frozen contracts, guardrail semantics, and the four-action set change only
   with explicit coordinator approval.
3. **Code standards:** typed Python, pydantic on all contract objects,
   docstrings with units + assumptions. Config over constants (costs,
   thresholds, simulator/mapping params, guardrails live in `configs/`).
   Deterministic by default: fixed seeds, versioned configs.
4. **Tests required:** unit (guardrails, VOI, action mapping, adapters);
   property (idempotent replay, out-of-order events); leakage (feature
   pipeline); contract (schemas); end-to-end smoke (event in, decision out).
5. **Log it:** `docs/decision_log.md` gets date, decision, reason,
   alternatives — plus every assumption with its owner (CN/SME/coordinator).

## 5. Definition of done

- Code typed and tested; new behaviour covered, full suite green
  (`python -m pytest tests -q`); lint clean if available.
- Configs, not constants; seeds fixed; `make data/train/eval/test` reproduce.
- PIT and leakage implications considered and tested; no new tripwire
  violations (keep the leakage tests strict — fix code, not tests).
- Decisions and assumptions logged; coordinator sign-off recorded where the
  audit requires it (rpc_* set, grain/horizon/embargo, gold scope, split
  policy).
- Outputs contract-valid with sane content per §1; smoke test passes on real
  extracts; results labelled with data grade.

## 6. When to stop and ask

One clarifying question at a time, only when the answer changes what gets
built. Always stop for: frozen-contract changes, guardrail semantic changes,
new actions, weakening any leakage/compliance test, spending choices that
trade scope between the three protected bets (avoiding-vs-invalid, recycled
safety, incremental VOI), and any ask-CN item in `docs/ask_cn.md` whose answer
arrives mid-build (record it; never silently change an assumption).
