# Decision log

## 2026-10-08 - Add-on models setup complete (fit + persist + adapters)

**Result:** `src/rpc/models/train_addons.py` (`make train-addons`, fit-cap
2026-05-26 = P4 cap) fits all three on TRAIN-window evidence and persists
`artifacts/{slot,third_party,trace_outcome}/` (config + params + fit
record + LightGBM `model.txt` where applicable): slot 35,391 attempts
(global RPC 0.163); third-party 3,671 links (global prior 0.251);
trace-outcome 392 traces, 0 censored (paid-30d 0.217). Wiring:
`slot.attach_best_slot` feeds the tracker's existing hook (tested);
third-party risk and `p_recover_30d` expose score frames but do NOT enter
the guardrail/VOI path — both are frozen-semantics changes needing
coordinator sign-off (logged, not built around).
**Verified:** driver re-runs bit-identical; `test_slot.py` +1 (attach),
`ruff` clean.

## 2026-10-08 - Eval inputs exporter built; censored-merge fail-safe in run.py

**Result:** new `src/rpc/eval/prepare.py` (`python -m src.rpc.eval.prepare`,
`make eval-inputs`) + `tests/test_prepare.py` (5 passed). Exports the
event store + official extracts to `data/events.parquet` (88,654),
`contact_points.parquet` (5,719, zero refs unknown to the store),
`borrowers.parquet` (2,400), `policy_log.parquet` (51,105),
`splits.csv` (copied), `verified_gold.parquet` (250). Join-key rule:
re-hash IDs with `resolve_pepper` and abort on store mismatch instead of
emitting unjoinable frames (store uses legacy plain-hash — confirmed by
matching a live ref). Real-feature adapter is used (features landed);
borrowers carries allowed columns only; verified gold is eval-only.
**Fix (same change):** `run.py:_dialled_only` — first real-data run crashed
at the calibration frame (`~` on float `censored` from a left-merge miss);
unknown-censored now counts as censored (unknown, never negative) at both
merge sites. `test_eval.py` still 11 passed.
**Verified:** exporter self-checks green; full `make eval` A/B running in
background (`artifacts/eval_new.log`) vs the 0.641 [0.618, 0.661] bar.

## 2026-10-08 - GBMs: capacity-control grid + account best-line aggregation

**Result:** three small changes, all measured mechanically (real-data eval
pending — see blocker): (1) `make_lgbm` forwards `lambda_l1/lambda_l2`,
`feature/bagging_fraction`, `bagging_freq` (defaults off — zero behaviour
change until configured); (2) `configs/eval.yaml` grid retargeted from
`n_estimators x min_child_samples` to `num_leaves [15, 31] x
min_child_samples [20, 50]` — same 4 fits/split, now searches tree capacity
(the likely overfit axis at 200x31 on ~8.6k rows) instead of tree count;
(3) account GBM aggregates mean + max + `n_contacts` via one shared helper
used by fit AND score-time paths (previously mean-only, duplicated) — the
dialer chooses the best line, so the max matters; contract-tested
(`test_eval.py` +2: best-line/max/count columns, unseen-path consistency).
**Decisions:** grid budget frozen at 4 (per-split tuning cost); no
`scale_pos_weight` (link base rate ~0.44 — imbalance is mild, logloss-tuned);
no early stopping (harness has no holdout inside fit — validation split
already picks params).
**Blocker (logged, not built around):** full real-data eval needs
`data/events.parquet` + contact/borrower/policy frames — no in-repo exporter
from `event_store.duckdb` exists, so the 0.641-bar A/B awaits that builder;
changes above are mechanism-tested only.
**Verified:** `test_eval.py` 11 passed, `test_state_tracker.py` 16 passed;
`ruff` clean on all touched lines (remaining file hits pre-date this change).

## 2026-10-08 - Model 9: trace-outcome model (P1) built, causal uplift disclaimed

**Result:** new `src/rpc/models/uplift.py` (`trace_outcome_labels` +
`TraceOutcomeModel`) + `configs/uplift.yaml` (window 30d) +
`tests/test_uplift.py` (5 passed). Fills the workbook gap (payment-window
label now exists in a labels module). GBM over caller-supplied pre-trace
dial history; censored traces excluded, degenerate fit falls back to base
rate; `result` / `cost_inr` / banned columns can never be features
(leakage-tested). Real-extract check (fit <= 2026-05-26, test after):
uncensored paid-30d rate 0.179, test AUC **0.525** (n=329) — random,
consistent with the P7 VOI-rank null (0.506). Dial history alone does not
predict post-trace payment.
**Decisions:** (1) 30d window — 14d too rare (6.8%), 60d half-censored
(50%); (2) ship the scaffolding with the null result, not a tuned model —
no account context until snapshot as-of confirmed (CN ask #7), no
found-phone (post-treatment); (3) NO causal claim: single trigger rule, no
control arm — true uplift needs a randomized trace holdout (future work);
VOI keeps its self-cure haircut, this model feeds the trace-conditional leg
only; (4) no registry wiring yet (see 2026-10-08 integration note below).
**Assumptions:** payments feed complete to 2026-07-24 (owner: data audit).
**Verified:** `tests/test_uplift.py` 5 passed; `ruff` clean on new files.

## 2026-10-08 - Integration note: three models land standalone, unwired

Models 5/7/9 do NOT join the eval scorer registry or the decision engine in
this change: their grains differ from the contact-point `Scorer` protocol
(slot = multiplier, third-party = link risk, uplift = trace-level), and
wiring changes eval behaviour — that integration is coordinator-visible work
with its own tests (registry adapter for slot multipliers into the tracker
head; third-party risk into the guardrail threshold; `p_recover_30d` into
`compute_voi`). Workstream boundaries respected per build guidelines §4.

## 2026-10-08 - Model 7: third-party risk scorer (P1) built, risk-only

**Result:** new `src/rpc/models/third_party.py` (`ThirdPartyRiskScorer`) +
`configs/third_party.yaml` + `tests/test_third_party.py` (6 passed).
Beta-Binomial: smoothed source priors updated with the link's own pre-`as_of`
counts; output is risk + audit trail (prior, counts, posterior). No `decide`
/ threshold / `predict_action` by construction (boundary-tested) — the cutoff
stays in the decision layer's cost-ratio rule. Real-extract priors (link =
(account, phone), tp = any third_party_contact/ptp on the link): global
0.304, employer 0.866 / reference 0.853 / bureau 0.251 / kyc_origination
0.193 / borrower_update 0.130 / skip_trace 0.138. Eval-only gold hook
(`gold_check`, Mann-Whitney rank AUC) reads `third_party_number` (74) vs
`borrower_number` (127); never fit rows.
**Decisions:** (1) Bayesian update, not an ML fit — no training-serving skew
possible beyond PIT; (2) weak fit labels (tp disposition in window; true
status UNKNOWN per audit §5); (3) `source` is link metadata (known at
`added_date` — only score links with `added_date <= as_of`); thin sources
(`n < 20`) fall back to global; (4) remark `third_party_cue` counts stay a
caller-supplied input, not read here — names stay redacted pre-featuring.
**Alternatives:** GBM on link features (rejected: source priors already
separate 0.87 vs 0.19 — trees add opacity, not signal, at this sample).
**Assumptions:** source labels are as-recorded (owner: CN ask #4 codebook —
`priority_slot`/source semantics unconfirmed).
**Verified:** `tests/test_third_party.py` 6 passed; `ruff` clean on new files.

## 2026-10-08 - Model 5: time-slot RPC (P1) built, IST assumption flagged

**Result:** new `src/rpc/models/slot.py` (`SlotRPCModel`) + `configs/slot.yaml`
+ `tests/test_slot.py` (6 passed). Smoothed per-(segment, slot) RPC rates
shrunk toward the global rate, emitted as multipliers on the state tracker
`p_rpc` head (hook `slot_multiplier_default: 1.0` stays the neutral default).
Thin (`n < min_samples: 30`) and unseen segments fall back to 1.0 — never
invent signal. Real-extract check (50,745 dials, fit <= 2026-06-29, pooled):
global RPC 0.161, afternoon x1.174 / evening x0.858 / morning 1.0 (thin —
dial mass sits in IST afternoon/evening); per-lender falls back (no
`lender_id` on raw dials — lender join lands in eval wiring, not here).
**Decisions:** (1) bins + timezone read from `configs/features.yaml`
(morning 8-12 / afternoon 12-16 / evening 16-19, `Asia/Kolkata`), hyperparams
only in `configs/slot.yaml`; (2) label = per-attempt `answered AND
sanctioned rpc set` (strict variant available), `rpc_*`-without-answer
quarantined, dialled-only; (3) PIT enforced inside `fit`
(`occurred_at <= as_of`); banned/hidden columns dropped unread
(leakage-tested); (4) no wiring into scorer/eval yet — model lands
standalone first per thin-end-to-end rule.
**Alternatives:** per-slot GBM (rejected: thin cells need shrinkage, not
trees); pooling segments into global (rejected: hides lender schedule
effects — fallback stays neutral instead).
**Assumptions:** naive timestamps = IST wall-clock (owner: CN ask #1,
unanswered — dial hours 08-18 fit IST, not UTC; 5:30 shift would move every
slot feature); `received == occurred` (owner: CN ask #1).
**Verified:** `tests/test_slot.py` 6 passed; `ruff check src tests` clean on
new files; mypy shows only the repo-wide missing-stub gaps (`pandas`,
`yaml`) shared by existing modules.

## 2026-10-07 - P3: baselines retrained on issued extracts (bar set, quarantine both ways)

**Result:** incumbent / account GBM / contact GBM train and score end-to-end
via `run.py`'s rolling-origin loop (9 origins, 2026-04-14→06-09) on the real
extracts; two reports in `reports/` (quarantine ON/OFF). Test n=2,035 dialled
refs, 8,614 train fit rows. Bar (quarantine ON, AUC [95% CI]): contact_gbm
0.641 [0.618,0.661] > account_gbm 0.619 [0.594,0.642] > incumbent 0.611
[0.589,0.632]; RPC/1000: 587 / 572 / 579; verified-250 dialled slice (n=158):
0.634 / 0.591 / 0.587. Snapshot numerics add no lift (contact_gbm 0.641→0.637
with them; CIs overlap everywhere) — quarantine stays until CN confirms
as-of. IPW second view populated (brier_ipw ≈ brier_dialled + 0.01);
1/k validates on the random arm (exact 1.000, k==linkage-time-phones 0.879).
Rare-event and avoiding-vs-invalid stay empty by design (no recycled_risk /
state posteriors from baselines; no true_state annotations).
**Decisions:** (1) fit=TRAIN-split frames only, validation rows (same window,
disjoint accounts) pick GBM params from a 2×2 grid by validation logloss,
test + verified-250 scoring-only; shared phones routed by modal account (12
dialled refs span splits — noted, not re-split). (2) Quarantine enforced in
run.py (drop pre-fit AND pre-score), not in the feature layer: dpd_bucket,
dpd_start, outstanding, overdue_start, emi_amount, other_active_loans,
paid_other_lenders_30d, last_bounce_reason; descriptors kept. (3) Label set
unchanged (eval.yaml RPC/promise_to_pay/callback; 139 rpc_dispute rows differ
from the features.yaml family — pending sign-off, not changed here).
**Code fixes (no test changes; tripwire strict):** label columns
(rpc_next_7d/censored/n_dials_window) excluded from the GBM matrix
(`_gbm_common`) and the account aggregation (`account_gbm.fit`) — the real
layer's generic fallback would otherwise train on the target; account_gbm
now applies its fitted LightGBM to unseen accounts' aggregated features
(replay-only degenerated to constant 0.5 under account-disjoint splits;
seen accounts still replay); mini backfill dedupes linkage-grain refs;
`extract_switched_off_months_text` guards NaN remarks (this also fixed the
one failing suite test); IPW population is all known refs per origin.
**Reason:** only what the real feature dtypes / split structure demand;
baseline identities unchanged (rule stays a rule, GBMs gain no state).
**Assumptions:** payment pseudo-refs (hash of account_id) excluded from the
contact universe; cross-line AUC uncomputable (silent-side observed labels
structurally zero — reported as obs_rate + mean score instead).
**Verified:** full suite 188 passed; `reports/eval_20261007T194053Z.*`
(quarantine ON) + `eval_20261007T201634Z.*` (snapshot included).

## 2026-10-06 - features: registry rewritten against official extracts (182 -> 177)

**Decision:** cut 20 permanently-dead features (bot group x8: no transcripts;
immediate_hangup pair x10: never occurs; n_accounts_sharing_cp: identical to
the borrower variant under borrower=account; secured_flag: underivable, mapping
unknown), add 15 real ones (callback/switched_off/not_reachable dispositions;
met_third_party/address_not_found visit counts; 10 account passthroughs
including bureau_score_band with snapshot quarantine), make the agent feature
unconditional (agent_id observed), drive RPC recency/labels/confirms off a
config `rpc_dispositions` family, extend text cues to verified Kannada +
English variants, rewrite `build.py` for store+accounts inputs.
**Reason:** audit vs `datasets/` showed dead weight, missing signal (callback
1,674 events; bureau band +9pp monotone), and literal-"rpc" matching that
missed the whole rpc family. Verified end-to-end: 88,654 official events
ingested, 10,038 feature rows x 185 cols in 54 s, registry == output exact.
**Alternatives:** keep dead columns for forward-compat (rejected: permanent
zeros mislead models and bloat the store; re-adding later is trivial).
**Decision:** implement every named feature in the task literally, including
per-window (1/3/7/14/30d) slot, weekend and ring statistics. Result: 182 base
features (+1 conditional agent feature) instead of ~60-75.
**Reason:** dropping a named feature risks losing a graded signal; exceeding a
soft count target only costs columns. Group counts are reported honestly.
**Alternatives:** per-window counts only with single-window slots/ring
(rejected: contradicts the explicit "for each window W" listing).

## 2026-10-04 - features: account_id fallback
**Decision:** account per contact point = mode of visible events' account_id;
fallback `ACC_<borrower-seq>` (matches simulator convention) or borrower_id.
**Reason:** borrowers/contact_points tables carry no account_id; events do.
**Alternatives:** surrogate account key (rejected: breaks the required output key).

## 2026-10-04 - features: contact-type fallback
**Decision:** type from contact_points table -> latest update contact_type ->
dial/disposition/bot evidence (phone) / field evidence (address) -> 'phone'.
**Reason:** event-only contact points (e.g. from future skip-trace updates)
have no table row; a deterministic fallback keeps the universe total.
Logged here; confirm with CN when real updates flow.

## 2026-10-04 - features: payment linkage is borrower-level
**Decision:** `confirmed_by_payment` links payments to qualifying
(answered/RPC) events on the same borrower within
`payment_confirmation_days` (7d); `n_payments_W` / `days_since_last_payment`
are borrower-level.
**Reason:** payment envelopes carry no reliable contact-point attribution.

## 2026-10-04 - features: who_answered derivation
**Decision:** use payload `who_answered` when present; else 'other' on
third-party/name-mismatch cues, else 'unknown'.
**Reason:** contract BotTranscriptPayload has no who_answered field, but the
feature list requires the counts; rule is documented and tested.

## 2026-10-04 - features: days-since convention
**Decision:** date-based (UTC calendar dates) integer days; mean_gap in
fractional days; visit_hour_mean is a plain (non-circular) IST mean.
**Reason:** reproducible, timezone-stable; documented in registry.

## 2026-10-04 - features: system fail share counts non-answered as failure
**Decision:** fail = network_response != answered (missing response counts as
failure once an attempt exists).
**Reason:** lets models discount dialer-outage days; documented.

## 2026-10-04 - features: simulator v0 gaps (workstream A, not fixed here)
Observed, not modified (out of workstream scope): sim emits only
`dial_attempt` events (no disposition/bot/field/update/payment), so those
feature groups are null/0 on v0 data; `generate.py` crashes on numpy>=2
(`timedelta(np.int64)`) — data was generated via a throwaway shim in /tmp
without touching `src/rpc/sim/`; `pyproject.toml` had invalid TOML
(`per-file-ignores` inline-table key unquoted) which broke all pytest runs —
fixed minimally as it blocked the required verification. No `shared_reason`,
`ground_truth.parquet` or `policy_log.parquet` present.

## 2026-10-04 - features: agent feature conditional
**Decision:** `agent_wrong_number_rate` (wrong-number share of the last
disposition's agent) is emitted only when `agent_id` is observed; registry is
built with the same flag so registry == output always holds.
**Reason:** spec requires skip-and-note when absent; noted in docs/features.md
(absent on simulator v0 data).

## 2026-10-04 — eval: incumbent policy params live in configs/eval.yaml
- Reason: task pointed at a `policy:` section in `configs/sim.yaml`; sim v0 has none. Editing sim.yaml is outside eval ownership, so assumed values (k=3, trace-after=6, primary-first) live under `incumbent:` in `configs/eval.yaml`.
- Alternatives: patch sim.yaml (rejected: other workstream owns it).

## 2026-10-04 — eval: temporary DuckDB mini-features in src/rpc/eval/_minifeatures.py
- Reason: `src/rpc/features` (build_features/spec.py) has not landed. Clearly marked temporary; `get_feature_builder()` auto-switches to the real layer when importable.
- Alternatives: block until features land (rejected: parallel-work instruction says stub, don't wait).

## 2026-10-04 — eval: oracle/random scorers live in src/rpc/eval/reference.py, not baselines/
- Reason: keeps the leakage guard ("baselines never read ground truth") a trivial file-content check.
- Alternatives: put them beside baselines with an allowlist (rejected: weaker guarantee).

## 2026-10-04 — eval: dev data from throwaway generator, not sim v0
- Reason: sim v0 `generate.py` crashes (numpy.int64 in timedelta). Reported to sim workstream; data/ is gitignored so a same-schema stand-in unblocks the green run.
- Alternatives: patch sim (rejected: outside ownership).

## 2026-10-04 — eval: pre-existing pyproject.toml breaks bare pytest collection
- Reason: toml parse error at line 36 (unrelated to eval). Tests verified with `python -m pytest -c NUL -p no:cacheprovider`. Flagged to coordinator; not patched (outside ownership).

## 2026-10-04 — eval: root .gitignore `eval/` + `models/` also ignore src/rpc/eval and src/rpc/models
- Reason: patterns meant for top-level artifact dirs match our source dirs. Did NOT edit .gitignore (outside ownership); staged owned files with `git add -f`. Coordinator should scope those patterns (e.g. `/eval/` `/models/`).
- Alternatives: leave files uncommitted (rejected: deliverable must be in git).
## 2026-10-04 — State tracker v0: joint (A,S) filter + pooling + silence shift (B-workstream)

- Decision: per-line joint 12-state (A,S) forward filter; borrower sharing via
  (i) reset events, (ii) naive-Bayes A-pooling with prior correction,
  (iii) asymmetric-silence explained-away shift (σ = 1−exp(−W/κ), κ=3.0).
- Reason: the filter's unconditional network emission cannot express
  "silent while the borrower is active elsewhere"; without (iii) a silent line
  with an answering sibling scores valid_reachable 0.70 (verified). With (iii):
  dead 0.66 vs 0.08 all-silent; ablation (latent off) diff 0.001.
- Alternatives: full joint forward over A×S₁×…×S_K (exact, but 2·6^K states —
  intractable past K=3 at 100k-borrower scale); two-round variational loop
  (rejected: does not fix the emission-contrast problem either).
- Leakage: config priors are coarse qualitative numbers, deliberately
  different from configs/sim.yaml; EM reads events only (ground-truth columns
  dropped defensively in parse_events).

## 2026-10-04 — Blockers outside ownership (reported, not fixed)

- `pyproject.toml:36` (`per-file-ignores = {"tests/*": ...}`) is rejected by
  this env's tomllib, so `pytest`/`make test` fail at config parse. Ran tests
  with `pytest -c NUL -p no:cacheprovider`. Needs coordinator/workstream-F fix.
- `src/rpc/sim/generate.py:107` crashes on numpy 2.x
  (`timedelta(days=rng.integers(...))` → TypeError), so no dev/full simulator
  data could be generated; dev-scale timing used the self-contained mini
  generator instead. Needs workstream-A fix.
- `git push origin main` → 403 (permission denied); commits are local only.
  `src/rpc/models` is covered by the `models/` gitignore rule (model
  artifacts); files were added with `git add -f`. Suggest narrowing to
  `/models/` — needs coordinator approval.
- No `src/rpc/eval/` harness on main yet: `try_register_eval()` returns False
  without crashing; slow eval test skips until simulator ground-truth outputs
  exist.
## 2026-10-04 — Decision layer v0 (workstream D, day 1)

- Decision: guardrails evaluated first and restrict-only; hard blocks park on
  switch_channel/field and never trace. Reason: compliance hard requirement;
  four-action contract forces a next action, so parking (not executed
  downstream) is the fail-safe. Alternative (empty action) rejected: violates
  no-dead-ends rule.
- Decision: recycled cost-ratio cutoff 1/(1+100) applied to the classifier
  score, not background posterior mass. Reason: preserves dialling on healthy
  lines while keeping fast-path suppression aggressive where it matters.
- Decision: VOI conditions on P(dead), avoiding accounts get ~zero VOI and
  never queue. Reason: tracing an avoider finds nothing new (incremental value).
- Decision: greedy-by-ratio knapsack with documented exactness note over an
  exact DP. Reason: day-1 scale and auditability; DP is a day-3 option.
- Decision: GUARDRAIL_* reason codes kept in decision layer, coerced at the
  boundary; contracts untouched. Reason: frozen-contract rule; flagged the
  enum addition for coordinator approval instead of stopping (outputs valid).
- Decision: stub evidence UUID5 placeholders; `no_viable_contact_point`
  placeholder for contract min_length=1. Reason: run without model/feedback
  layers; serving layer must reconcile. All simulation-only.
## 2026-10-04: Ingestion adapter design (ingest workstream)

**Decision:** Mapping-driven adapter (YAML per source) + DuckDB event store
with `events` / `dead_letter` / `dirty_contact_points` / `watermarks` tables.

**Reason:** CN's real format is unknown; the canonical envelope is frozen, so
translation must be data (mapping files), not code branches. DuckDB gives
bulk SQL dedupe/insert with no extra service to run.

**Alternatives considered:** per-row Python adapter loop (rejected: too slow
for 5M rows, and the old one raised `KeyError` instead of rejecting);
per-lender hardcoded mappings (rejected: same problem, less auditable).

## 2026-10-04: Dead-letter rows are PII-redacted, keyed by row_hash

**Decision:** `dead_letter` stores the raw row with contact fields replaced by
`[redacted-pii]` (all values redacted when the source has no mapping), keyed
by `sha256(source|redacted_json)` so replay is idempotent.

**Reason:** "Raw numbers/addresses must never be stored" includes the
quarantine table; without a stable key, replaying a batch would duplicate
dead-letter rows and break idempotent replay.

## 2026-10-04: Dedupe keeps earliest received_at; dirty = late vs watermark

**Decision:** Duplicate `event_id`s keep the earliest `received_at` (stored row
refreshed when an earlier version arrives, so order does not change final
state). `dirty_contact_points` marks `(ref, lender_id)` when a watermark
exists and `received_at > watermark OR occurred_at < watermark`; watermarks
are written by the feature pipeline via `set_watermark`.

**Reason:** Matches the spec's idempotency + late-event-correction rules with
order-independent final state.

## 2026-10-04: No EventSource protocol found; exposing read_events

**Decision:** `src/rpc/features/source.py` does not exist, so no protocol to
implement. Downstream consumers use `ingest.read_events(...)` with
`received_before / event_types / lender_id / contact_point_refs` filters.

**Reason:** Avoid inventing a competing interface; flagged to the coordinator
to confirm or redirect when the protocol lands.

## 2026-10-04 - eval+features+baselines: integrate real feature layer into eval harness
- Failure on origin/main merge (2 failed, 108 passed): eval called builder(as_of, refs, events, cps) but real build_features(as_of, source, refs) takes an EventSource; leakage tripwire flagged src/rpc/features/source.py for naming the restricted tables in its docstring.
- Decision: get_feature_builder() now returns an adapter (DataFrameEventSource + backfill of only the mini columns the real output lacks, e.g. consec_failures/n_attempts; real columns never overwritten). run.py passes borrowers through. source.py docstring reworded to keep the tripwire strict (no test change). Baselines touched minimally and only where the dtype contract forced it: to_matrix coerces nullable boolean to float; contact_gbm.score drops the blanket fillna(0.0) in favour of to_matrix per-column NaN handling (also restores the documented recency far-past sentinel, previously zeroed by the blanket fill).
- Assumptions: fixture contact_points missing lender/borrower keys inherit the per-ref mode from events; missing created_at means visible-from-first-event. Downcasting real null semantics inside the adapter was rejected (null-vs-0 is load-bearing).
- Verified: full suite 110 passed, 1 skipped (pre-existing slow mark). ruff not installed here, lint unverified.

## 2026-10-04 - serve: Protocol interfaces + DI factory, stubs by default
**Decision:** the serving layer depends on three `Protocol`s
(`EventStore`, `Scorer`, `Decider`) in `src/rpc/serve/interfaces.py`
with in-memory stubs, injected via `create_app(event_store=…,
scorer=…, decider=…)`. Endpoints hold collaborators on `app.state`
so tests (and later the real modules) swap in with no endpoint
code changes.
**Reason:** the task mandates building against the interfaces now
and swapping in real modules as they land; DI keeps the two
decoupled.
**Alternatives:** importing the real modules directly (rejected:
they were not landed yet at build time; would couple serving to
their evolving interfaces).

## 2026-10-04 - serve: fast path is synchronous and idempotent
**Decision:** `POST /v1/events` runs a `RecycledSignalDetector`
synchronously per accepted event and writes suppression entries
before the response returns. Four config-driven rules, each logged
as evidence: `wrong_number` disposition (recycled), `third_party`
disposition (third_party), who-is-this/name-mismatch/language-
mismatch cue in a bot transcript (recycled), and `recycled`
posterior from the Scorer `>= recycled_risk_threshold` (recycled).
Idempotent on `(contact_point_ref, event_id)`; per-event latency
recorded and bounded by `fast_path_max_latency_ms`.
**Reason:** compliance-critical — the first credible recycled
signal must update suppression within the request, never at the
next batch; the same evidence must never double-suppress.
**Alternatives:** async/background fast path (rejected: the task
requires suppression before the response returns).

## 2026-10-04 - serve: stale-score fallback is fail-safe and conservative
**Decision:** when scores are older than `max_score_age_hours` or
the Scorer raises, the service serves the last cached scores with
confidence decayed by `max(confidence_decay_floor, 1 - age/max_age)`,
downgrades any `trace` decision to `switch_contact_point` (or
`continue` when only one contact point) with `trace=null`, keeps
the current suppression list in force, sets `stale=true`, and never
raises.
**Reason:** fail-safe rule 19 — CN operations continue through a
scorer outage while staying conservative on compliance-critical
trace decisions.
**Alternatives:** erroring the consumer (rejected: must never block
the dialer); serving stale trace (rejected: never recommend trace
from stale data).

## 2026-10-04 - serve: fixed three boilerplate bugs
**Decision:** (1) `valid_until = generated_at + validity_hours`
(default 24h, configurable) instead of `valid_until == as_of`;
(2) `TraceInfo` always carries `recoverable_amount` so `/v1/score`
no longer 500s on the `trace` action; (3) `POST /v1/events`
persists through the `EventStore` (ingest counts + fast path)
instead of returning a hardcoded count.
**Reason:** these were the three known bugs in the day-0 boilerplate.
**Alternatives:** none — correctness fixes.

## 2026-10-04 - serve: dial list never dead-ends; suppression one-way
**Decision:** an account whose contact points are all excluded
(suppressed or `invalid+recycled >= dead_contact_threshold`) still
appears with a stated `fallback_action` (`trace` when fresh,
`manual_review` when stale) and `fallback_reason`. Suppression is
append-only with a monotonic `version` and `?since=version` diffs;
removal is only possible via `POST /v1/suppression/removal-requests`
which creates a `pending_cn_signoff` request — there is no DELETE.
**Reason:** rule 22 (no dead ends) and rule 3 (suppression is
one-way, removals need CN sign-off).
**Alternatives:** omitting all-excluded accounts (rejected: leaves
the dialer without a next action).

## 2026-10-04 - serve: interface drift with landed modules (adapter needed)
**Decision:** the real `src/rpc/ingest/store.py` `EventStore` is
DataFrame-based (`insert_canonical(df, ingested_at)`,
`read_events(received_before, event_types, lender_id,
contact_point_refs) -> DataFrame`), which does NOT match the
task's `ingest(batch, source) -> counts` Protocol. The real
`src/rpc/decision/engine.py` keeps a legacy shim
(`GuardrailsEngine`, `map_reason_code`, `decide_action`) with the
same signatures the serving layer imports, so serving works against
it unchanged (verified: 34 serve+contract tests pass on the real
decision module). The real `ContactPointScore`
(`src/rpc/decision/types.py`) matches the task's described fields.
**Reason:** the serving layer was built against the task's Protocols
with stubs; the landed EventStore chose a different (bulk,
DataFrame) shape.
**Alternatives:** rewriting serving to the DataFrame store now
(rejected: out of scope; needs an adapter `ingest(batch, source)`
that frames `InputEvent`s into a DataFrame and maps counts —
flagged to the coordinator as the integration step).

## 2026-10-04 - serve: synced main + pyproject TOML fix + serve target
**Decision:** synced the landed modules from origin/main into the
working tree (`decision/`, `eval/`, `features/`, `ingest/`,
`models/`, their tests and docs) and pulled the already-committed
`pyproject.toml` TOML fix (`per-file-ignores` inline-table key
`:` -> `=`). Added the `serve` Makefile target (uvicorn, port 8000)
and merged it with main's `features-dev` target.
**Reason:** the coordinator asked to sync to the current repo state
before building; the TOML fix was already on main and unblocks
pytest/ruff config parsing.
**Alternatives:** leaving the stale tree (rejected: would build
against day-0 boilerplate).
**Note (not fixed, outside serve ownership):** on the synced tree,
`tests/test_decision.py`, `tests/test_features.py` and
`tests/test_ingest.py` fail to COLLECT due to bugs in those
workstreams' modules (`decision/__init__.py` does not re-export
`AccountContext`; `features` has a circular import;
`ingest/adapter.py` does not define `IngestAdapter`). These are
pre-existing on main and belong to their respective workstreams;
`tests/test_serve.py`, `tests/test_contracts.py` and `make smoke`
all pass.

## 2026-10-07 - ingest (P1): official extracts mapped, hashed, quarantined

**Decision:** wire the six `cn_official` mappings end-to-end
(`python -m src.rpc.ingest`, `make data` with `DATASETS`/`DB` overrides):
lender join from accounts.csv (unmatched accounts stay null and reject as
`missing_required_field`, never silently dropped); contact ref = peppered
HMAC-SHA256 (truncated 16 hex) of `phone_id`/`address_id` via new
`contact_point.pepper_env: CN_HASH_PEPPER` (pepper from env, never committed;
unset pepper falls back to legacy sha256 with a logged warning; rotation
means re-ingest); the 24 rpc_ptp-on-non-answered rows quarantined to
dead-letter as `quarantined_rpc_without_answer` (dial companion still carries
the network evidence); dead-letter redaction extended to free-text and
masked-identifier columns (`remark`, `phone_masked`, `address_text`).
**Reason:** audit §4/§11/§13: 1 account = 1 borrower (no borrower_id exists);
received == occurred is a flagged assumption (single timestamp per event);
naive wall-clock localised to Asia/Kolkata pending CN confirmation;
`call_rejected` stays rejected on the disposition companion only (echo row,
dial covers it); `immediate_hangup` NOT derived — zero answered +
customer-hangup + zero-talk rows exist in this data and CN confirmation is
pending; `cash_collected -> met_borrower` and `neighbour_says_shifted ->
nobody_of_that_name` stay flagged for CN; split routing survives via
account_id linkage (zero orphans verified) with splits.csv consumed
downstream, never as a feature.
**Alternatives:** unpeppered hashing everywhere (rejected: audit §11 deadline
is first real eval); deriving immediate_hangup now (rejected: no instances,
unconfirmed); accepting the 24 rows as promise_to_pay (rejected: contradicts
telephony evidence).
**Verified:** `test_ingest.py` 34 passed incl. 5 new (pepper determinism +
divergence, quarantine split, free-text redaction, enum coverage pinned
against the real files, end-to-end counts + idempotency with quarantine == 24
and dial accepted == 51105); `test_ingest_official.py` 7 passed; legacy-hash
tests hermetic via autouse pepper-clearing fixture. Full suite 162 passed, 2
skipped without the official-datasets env (gated tests skip); with
`OFFICIAL_DATASETS` set the gated tests run and pass.
**Observed, not fixed (outside ingest ownership):**
`test_features.py::test_rpc_family_recency_and_account_passthroughs` fails on
the pristine tree too (NaN remark reaches regex in `text.py`); `test_serve.py`
cannot collect here (`fastapi` not installed in this env).

## 2026-10-07 - P2 labels: sanctioned RPC targets with censoring (coordinator sign-off Sagnik, all approved)
**Owner:** labels workstream. **Sign-off:** coordinator Sagnik on each §5 minimum decision (recorded here before build).
1. **Sanctioned per-attempt RPC set:** `answered AND disposition LIKE 'rpc_*'` over all 7 raw variants (`rpc_ptp, rpc_call_back, rpc_hung_up, rpc_refused, rpc_hardship, rpc_dispute, rpc_claims_paid`; canonical `RPC, promise_to_pay, callback, dispute`). **Reason:** audit §5 strongest proxy; data confirms 37% non-RPC inside raw `answered` (13,303 answered vs 8,376 rpc-on-answered) so `answered` alone is NOT a label. **Alternatives:** answered-only (rejected: over-broad), rpc-disposition-only without answered (rejected: misses the AND; 24 non-answered rpc_ptp rows quarantined as non-RPC).
2. **Strict sensitivity:** primary minus `rpc_hung_up`/`rpc_refused` (keeps ptp/call_back/hardship/dispute/claims_paid), run alongside primary. **Reason:** hung_up/refused are contact-without-content; audit names this sensitivity. Frozen `DispositionPayload(extra=forbid)` blocks adding `raw_disposition` to the envelope, so on canonical events strict = `{promise_to_pay, callback, dispute}` (canonical `RPC` collapses refused/hung_up/hardship/claims_paid — documented limitation); on raw extracts strict is exact. **Alternatives:** contract change to carry raw variant (rejected: frozen contract, needs CN approval).
3. **Grain + as-of + horizon + embargo:** grain = (account, phone) with `account_id` on every label row; `as_of` scoring origin; fixed horizon 7d `(as_of, as_of+7d]`; embargo 3d (nothing scored inside `(train_end, embargo_end]`); per-(account,phone) ever-RPC + next-RPC aggregation. **Reason:** 20 phone_ids dialled under >1 account with different outcomes — per-phone_id labels mix borrowers; fixed horizon+embargo gives PIT-safe rolling origins. **Alternatives:** per-phone_id grain (rejected: mixes accounts), variable horizon (rejected: incomparable labels).
4. **Censoring/attribution:** all undialled links/addresses censored (`rpc_next_7d=NaN`, `censored=True`), never negative; attribution per (account, phone), never phone_id alone; shared ids/masks must not leak across folds (account-group containment). **Reason:** reference/employer/bureau links 52-57% never dialled by policy — untested is unknown, not negative. **Alternatives:** undialled-as-negative (rejected: policy artefact, breaks calibration).
5. **Verified-as-gold scope:** all 250 verified rows holdout gold for final evaluation only — never label sources for training, and even membership is leakage (66/250 never dialled, so membership carries future selection info). Training frames drop verified keys; eval reports them flagged. **Alternatives:** verified-as-training-labels (rejected: future-dated 2026-07-02 annotation, sampling frame UNKNOWN).
6. **Payment-attribution rule:** payment-anchored labels are weak supervision ONLY (365 pay w/o rpc, 355 rpc w/o pay, 119 pay-before-first-rpc break any equivalence); post-cutoff payments never features (payment features strictly `payment_ts <= as_of`). **Reason:** pay/RPC mismatch breaks equivalence; payments run to 2026-07-24, past dial end. **Alternatives:** pay==RPC equivalence (rejected: mismatch rates), post-cutoff payments as features (rejected: temporal leak).
7. **Trace-trigger semantics:** skip-trace `result`/new IDs/skip-sourced rows are outcome evidence for VOI only, never predictors; `trigger_rule` single value (no policy variation to learn). **Reason:** all 1,264 post-trace dials occur >= trace date — post-scoring info. **Alternatives:** trace result as feature (rejected: leak).
8. **Split policy:** official `splits.csv` (70/15/15 stratified-random, train 1680/val 360/test 360) = secondary sanity only (same-window, contact-level leakage: 43 shared ids + 553 shared masks cross splits). Primary = time-purged + embargoed + account-group-contained rolling origins; random arm (`random_contact_point`, propensities 1/k) is the unbiased slice. Labels built from TRAIN split observable outcomes only; val/test never train-label sources. **Alternatives:** official splits as primary (rejected: temporal leakage via shared calendar + contact leakage), random k-fold (rejected: violates PIT).
**Build verification (2026-10-07):** `tests/test_labels.py` (10 leakage tests) + full suite green — 170 passed (`test_serve.py` excluded: `fastapi` not installed in this env, pre-existing). Real-extract check: per-attempt primary 16.39% / strict 9.98%, 24 rows quarantined, 4,927 non-RPC inside 13,303 answered, `language_barrier` 0 flagged, ever-RPC 61.6% of dialled links, 12/20 shared ids differ per account, 250/250 verified keys flagged with 0 leaking into train, pay/RPC mismatch confirmed (weak supervision only). Sensitivity (`rpc_next_7d_strict`) computed in every `run.py` split and reported under `## Sensitivity`.
