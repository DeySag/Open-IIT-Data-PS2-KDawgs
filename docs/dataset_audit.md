# Unified Data Audit — official CN extracts, PS2 scope

Date: 2026-10-05. Auditors: ingest/integration workstream + coworker deep-audit.
Consolidates: `docs/dataset_audit.md` (pipeline-facing brief) and the four
deep records in `datasets/` (`01_dataset_audit.md` schema inventory,
`02_labels_and_entities.md` labels/entities, `03_leakage_and_censoring.md`
leakage/PIT, `04_ps2_eda.md` exploratory analysis). The deep files are the
evidence base and are left untouched; this document is the single committed
entry point. If they disagree, this document says so explicitly (§14).

Tag convention (adopted from the deep audit): **OBSERVED** = directly
supported by data; **INFERRED** = strongly suggested, not confirmed;
**UNKNOWN** = cannot be established. No models trained for this audit; no
rows modified.

Status note: the dataset README states everything is invented (synthetic).
Formats are realistic (masked phones, code-mixed remarks, lender IDs), so
raw values are discussed only as formats/counts and short synthetic remark
fragments are quoted solely for the codebook. `datasets/` is gitignored and
must never be committed. The deep record references companions not in the
repo (`reports/01_*.csv`, `/tmp/opencode/probe_*.py`, `/tmp/opencode/eda*.py`).

## 1. Inventory and scale

11 CSVs (PS1/PS3-only files from the dataset README are absent):

| File | Rows | Grain | Columns |
|---|---|---|---|
| `accounts.csv` | 2,400 | 1 row / account | 20 |
| `splits.csv` | 2,400 | 1 row / account | 2 (`account_id`, `split`) |
| `lenders.csv` | 6 | 1 row / lender | 4 |
| `agents.csv` | 30 | 1 row / agent | 6 (20 tele + 9 field + 1 bot) |
| `dial_attempts.csv` | 51,105 | 1 row / attempt | 16, zero nulls except `ptp_id` (94%) |
| `payments.csv` | 2,162 | 1 row / payment | 5, zero nulls |
| `addresses.csv` | 3,117 | 1 row / address | 7, zero nulls |
| `field_visits.csv` | 5,578 | 1 row / visit | 15, zero nulls except `ptp_id` (89%) |
| `phones.csv` | 5,719 | 1 row / account-number link | 7, zero nulls |
| `skip_traces.csv` | 766 | 1 row / trace | 7, `new_contact_point_id` 77% null |
| `verified_contact_points.csv` | 250 | 1 row / check | 4, zero nulls |

Coverage: 2,368/2,400 accounts dialled (32 never dialled); 4,008/5,618 phone
IDs and 4,035/5,719 (account, phone) links dialled; 1,339/2,400 accounts
visited; 1,667/2,400 accounts paid; 603/2,400 accounts traced.

## 2. Entities and join keys

- **No `borrower_id` exists in any file** (OBSERVED: all headers searched).
  "Borrower" appears only in free text and docs, never as a key. Working
  assumption must be **1 account = 1 borrower for this extract** (INFERRED,
  fragile — nothing confirms a real borrower holds no second account). The
  borrower-level reachability latent is therefore built at **account level**
  unless a borrower map is supplied.
- `account_id` is the de-facto entity key: unique in accounts (2,400), 1:1
  with splits, sole FK in dials/payments/addresses/phones/traces/visits/
  verified with zero orphans everywhere (OBSERVED).
- Grain warning: `phones.csv` is (account, phone) **linkage**, not phones —
  5,719 rows but 5,618 distinct `phone_id`. Dial attribution must follow
  (account, phone): 20 phone_ids were dialled under >1 account with different
  outcomes per account, so a per-`phone_id` label without account context
  mixes borrowers (INFERRED hazard).
- **Shared numbers are real**: 74 `phone_id`s span multiple accounts (101
  excess rows; max PH000546 ×5), 55 of them cross lenders. Sharing
  concentrates in non-KYC sources but both patterns occur. By contrast, mask
  sharing (1,106 last-4 values multi-account) is mostly distinct numbers
  colliding (INFERRED) — **id-degree is signal, mask-degree is noise** (shared
  id: 4.9% RPC vs solo 16.6%; shared mask: 16.39% vs 16.47%, i.e. nothing).
- Addresses: no `address_id` sharing; 3 texts shared across 2 accounts each —
  the address graph is essentially account-local.
- Skip-trace loop is closed: all 175 new IDs (148 phones + 27 addresses)
  resolve in phones/addresses with `source=skip_trace`, and all 1,264
  post-trace dials occur ≥ trace date (OBSERVED correct temporal order).
  `new_contact_point_id` null ⇔ `no_new_info`, perfectly.
- `ptp_id` dangles (PS1 table absent): dial PTPs only with
  rpc_ptp/third_party_ptp (3,091); visit PTPs only with
  met_borrower/met_family (605). Treat as missing, never join or impute.

## 3. Time coverage, formats, timezones

- Dial window **2026-04-01 08:01 → 2026-06-29 18:54** (~90 days, 90 distinct
  dates); visits same bounds; phones/addresses `added_date` massed on
  2026-04-01 (baseline inventory) with later rows exactly the skip-trace
  finds; traces 2026-04-13 → 06-29; payments to **2026-07-24** (150 past dial
  end); verification all **2026-07-02** (post-observation).
- All timestamps are **naive** (`2026-04-01 09:48:09`; dates bare).
  Dial hours 08:00–18:xx fit IST wall-clock and contact-hours rules, not UTC
  (which would put dialing at 13:30–23:30 IST). The deep audit treats zone as
  UNKNOWN; this document recommends localizing to **Asia/Kolkata** for mapping
  but records the conflict as **UNRESOLVED pending CN confirmation** — a 5:30
  shift moves every slot feature, so slot-model work must not assume either.
- **One timestamp per event — no `received_at`.** Assume received == occurred
  on load; late/out-of-order and dirty-marking logic cannot be exercised on
  this data (covered synthetically in unit tests).
- `checkin_ts >= start_ts` holds 100% (median lag 12.1 min). Visit `visit_date`
  is day-resolution; order by `start_ts`/`checkin_ts`. Trace `trace_date` is
  day-only, so same-day dial↔trace ordering is UNKNOWN.
- Regime is non-stationary (OBSERVED): dial volume declines weekly
  (5,045 → 2,709); `voice_bot` collapses (3,709 April → 52 May → 0 June);
  traces ramp from 04-13. Do not pool calibration across time naively.

## 4. Codebook → canonical enums

`network_response` (6 values) maps to 6 of our 7 codes: `ring_no_answer→
no_answer`, `busy_rejected→busy`, `not_reachable`, `switched_off`,
`answered`, `number_does_not_exist→does_not_exist`. Deterministic artefacts:
ring==0 exactly for the last three + does_not_exist; talk>0 only when
answered. **`immediate_hangup` never occurs** (candidate derivation:
`hangup_by=customer` with 0 talk — a decision, not a silent map).

`disposition` (18 values) is block-diagonal with network (each disposition
pairs 1:1 with one response, except 24 `rpc_ptp` rows on non-answered
networks — quarantine these edge rows). The six `rpc_*` variants
(ptp 2,937 / hung_up 2,229 / call_back 1,674 / refused 1,049 / hardship 237 /
dispute 139 / claims_paid 135) all imply contact but carry outcome subtypes:
map the family to `RPC`, keep variants in remarks. `third_party_contact`
3,537 + `third_party_ptp` 154 → `third_party`. `call_rejected` needs a mapping
decision (busy-like; paired 1:1 with `busy_rejected`). `wrong_number`,
`no_answer`, `switched_off`, `not_reachable`, `invalid_number` (88, hard
high-precision evidence), `language_barrier` (44, contact-without-content)
map directly. Standalone `promise_to_pay`/`callback`/`dispute` codes are
absent (present only inside `rpc_*`) — enum coverage must be validated.

`field_visits.outcome` (7): `locked_premises`, `met_borrower` 1,114
(address-RPC analogue), `met_family` 1,062 → `met_third_party`,
`no_such_person` 206 + `neighbour_says_shifted` 455 → moved/fabricated
evidence, `address_not_traceable` 1,400 → `address_not_found` (findability,
geocoder route), `cash_collected` 92 (visit payment — link to payments,
currently unlinked; only 47/92 overlap payment accounts).

`verified_status`: borrower_number 127 / third_party_number 74 /
not_borrower_number 27 / switched_off 19 / invalid_number 3 — annotation
gold, eval-only, never a feature (66/250 verified phones were never dialled,
so even membership carries future selection info).

`skip_traces.result`: no_new_info 591 / new_phone_found 148 /
new_address_found 27; single `trigger_rule` (no policy variation to learn
from; trigger not reproducible from dials alone — only 192/258 first-traces
with ≥15 prior dials are all-non-PTP). `cost_inr` 63–149 replaces the assumed
flat trace cost.

Payment `channel` (7: upi_link/app/nach_represent/branch_cash/neft/bbps/
cash_field) passes through as `payment_mode`. Agent `channel`: tele / field /
voice_bot; dial `channel`: tele_agent 47,344 / voice_bot 3,761 (bot attempts
have only 261 transcripts — `has_transcript` is outcome-skewed, 6.6%).
Split values: train/validation/test. Lender IDs `L01`–`L06` (rename in
mapping, not code).

## 5. What may serve as targets (graded; nothing invented)

- **Per-attempt RPC**: `answered AND disposition LIKE 'rpc_*'` — strongest
  available proxy (needs sign-off on the rpc_* inclusion set; agent noise runs
  both directions; 24 pattern-breaking rows quarantined; `language_barrier`
  ambiguous; untested = censored, never negative). Strict variant minus
  hung_up/refused is a defensible sensitivity. Raw `answered` alone is
  over-broad (37% non-RPC inside answered).
- **Per-(account,phone) ever/next-RPC**: aggregation with fixed horizon +
  censoring rules; shared-phone attribution per account (see §2).
- **Verified status as holdout gold**: eval only (n=250, invalid n=3,
  sampling frame UNKNOWN).
- **Payment-anchored labels**: weak supervision only — 365 pay w/o rpc, 355
  rpc w/o pay, 119 pay-before-first-rpc break any equivalence; post-cutoff
  payments (to 07-24) must never be features.
- **Auxiliary evidence (positive-only, never proof of absence)**:
  invalid↔does_not_exist 88 (hard/rare); third-party set + verified anchor;
  unreachable streaks (hazard features, not state proof); wrong_number +
  new-number remarks + verified not_borrower (recycled *proxy* — true
  recycled status is UNKNOWN, so cost handling stays rule+review);
  `ptp_id`/rpc_* subtypes (intent, not validity); per-visit address evidence
  (entropy 2.16 outcomes/address — single visits don't fix state); lagged
  trace history.
- **Must stay unlabelled**: all 1,684 never-dialled links and 1,640 unvisited
  addresses; single non-contact dispositions as validity statements;
  language_barrier/call_rejected motive/switched_off duration; any
  borrower-level "avoiding" or "recycled" flag; any per-phone validity
  probability (all latent — modelling targets, not given labels).

Minimum decisions required before label construction: sanctioned rpc_* set;
prediction grain + as-of + horizon + embargo; censoring/attribution rules;
verified-as-gold scope; payment-attribution rule; trace-trigger semantics;
split policy (account- vs contact-disjoint + time-purged, see §7).

## 6. Leakage paths (each verified; verdicts: SAFE / TEMPORAL-AGGREGATION /
   LEAKY / UNCERTAIN)

LEAKY (never inputs; labels/eval only under embargo): full-history or
post-as-of payments (150 past window); `verified_status`/membership/date;
trace `result`/new IDs/skip-sourced rows dated after scoring; same-row
outcome columns (`network_response`, ring/talk, `hangup_by`, `disposition`,
`remark`, `ptp_id`, `has_transcript`) when predicting that row; remark
substrings stating future PTP dates/amounts (observed in rpc_ptp remarks);
predicted-event transcript content; anything with ts > as-of. Deterministic
telephony rules (ring==0 ⇔ unreachable etc.) are a *generalisation* leak:
features memorizing them break on noisy real data.
TEMPORAL-AGGREGATION (history ≤ as-of minus embargo only): per-(account,
phone) dial sequences, ring/talk/hangup/hour/slot patterns, arm/propensity as
exposure controls, visit sequences + dwell + remarks (commitment-dates
excluded), payment amount/count/recency strictly ≤ as-of, lagged trace
history, phone age, exposure flags. SAFE (no time logic): lender/lender_type/
kyc format, account as group key (never learned id), split (routing only),
agent metadata as population descriptors, phone/address provenance with
`added_date ≤ as-of`, masked last-4 (weak, not leakage), address attributes,
account numerics **only if** snapshot as-of is confirmed pre-window.
UNCERTAIN (exclude until answered): all account snapshot numerics
(`dpd/bucket/emi/overdue/outstanding/salary/ability/prev_ptp/paid_other/
bounce` — no timestamp column; consistent with start-of-window but
unconfirmed), per-event `agent_id`/tenure (policy confound), GPS CRS,
photo_hash, `priority_slot` semantics, propensity formula (validate 1/k),
cross-account/lender graph aggregates (tenancy sign-off).

## 7. Selection, censoring, splits

- Two arms, different selection (OBSERVED): rule arm (2,277 accounts,
  propensity always 1.0, concentrates on KYC/slot-0, under-dials reference
  phones) vs random arm (121 dialled of 123, propensities {1.0, 0.5, 0.3333,
  0.25}, consistent with 1/k uniform-over-k). Only the random arm supports
  unbiased estimation; validate the 1/k formula before IPS use.
- Within-account concentration is extreme (median max-phone share 0.92):
  non-favoured numbers are thin by policy, not validity. Reference/employer/
  bureau links are systematically underexposed (52–57% never dialled vs KYC
  1.5%) — exactly the third-party/recycled-relevant numbers. Untested =
  unknown, never negative.
- 867/1,635 pay+dial accounts stopped dialling *before* first payment, yet
  18.9% of all dials occur *after* first payment — no clean exit rule; keep
  post-pay/post-PTP exposure explicit. 15 accounts paid before first dial, 32
  pay-accounts never dialled: an unreachable-driven recovery stream bounds
  trace VOI (self-cure).
- Provided splits are stratified-random over one shared window (70/15/15,
  proportional by lender/bucket/arm) — valid for same-period account-level
  comparison ONLY. Not valid for forward deployment estimates (temporal
  leakage via shared calendar: voice_bot collapse, payment surge learnable
  within-sample), contact-level generalisation (43 shared ids + 553 shared
  masks cross splits; 12 dialled ids dialled under accounts in different
  splits), or any post-window label. Keep as secondary sanity; primary
  validation stays time-purged + embargoed + group-contained, with the random
  arm as the unbiased slice.

## 8. EDA base rates (RPC proxy = `disposition LIKE 'rpc_*'`, 16.4%/attempt)

Per attempt 16.4%; per dialled link ever-RPC 61.7%; per dialled account
88.7% — negatives *with exposure* are the scarce resource, not positives.
Previous-disposition → next-RPC is the strongest single-step signal
(switched_off 0.8%, wrong 4.9%, third-party 7–9%, invalid 0% vs rpc_* 24–30%
stickiness, no_answer 19.6% still viable). Attempt index k=8–25 rises
(17–22%) — survivor effect, not redial efficacy. Provenance gaps are wide
(skip_trace 20.7% / borrower_update 19.8% / KYC 18.0% vs bureau 10.7% /
reference 8.4% / employer 4.3%); bureau wrong-number rate 10.8% is the top
recycled-risk correlate; employer/reference phones carry 30–38% third-party
rates. Bureau band is the strongest account segment (750-900: 24.4% vs
300-549: 15.5%, monotone). Hour edges (08:00 21%, 18:00 19.8%) beat midday
but volumes are policy-set — slot effects need propensity controls. 330 links
show both rpc_* and wrong_number (contact-then-stranger sequences — the only
longitudinal recycled hint); 690 show rpc_* and third_party (identity is
agent-judged, contradicts provenance both ways — neither is truth).
Recycled is unsupervised (proxies only, thresholds need cost-ratio + human
review, suppression reversible); third-party is best-evidenced but per-attempt
precision <1 (risk score + audit trail, never auto-decision from one
disposition). Rare states (invalid 88, barrier 44, verified-invalid n=3)
need weak supervision + cost weighting, never resampling that breaks PIT.

## 9. Schema gaps vs our pipeline

1. No `received_at` — assume equal to event time; late-event logic untestable here.
2. Masked phones unhashable — **use stable `phone_id`/`address_id` as the contact reference** (hash the ID); number normalization reserved for full-number extracts.
3. No transcript content (`has_transcript` without table) — bot-transcript features cannot run.
4. No consent/DND/dispute/deceased columns — those guardrails stay unevaluable.
5. `ptp_id` dangles — ignore, never join.
6. GPS is grid coords, not lat/lon — no geographic math on them.
7. `salary_credit_day` (68% null, salaried-only by design) and `ability_to_pay_estimate` (30% null, no clean driver) are expected feature nulls.
8. Lender IDs need renaming; per-lender economics unsettled (`lender_overrides` exists for guardrails, costs has no per-lender keys).
9. Account snapshot fields quarantined (§6) — the pipeline must run with and without them until as-of is confirmed.
10. Borrower-level constructs (reachability latent, cross-line pooling) drop to account level until a borrower map exists (§2).

## 10. Feature availability (182-group check)

Computable: telephony, dispositions (with the §4 handling), text cues
(re-validate Hindi/Kannada/English lists on issued remarks — do not reuse
patterns tuned on any earlier data), shared-contact graph by **id-degree** (never mask-degree),
record history, borrower cross-line signals (at account grain), field visits,
account/VOI inputs (pending snapshot confirmation), calendar. Degraded:
bot transcripts (absent), trace-cost modeling (use observed `cost_inr`),
`immediate_hangup`, slot effects (confounded until propensity-adjusted),
prev_ptp/ability/paid_other marginals (flat — require conditional proof),
agent-id means (policy echoes).

## 11. Privacy handling for this data

Stated synthetic, handled as sensitive: realistic formats, masked phones
(source-masked, nothing to leak), code-mixed remarks. `datasets/` is
gitignored (audit change, reported). Open privacy items with a deadline of
first real eval: unsalted contact hashing → peppered HMAC; dead-letter
free-text redaction; `reports/` gitignore + no row-level IDs in reports;
event-store/dead-letter retention TTLs; no raw rows in prompts, chats,
fixtures, or tickets.

## 12. Ask-CN list (merged, priority order)

1. Timestamp timezone (IST wall-clock vs UTC) — blocks slot features.
2. Consent / DND / dispute / deceased source columns (absent — guardrails gap).
3. Transcript content for `has_transcript=True` rows, or confirmation of unavailability.
4. `call_rejected`, `cash_collected`, `immediate_hangup`-derivation, bucket-X and `priority_slot` definitions, GPS CRS, photo_hash meaning.
5. Per-lender economics (single vs per-lender cost tables).
6. Campaign/policy flags beyond `dialling_arm`; `selection_propensity` formula confirmation (1/k).
7. Account snapshot as-of for all account fields.
8. Annotations: recycled confirmations first, avoiding-vs-invalid sample second (columns: `contact_point_ref`, `true_state` vocabulary, `valid_from`/`valid_to`); lawful basis for cross-account/lender contact graph.

## 13. Mapping plan (from this audit, for the mapping work)

One YAML per source table: dials → `dial_attempt` (+`disposition` companion
from the same row's disposition block, per our envelope split);
phones/addresses rows → `contact_point_update` (source/kind/is_primary carried);
payments → `payment`; visits → `field_visit`; traces → outcome evidence
(not a canonical event — record as trace history for VOI, never as a
predictor); verified rows → held-out gold only. Contact ref = hash of
`phone_id`/`address_id`; `occurred_at` = event ts localized per §3 resolution;
`received_at` = occurred_at (flagged assumption). Lender rename L01→canonical
in mapping. Unknown enum values reject as `unknown`, never crash (fuzz-tested).
