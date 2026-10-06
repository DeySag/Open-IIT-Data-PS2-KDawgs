# Decision layer design (workstream D, day 1)

Simulation-only: every number below is illustrative until CN confirms costs
and curves. No learned model overrides compliance.

## Pipeline order

```
guardrails (restrict-only) -> risk exclusions + suppression -> action mapping
-> trace gate (deferral, VOI, eligibility) -> OutputDecision + suppressions
```

`decide(ctx, scores) -> OutputDecision`; `decide_full(...)` additionally
returns suppression requests (serving layer persists near-real-time),
the internal reason code, and fired guardrail rules for audit.
`rank_trace(candidates, budget)` ranks the portfolio skip-trace queue.

## Guardrails

Evaluated first; they only remove actions/channels/contact points. Hard
blocks (no-consent, dispute, deceased/insolvent, legal case) park the
account on `switch_channel/field` and never trace; the serving layer must
not execute outreach for parked accounts. Contact hours use Asia/Kolkata
from config (overnight windows supported). Attempt caps remove fresh phone
attempts. DND removes voice/telecaller channels. `trace_pending` removes
trace (no double-queue).

## Action mapping (exactly four actions)

All contact points ranked by `(p_rpc, confidence)`; suppressed and
recycled/third-party-risk lines excluded from dial order (appended after
viable lines for audit, never first). Recycled cutoff `1/(1+100) ~= 0.0099`
from the explicit cost ratio in `costs.yaml`, applied to the recycled-risk
classifier score; recycled-dominant posteriors suppress independently
(fast path). Borrower avoidance = max avoiding mass over viable lines;
high avoidance on callable lines -> `switch_channel` (WhatsApp only with
opt-in, else SMS unless DND, else field). Healthy best -> `continue`;
temp-unreachable -> `continue` + backoff (or move when a healthy line
exists); dead best -> move or trace; nothing viable + trace ineligible ->
parked `switch_channel/field`. No account is left without a next action.

## Reason-code table

| Code | Driver | Action |
|---|---|---|
| VALID_CONTINUE | valid_reachable dominant, avoidance low | continue |
| TEMP_UNREACHABLE_BACKOFF | temp_unreachable dominant / deferred trace | continue + backoff |
| AVOIDING_SWITCH_CHANNEL | borrower avoidance high on callable lines | switch_channel |
| SWITCHED_OFF_MOVE_OR_TRACE | switched_off_long dominant | switch_contact_point / trace |
| INVALID_TRACE | invalid dominant / no viable line | trace |
| RECYCLED_SUPPRESS | recycled risk above cost-ratio cutoff | suppression + move |
| THIRD_PARTY_RESTRICT | third-party mass above threshold | switch_contact_point |
| ADDRESS_ABSENT_CHANGE_TIME | address usually absent (stub) | continue (new visit time) |
| ADDRESS_MOVED_TRACE | borrower moved (stub) | trace |
| ADDRESS_FABRICATED_TRACE_FLAG | fabricated/incomplete (stub) | trace + origination_review |
| ADDRESS_UNRESOLVED_REVIEW | hard to find (stub, never written off) | continue (manual review) |
| GUARDRAIL_NO_CONSENT / DISPUTE / DECEASED / LEGAL_CASE / SUPPRESSED | hard block | parked switch_channel/field |
| GUARDRAIL_DND / CONTACT_HOURS / FREQ_CAP / TRACE_PENDING | restriction | narrowed action |
| DEFERRED_TRACE_WAIT | EV(wait) > trace now | continue (trace later) |
| LOW_VOI_DEPRIORITISED | VOI/rupee below threshold | parked, not queued |

`GUARDRAIL_*` / `DEFERRED_*` / `LOW_VOI_*` are decision-layer codes; at the
serving boundary they coerce to the closest frozen contract code
(restrictions -> `THIRD_PARTY_RESTRICT`, waits -> `TEMP_UNREACHABLE_BACKOFF`)
while audit keeps the full code. Recommended contract change (needs
coordinator approval): add the `GUARDRAIL_*` values to
`contracts.ReasonCode`. Not required to ship: outputs always validate today.

## VOI ranker

`VOI = P(find|trace) x gain x recoverable - trace_cost - collection - goodwill`,
with `P(find|trace) = method_success_rate x P(dead)` where `P(dead)` is
P(invalid or long-dead) -- never P(no contact), so avoiding accounts get
~zero VOI. Recoverable = outstanding x bucket x secured x (1-haircut),
discounted. Ranked by VOI/rupee; greedy fill under budget. Exactness note:
greedy-by-ratio is exact for fractional knapsack but a heuristic for 0/1
(whole-trace) knapsack; the optimality gap is small at portfolio scale and
a DP exact solver is a day-3 option.

### Worked example (illustrative numbers)

Account: outstanding Rs 100,000, bucket 31-60, unsecured_retail,
P(dead)=0.8, digital trace.
recoverable = 100000 x 0.85 x 0.60 x 0.80 / 1.12^0.5 = Rs 38,552.27;
gain = 0.25-0.03 = 0.22; P(find) = 0.25 x 0.8 = 0.20;
VOI = 0.20 x 0.22 x 38552.27 - 100 - 60 - 25 = Rs 1,511.30;
VOI/rupee = 15.11. Test `test_hand_computed_voi_example` pins this.

## Assumptions / unverified

- All costs, recovery curves, haircut (20%), lag (0.5y), thresholds are
  assumptions in `configs/costs.yaml` / `guardrails.yaml`.
- Borrower avoidance currently = max avoiding mass (proxy until workstream
  B's hierarchical latent lands); deferral EV(wait) is a stub heuristic.
- `feature_snapshot_id` = `"stub_snapshot"` until the feature store lands.
- Empty ranked lists use a zero-score `no_viable_contact_point` placeholder
  to satisfy the contract's `min_length=1` (do-not-dial signal).
- Missing model evidence ids use a deterministic UUID5 placeholder;
  serving layer must reconcile with real event ids.
- `src/rpc/serve/app.py` (neighbour module, untouched) still builds
  `TraceInfo` without `recoverable_amount` and scores only the first
  contact point -- flagged for workstream F; this layer is correct.
