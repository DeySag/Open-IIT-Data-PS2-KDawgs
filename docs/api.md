# Serving API (PS2 RPC)

The serving layer exposes the contact-point state decisions to CN
consumers (dialer, campaign engine, skip-trace queue, compliance
controls, visit planner). All endpoints live under `/v1`.

**Simulation-only.** Every number returned by this service is
derived from the synthetic simulator. Nothing here is a real-world
measurement. Assumptions live in `configs/` and `docs/assumptions.md`.

## Conventions

- **Lender scoping:** pass `X-Lender-Id` header. A request scoped
  to lender A never returns lender B's data (per-lender isolation).
- **API key:** a placeholder `X-API-Key` header check is enforced
  on every `/v1` endpoint when `serve.api_key` is configured; it is
  disabled (dev mode) when unset.
- **Response envelope:** every response carries `model_version`,
  `feature_snapshot_id`, `generated_at` and `valid_until`.
  `valid_until = generated_at + validity_hours` (default 24h,
  configurable via `serve.validity_hours`).
- **No PII in logs:** logs contain hashes only (contact point
  refs are already hashed); no raw numbers or borrower identifiers.

## Endpoints

### `POST /v1/events`

Batch event intake. Accepts a JSON array of the canonical input
event envelope. Returns ingestion counts and runs the **compliance
fast path** synchronously on each accepted event before the
response returns.

Response: `accepted`, `duplicate`, `rejected`, `dirty_marked`,
`fast_path` (per-event rule/reason/evidence/latency),
`fast_path_latencies_ms`, plus the response envelope.

### `GET /v1/dial-lists?lender_id&date`

Per account, contact points ranked by health with `p_rpc`,
`best_slot`, `confidence` and an `excluded` flag. A contact point
is excluded when it is **suppressed** or **dead beyond threshold**
(`invalid + recycled >= serve.dead_contact_threshold`).

**Never a dead end:** if every contact point for an account is
excluded, the account still appears with a stated
`fallback_action` (`trace` when fresh, `manual_review` when
stale) and a `fallback_reason`.

### `GET /v1/decisions?lender_id&page&page_size` and `POST /v1/score`

The `OutputDecision` contract (one of `continue`,
`switch_contact_point`, `switch_channel`, `trace`) with
`action_params`, `reason_code`, `ranked_contact_points`, optional
`trace` (with `recoverable_amount`) and `flags`. `/decisions` is
batch + paginated; `/score` scores a single account.

### `GET /v1/trace-queue?lender_id&budget`

Accounts ranked by `voi_per_rupee` with `recoverable_amount`,
`p_find`, `est_cost` and `rank`, cut at the trace budget
(knapsack-style). VOI is incremental:

```
VOI = p_find * (p_recovery_if_reached - p_recovery_if_not_reached)
      * recoverable_amount
      - trace_cost - collection_cost - compliance_cost
```

### `GET /v1/suppression?lender_id&since=version`

Full suppression list with a monotonically increasing `version`.
Pass `since=version` to get only entries added/modified after that
version (diff). Entries are append-only with an audit log:
`contact_point_ref`, `lender_id`, `reason` (`recycled` |
`third_party`), `evidence` (event ids), `added_at`,
`model_version`, `removal_requires=cn_signoff`.

### `POST /v1/suppression/removal-requests`

Creates a **pending** removal request requiring CN sign-off. There
is **no direct delete**; the entry stays in force until CN approves
the request out-of-band. Response: `request_id`, `status`
(`pending_cn_signoff`), `removal_requires`.

### `GET /v1/visit-candidates?lender_id`

Address health stub derived from `field_visit` events:
`address_health` (`occupied` | `absent` | `moved` |
`unresolved`), `best_visit_window`, `recommendation` and
`origination_review` flag. **Location confidence is always
`null`** (PS3 is dropped from scope).

### `GET /v1/health`

Service health with data freshness (`last_score_time`,
`score_age_hours`, `stale`), `last_batch_time`, `staleness_state`
(`fresh` | `stale`) and the current `suppression_version`.

## Compliance fast path

On `POST /v1/events`, each accepted event is run synchronously
through a `RecycledSignalDetector` and any suppression entries are
added **before the response returns**. Rules are config-driven and
each triggered rule is logged as evidence:

| Rule | Trigger | Suppression reason |
|---|---|---|
| `wrong_number_disposition` | `disposition = wrong_number` | `recycled` |
| `third_party_disposition` | `disposition = third_party` | `third_party` |
| `transcript_cue:<cue>` | who-is-this / name-mismatch / language-mismatch cue in a bot transcript | `recycled` |
| `recycled_risk_threshold` | `recycled` posterior from the Scorer `>= serve.recycled_risk_threshold` | `recycled` |

- **Idempotent:** the same evidence (event id) never creates a
  duplicate suppression entry.
- **Latency:** per-event fast-path latency is recorded
  (`fast_path_latencies_ms`) and bounded by
  `serve.fast_path_max_latency_ms`.
- **One-way:** entries are never removed by the service; only
  removal requests (requiring CN sign-off).

## Stale-score fallback (fail-safe)

If the latest scores are older than `serve.max_score_age_hours`
(default 24h) **or** the Scorer fails, the service:

1. serves the **last cached scores** with **confidence decayed**
   by `max(serve.confidence_decay_floor, 1 - age_hours /
   max_score_age_hours)`;
2. **never recommends `trace`** from stale data (a stale `trace`
   decision is downgraded to `switch_contact_point` when another
   contact point exists, otherwise `continue`, and `trace` is set
   to `null`);
3. keeps the **current suppression list in force** (suppression is
   still enforced in dial lists);
4. sets a `stale: true` flag on the response;
5. **never blocks or errors the consumer** (the fallback path
   cannot raise).

This keeps the dialer and campaign engine running through a scorer
outage while staying conservative on compliance-critical decisions.

## Configuration

All tunables live in `ServeConfig` (`src/rpc/serve/config.py`),
optionally overridden by `configs/serve.yaml`:

| Parameter | Default | Purpose |
|---|---|---|
| `validity_hours` | 24 | `valid_until = generated_at + validity_hours` |
| `max_score_age_hours` | 24 | staleness threshold for the fallback |
| `confidence_decay_floor` | 0.1 | minimum confidence multiplier when stale |
| `fast_path_max_latency_ms` | 100 | fast-path latency bound |
| `recycled_risk_threshold` | 0.5 | recycled posterior that triggers suppression |
| `dead_contact_threshold` | 0.8 | `invalid + recycled` above which a contact point is excluded |
| `default_trace_budget` | 100000 | trace queue budget |
| `trace_cost` / `p_find` / `recoverable_amount_default` | 500 / 0.30 / 50000 | trace VOI assumptions |

## Swapping in the real modules

The serving layer depends on three `Protocol` interfaces
(`src/rpc/serve/interfaces.py`): `EventStore`, `Scorer` and
`Decider`. In-memory stubs are used now; when the real modules
land (`src/rpc/ingest/`, `src/rpc/models/state_tracker/`,
`src/rpc/decision/`), inject them via `create_app(event_store=…,
scorer=…, decider=…)`. No endpoint code changes are required.
