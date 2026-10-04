# Decision Log

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

## 2026-10-04: Normalized valid_reachable dispositions

**Decision:** Normalized `simulator.dispositions.valid_reachable` from sum=1.05 to sum=1.0.

**Reason:** Probability distributions must sum to 1.0. The original config had:
- RPC: 0.70
- wrong_number: 0.02
- third_party: 0.05
- switched_off: 0.05
- not_reachable: 0.05
- promise_to_pay: 0.10
- dispute: 0.03
- callback: 0.05
Sum = 1.05

**Alternative considered:** Proportionally reduce RPC from 0.70 to 0.65 and adjust others. Chose proportional normalization to preserve relative ratios.

**Impact:** All downstream disposition sampling now uses normalized probabilities. RPC probability for valid_reachable state changed from 0.70 to ~0.667.

**Assumption added to docs/assumptions.md:** Simulator disposition probabilities are normalized to sum to 1.0.