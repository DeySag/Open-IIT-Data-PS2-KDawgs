# Ingestion adapter + event store (SIMULATION-ONLY)

All data handled here is synthetic (repo simulator output or invented fake CN
formats). Nothing in this module has been validated against real CN data.

## What it does

`src/rpc/ingest/` translates per-source extracts into the frozen canonical
`InputEvent` envelope (`src/rpc/contracts`), normalises contact points to
hashes, validates rows, and appends them to a DuckDB event store with dedupe,
dead-letter quarantine and late-event (dirty) marking.

## Public API (`src/rpc/ingest/__init__.py`)

```python
ingest(batch_or_path, source, db_path=None, config=None)
    -> {"accepted": int, "duplicate": int, "rejected": int, "dirty_marked": int}
read_events(received_before=None, event_types=None, lender_id=None,
            contact_point_refs=None) -> pandas DataFrame
replay(path, source, db_path=None) -> {...}   # idempotent re-ingest
IngestAdapter(config).ingest / .read_events / .replay / .set_watermark(...)
```

`batch_or_path` is a DataFrame, a list of dicts, or a path to
`.csv` / `.parquet` / `.ndjson` / `.jsonl` / `.json`.
`source` selects `configs/field_mappings/<source>.yaml`.
`db_path` defaults to `data/event_store.duckdb` (env override `INGEST_DB_PATH`).

For the features workstream: `src/rpc/features/source.py` does not exist, so
there is no `EventSource` protocol to implement. Read canonical events through
`read_events` here -- it supports the point-in-time filters the feature store
needs (`received_before`, `event_types`, `lender_id`, `contact_point_refs`).
(coordinator: confirm this interface or point us at the protocol when it lands.)

## Mapping files (`configs/field_mappings/`)

| source | format | description |
|---|---|---|
| `cn_dialer_csv` | CSV | fake dialer: renamed columns, epoch-ms IST, own result codes (`ANS`, `SWOFF`, ...) |
| `cn_disposition_ndjson` | NDJSON | fake dispositions: nested fields, `%d-%m-%Y %H:%M:%S` IST + ISO+05:30, own outcome codes (`PTP`, `WN`, ...) |
| `simulator` | Parquet | repo simulator `events.parquet`: canonical columns, hash passthrough, JSON-string payload |

Spec forms: `key: column`, `{field: a.b.c}` (nested), `{const: v}`,
`{field, map: {...}}` (unmapped values reject the row as `unknown_enum`),
`{field, format: epoch_ms|epoch_s|iso|<strptime>, tz: ...}`,
`{field, type: float|int}`, `{field, uuid5: true}` (deterministic UUID for
non-UUID source ids), `{contact_ref: true}` (hashed ref into the payload).
Contact points: `{raw_field, kind: phone|address|hash_passthrough}`.
Per-file knobs under `validation:`: `clock_skew_tolerance_seconds` (default 300),
`sample_rate` (default 0.01), `sample_seed`.

## Normalisation (never store raw PII)

Phones: strip spaces/dashes/dots/parens and `+`, then strip leading `91`
(12-digit) or trunk `0` (11-digit). Addresses: lowercase + collapse
whitespace. `sha256` truncated to 16 hex chars is the `contact_point_ref`.
Raw values are dropped before the store write, redacted in dead-letter rows
(`[redacted-pii]`; for unknown sources all values are redacted since the PII
field is unknowable), and never logged -- logs carry counts and hashes only.

## Validation (vectorised; rejections never raise)

Reason precedence: `missing_mapping` (no YAML for source -- whole batch) >
`missing_required_field` > `unknown_enum` > `bad_timestamp` >
`time_inversion` (`received_at` earlier than `occurred_at` beyond tolerance) >
`pydantic_sample_failed`. On top of the vectorised checks, the full pydantic
`InputEvent` schema runs on **all** rejected rows (confirms rejection) and a
**1% random sample** of accepted rows (failures quarantined to dead-letter).
Pydantic error summaries kept in logs contain locations/kinds only, no values.

Hidden ground-truth columns (`true_state`, `borrower_avoiding`,
`shared_reason`, ...) are dropped on load when present in input.

## Event store (DuckDB)

- `events`: `event_id` PK + canonical columns + `payload` JSON + `ingested_at`.
  Dedupe on `event_id` keeping the earliest `received_at`; within-batch dupes
  also collapsed. Same batch twice -> byte-identical tables.
- `dead_letter`: `(row_hash PK, raw_json redacted, reason, source,
  ingested_at)`; `row_hash` makes re-ingest idempotent.
- `dirty_contact_points`: `(contact_point_ref, lender_id)` PK, reason
  `late_event`. A row is late when a watermark exists for the point and
  `received_at > watermark` (arrived after scoring) or
  `occurred_at < watermark` (belongs to a scored period).
- `watermarks`: scoring watermarks upserted via `set_watermark` (owned by the
  feature pipeline; ingestion only reads them).

Bulk path: staged views + SQL anti-joins/updates in 500k-row chunks. The only
linear Python passes are single `Series.map` calls over individual columns and
one `zip` pass assembling payload JSON for non-passthrough sources; pydantic
runs only on rejected rows + the 1% sample.

## Performance (SIMULATION-ONLY numbers, measured 2026-10-04)

- Dev simulator output: `data/events.parquet`, 413,987 rows / 32.9 MB ->
  ingest 21.1 s: `{accepted: 411928, duplicate: 2059, rejected: 0,
  dirty_marked: 0}` (duplicates are the simulator's own 0.5% re-emitted
  `duplicate_event_prob`, as configured). Store: 411,928 rows, ~75 MB,
  ~19.5k rows/s -> 5M rows project to ~4.3 min, inside the 5-min budget
  (projection only; full-scale ingest not yet measured).
- Replay of the same file: 15.3 s, `{accepted: 0, duplicate: 413987,
  rejected: 0}`, all three tables byte-identical afterwards.
- 1M-row synthetic dialer-CSV stress (fake-CN path with hashing + payload
  assembly, not the full-scale scenario): 104.4 s, ~9.6k rows/s, 0 rejected;
  store ~196 MB.
- Full-scale (5M) ingest: NOT measured -- simulator team is regenerating, and
  real CN data may arrive first (2026-10-04). Must be re-run before claiming
  the 5-min budget.
