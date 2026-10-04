# Decision log (feature workstream)

All simulation-only unless stated otherwise.

## 2026-10-04 - features: window expansion exceeds the 60-75 target
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
