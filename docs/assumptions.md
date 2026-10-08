# Working Assumptions (A1-A6)

*Update this file when assumptions change. Notify coordinator.*

---

## A1 Transport
**Assumption:** Daily batch pull of event extracts plus an API for decisions; push path for suppression; transport sits behind the adapter layer.
**Status:** Pending CN confirmation
**Impact:** Affects ingestion adapter design and serving latency requirements

---

## A2 Identifiers
**Assumption:** Account and lender IDs are stable; contact points have no guaranteed stable ID, so we derive a canonical key by normalising and hashing the number or address.
**Status:** Pending CN confirmation
**Impact:** Hashing strategy in `hash_contact_point()`; collision handling

---

## A3 Suppression Authority
**Assumption:** We add to suppression list immediately; removal needs CN sign-off.
**Status:** Pending CN confirmation
**Impact:** Suppression entry schema `removal_requires: "cn_signoff"`; fast-path implementation

---

## A4 Existing Guardrails
**Assumption:** CN already enforces contact hours and frequency; we only restrict, duplicate checks are fine, CN rules win on conflict.
**Status:** Pending CN confirmation
**Impact:** Guardrails engine only adds restrictions, never removes them

---

## A5 Latency
**Assumption:** Daily batch completes before the outbound planning window; fast path targets minutes (our design choice).
**Status:** Design choice
**Impact:** Batch job scheduling; fast-path suppression API

---

## A6 Late Events
**Assumption:** Payment-to-contact links are retroactive; use event time vs ingest time; correction window about 7 days (our choice).
**Status:** Design choice
**Impact:** Point-in-time feature computation; event store correction logic

---

## Official extracts

The pipeline reads the official CN extracts (`datasets/`, gitignored;
see `docs/dataset_audit.md` for the audit). Assumptions awaiting CN
confirmation live with the dataset audit's ask-CN list:

- State transition/emission/disposition shapes (model priors stay qualitative
  and leakage-free per `configs/state_tracker.yaml`)
- Cost parameters, recovery rates, trace success rates (from observed
  `cost_inr` and recovery outcomes, not assumed flat rates)

## Official extracts: mapping assumptions (P1, all flagged pending CN)

- **Timestamps are Asia/Kolkata wall-clock.** Naive stamps localised to
  Asia/Kolkata then stored UTC. Dial hours fit IST contact-hours, not UTC.
  A 5:30 shift moves every slot feature: slot-model work must not assume
  either until CN confirms.
- **received_at == occurred_at.** One timestamp per event; late/out-of-order
  and dirty-marking logic cannot be exercised on this data (covered
  synthetically in unit tests).
- **1 account = 1 borrower.** No `borrower_id` exists in any file; borrower_id
  maps from account_id. Borrower-level constructs (reachability latent,
  cross-line pooling) run at account grain until a borrower map is supplied.
- **Contact refs are peppered HMAC-SHA256** (truncated 16 hex) of
  `phone_id`/`address_id`, pepper from `CN_HASH_PEPPER` (never committed).
  Unset pepper falls back to legacy sha256 with a warning. Rotation = re-ingest.
- **Split routing survives mapping untouched.** `split` is routing metadata,
  never a feature; account_id linkage verified orphan-free; splits.csv is
  consumed downstream (P2).