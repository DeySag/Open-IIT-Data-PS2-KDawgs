# Feature layer

Point-in-time feature pipeline in `src/rpc/features/`. One row per phone or
address contact point known at `as_of`, for health-model training and scoring.

## 1. Input schemas (official extracts)

The pipeline reads the official CN extracts. Schema,
codebook, null rates, join integrity, and leakage rules for those files:
`docs/dataset_audit.md` (committed brief) and the deep records in `datasets/`
(`01_dataset_audit.md` column inventory, `02_labels_and_entities.md`,
`03_leakage_and_censoring.md`, `04_ps2_eda.md`).

Key consequences for features (all verified in the audit): contact reference
is the stable CN `phone_id`/`address_id` (numbers arrive masked); only one
timestamp per event (assume received == occurred; late-event handling is
covered by unit tests); transcript content is absent (`has_transcript` without
a table); account snapshot fields are quarantined until their as-of is
confirmed; `verified_contact_points` is eval-only gold, never input.

## 2. Point-in-time contract

Visibility = `received_at <= as_of`; window membership = `occurred_at`.
Dedup on `event_id` keeps earliest `received_at`. Universe = contact points
with `created_at <= as_of` plus refs first seen in visible events. No post-
`as_of` data is touched for any reason. Never-attempted rows exist with
`has_any_attempt=false`, counts 0, rates/days-since null. Slots use
Asia/Kolkata (morning 8-12, afternoon 12-16, evening 16-19, from configs).

## 3. Outputs and entry points

- `build_features(as_of, source, contact_point_refs=None) -> DataFrame`
  keyed `(lender_id, borrower_id, account_id, contact_point_ref, as_of)` plus
  `feature_snapshot_id` (sha256 of config text + as_of + event watermark) and
  `event_watermark` (max visible `received_at`).
- `build_training_table(as_of_dates, source)` stacks the same code path.
- `EventSource` ABC; `ParquetEventSource` (DuckDB predicate pushdown) and
  `DataFrameEventSource` (tests). Ground-truth tables are never opened here.
- Labels live in `src/rpc/features/labels.py` (never imported by feature
  code); hidden-truth joins live in `src/rpc/eval/truth.py`.
- CLI: `python -m src.rpc.features.build --scale dev --as-of <date>`
  (`--as-of-range start end step_days`, `--out`, `--data-dir`,
  `--render-docs` refreshes the registry table below); `make features-dev`.

## 4. Parameters

All windows/thresholds/slots/holidays in `configs/features.yaml`; text
patterns in `configs/text_patterns.yaml`. Assumptions are tracked in
`docs/assumptions.md`; decisions in `docs/decision_log.md`.

## 5. Feature registry (generated from `spec.py`; do not edit by hand)

The conditional `agent_wrong_number_rate` row is emitted only when an
`agent_id` is observed in disposition payloads.

<!-- REGISTRY:START -->
| feature | group | dtype | source events | window | null semantics | description |
|---|---|---|---|---|---|---|
| `n_attempts_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Dial attempts with occurred_at in the last 1d. |
| `n_answered_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Attempts with network_response=answered in the last 1d. |
| `answer_rate_1d` | telephony | Float64 | dial_attempt | 1d | null when no attempts in window (never 0-imputed). | n_answered / n_attempts over the last 1d. |
| `n_switched_off_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Attempts with network_response=switched_off in the last 1d. |
| `n_not_reachable_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Attempts with network_response=not_reachable in the last 1d. |
| `n_does_not_exist_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Attempts with network_response=does_not_exist in the last 1d. |
| `n_no_answer_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Attempts with network_response=no_answer in the last 1d. |
| `n_busy_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Attempts with network_response=busy in the last 1d. |
| `n_attempts_morning_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Attempts placed in the morning slot (IST) in 1d. |
| `n_attempts_afternoon_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Attempts placed in the afternoon slot (IST) in 1d. |
| `n_attempts_evening_1d` | telephony | Int64 | dial_attempt | 1d | 0 when no attempts in window. | Attempts placed in the evening slot (IST) in 1d. |
| `answer_rate_morning_1d` | telephony | Float64 | dial_attempt | 1d | null when no attempts in that slot and window. | Answered share within the morning slot (IST) over 1d. |
| `answer_rate_afternoon_1d` | telephony | Float64 | dial_attempt | 1d | null when no attempts in that slot and window. | Answered share within the afternoon slot (IST) over 1d. |
| `answer_rate_evening_1d` | telephony | Float64 | dial_attempt | 1d | null when no attempts in that slot and window. | Answered share within the evening slot (IST) over 1d. |
| `weekend_attempt_share_1d` | telephony | Float64 | dial_attempt | 1d | null when no attempts in window. | Share of attempts on Sat/Sun (IST) over 1d. |
| `ring_seconds_mean_1d` | telephony | Float64 | dial_attempt | 1d | null when no attempts in window. | Mean ring_seconds over 1d. |
| `ring_seconds_std_1d` | telephony | Float64 | dial_attempt | 1d | null with fewer than 2 attempts in window. | Sample std of ring_seconds over 1d. |
| `short_ring_rate_1d` | telephony | Float64 | dial_attempt | 1d | null when no attempts in window. | Share of attempts with ring_seconds below configs short_ring_seconds. |
| `n_attempts_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Dial attempts with occurred_at in the last 3d. |
| `n_answered_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Attempts with network_response=answered in the last 3d. |
| `answer_rate_3d` | telephony | Float64 | dial_attempt | 3d | null when no attempts in window (never 0-imputed). | n_answered / n_attempts over the last 3d. |
| `n_switched_off_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Attempts with network_response=switched_off in the last 3d. |
| `n_not_reachable_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Attempts with network_response=not_reachable in the last 3d. |
| `n_does_not_exist_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Attempts with network_response=does_not_exist in the last 3d. |
| `n_no_answer_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Attempts with network_response=no_answer in the last 3d. |
| `n_busy_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Attempts with network_response=busy in the last 3d. |
| `n_attempts_morning_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Attempts placed in the morning slot (IST) in 3d. |
| `n_attempts_afternoon_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Attempts placed in the afternoon slot (IST) in 3d. |
| `n_attempts_evening_3d` | telephony | Int64 | dial_attempt | 3d | 0 when no attempts in window. | Attempts placed in the evening slot (IST) in 3d. |
| `answer_rate_morning_3d` | telephony | Float64 | dial_attempt | 3d | null when no attempts in that slot and window. | Answered share within the morning slot (IST) over 3d. |
| `answer_rate_afternoon_3d` | telephony | Float64 | dial_attempt | 3d | null when no attempts in that slot and window. | Answered share within the afternoon slot (IST) over 3d. |
| `answer_rate_evening_3d` | telephony | Float64 | dial_attempt | 3d | null when no attempts in that slot and window. | Answered share within the evening slot (IST) over 3d. |
| `weekend_attempt_share_3d` | telephony | Float64 | dial_attempt | 3d | null when no attempts in window. | Share of attempts on Sat/Sun (IST) over 3d. |
| `ring_seconds_mean_3d` | telephony | Float64 | dial_attempt | 3d | null when no attempts in window. | Mean ring_seconds over 3d. |
| `ring_seconds_std_3d` | telephony | Float64 | dial_attempt | 3d | null with fewer than 2 attempts in window. | Sample std of ring_seconds over 3d. |
| `short_ring_rate_3d` | telephony | Float64 | dial_attempt | 3d | null when no attempts in window. | Share of attempts with ring_seconds below configs short_ring_seconds. |
| `n_attempts_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Dial attempts with occurred_at in the last 7d. |
| `n_answered_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Attempts with network_response=answered in the last 7d. |
| `answer_rate_7d` | telephony | Float64 | dial_attempt | 7d | null when no attempts in window (never 0-imputed). | n_answered / n_attempts over the last 7d. |
| `n_switched_off_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Attempts with network_response=switched_off in the last 7d. |
| `n_not_reachable_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Attempts with network_response=not_reachable in the last 7d. |
| `n_does_not_exist_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Attempts with network_response=does_not_exist in the last 7d. |
| `n_no_answer_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Attempts with network_response=no_answer in the last 7d. |
| `n_busy_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Attempts with network_response=busy in the last 7d. |
| `n_attempts_morning_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Attempts placed in the morning slot (IST) in 7d. |
| `n_attempts_afternoon_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Attempts placed in the afternoon slot (IST) in 7d. |
| `n_attempts_evening_7d` | telephony | Int64 | dial_attempt | 7d | 0 when no attempts in window. | Attempts placed in the evening slot (IST) in 7d. |
| `answer_rate_morning_7d` | telephony | Float64 | dial_attempt | 7d | null when no attempts in that slot and window. | Answered share within the morning slot (IST) over 7d. |
| `answer_rate_afternoon_7d` | telephony | Float64 | dial_attempt | 7d | null when no attempts in that slot and window. | Answered share within the afternoon slot (IST) over 7d. |
| `answer_rate_evening_7d` | telephony | Float64 | dial_attempt | 7d | null when no attempts in that slot and window. | Answered share within the evening slot (IST) over 7d. |
| `weekend_attempt_share_7d` | telephony | Float64 | dial_attempt | 7d | null when no attempts in window. | Share of attempts on Sat/Sun (IST) over 7d. |
| `ring_seconds_mean_7d` | telephony | Float64 | dial_attempt | 7d | null when no attempts in window. | Mean ring_seconds over 7d. |
| `ring_seconds_std_7d` | telephony | Float64 | dial_attempt | 7d | null with fewer than 2 attempts in window. | Sample std of ring_seconds over 7d. |
| `short_ring_rate_7d` | telephony | Float64 | dial_attempt | 7d | null when no attempts in window. | Share of attempts with ring_seconds below configs short_ring_seconds. |
| `n_attempts_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Dial attempts with occurred_at in the last 14d. |
| `n_answered_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Attempts with network_response=answered in the last 14d. |
| `answer_rate_14d` | telephony | Float64 | dial_attempt | 14d | null when no attempts in window (never 0-imputed). | n_answered / n_attempts over the last 14d. |
| `n_switched_off_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Attempts with network_response=switched_off in the last 14d. |
| `n_not_reachable_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Attempts with network_response=not_reachable in the last 14d. |
| `n_does_not_exist_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Attempts with network_response=does_not_exist in the last 14d. |
| `n_no_answer_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Attempts with network_response=no_answer in the last 14d. |
| `n_busy_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Attempts with network_response=busy in the last 14d. |
| `n_attempts_morning_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Attempts placed in the morning slot (IST) in 14d. |
| `n_attempts_afternoon_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Attempts placed in the afternoon slot (IST) in 14d. |
| `n_attempts_evening_14d` | telephony | Int64 | dial_attempt | 14d | 0 when no attempts in window. | Attempts placed in the evening slot (IST) in 14d. |
| `answer_rate_morning_14d` | telephony | Float64 | dial_attempt | 14d | null when no attempts in that slot and window. | Answered share within the morning slot (IST) over 14d. |
| `answer_rate_afternoon_14d` | telephony | Float64 | dial_attempt | 14d | null when no attempts in that slot and window. | Answered share within the afternoon slot (IST) over 14d. |
| `answer_rate_evening_14d` | telephony | Float64 | dial_attempt | 14d | null when no attempts in that slot and window. | Answered share within the evening slot (IST) over 14d. |
| `weekend_attempt_share_14d` | telephony | Float64 | dial_attempt | 14d | null when no attempts in window. | Share of attempts on Sat/Sun (IST) over 14d. |
| `ring_seconds_mean_14d` | telephony | Float64 | dial_attempt | 14d | null when no attempts in window. | Mean ring_seconds over 14d. |
| `ring_seconds_std_14d` | telephony | Float64 | dial_attempt | 14d | null with fewer than 2 attempts in window. | Sample std of ring_seconds over 14d. |
| `short_ring_rate_14d` | telephony | Float64 | dial_attempt | 14d | null when no attempts in window. | Share of attempts with ring_seconds below configs short_ring_seconds. |
| `n_attempts_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Dial attempts with occurred_at in the last 30d. |
| `n_answered_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Attempts with network_response=answered in the last 30d. |
| `answer_rate_30d` | telephony | Float64 | dial_attempt | 30d | null when no attempts in window (never 0-imputed). | n_answered / n_attempts over the last 30d. |
| `n_switched_off_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Attempts with network_response=switched_off in the last 30d. |
| `n_not_reachable_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Attempts with network_response=not_reachable in the last 30d. |
| `n_does_not_exist_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Attempts with network_response=does_not_exist in the last 30d. |
| `n_no_answer_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Attempts with network_response=no_answer in the last 30d. |
| `n_busy_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Attempts with network_response=busy in the last 30d. |
| `n_attempts_morning_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Attempts placed in the morning slot (IST) in 30d. |
| `n_attempts_afternoon_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Attempts placed in the afternoon slot (IST) in 30d. |
| `n_attempts_evening_30d` | telephony | Int64 | dial_attempt | 30d | 0 when no attempts in window. | Attempts placed in the evening slot (IST) in 30d. |
| `answer_rate_morning_30d` | telephony | Float64 | dial_attempt | 30d | null when no attempts in that slot and window. | Answered share within the morning slot (IST) over 30d. |
| `answer_rate_afternoon_30d` | telephony | Float64 | dial_attempt | 30d | null when no attempts in that slot and window. | Answered share within the afternoon slot (IST) over 30d. |
| `answer_rate_evening_30d` | telephony | Float64 | dial_attempt | 30d | null when no attempts in that slot and window. | Answered share within the evening slot (IST) over 30d. |
| `weekend_attempt_share_30d` | telephony | Float64 | dial_attempt | 30d | null when no attempts in window. | Share of attempts on Sat/Sun (IST) over 30d. |
| `ring_seconds_mean_30d` | telephony | Float64 | dial_attempt | 30d | null when no attempts in window. | Mean ring_seconds over 30d. |
| `ring_seconds_std_30d` | telephony | Float64 | dial_attempt | 30d | null with fewer than 2 attempts in window. | Sample std of ring_seconds over 30d. |
| `short_ring_rate_30d` | telephony | Float64 | dial_attempt | 30d | null when no attempts in window. | Share of attempts with ring_seconds below configs short_ring_seconds. |
| `last_response_type` | telephony | string | dial_attempt | - | null when never attempted. | Most recent network_response (by occurred_at). |
| `consecutive_failures` | telephony | Int64 | dial_attempt | - | null when never attempted; 0 when the last attempt was answered. | Trailing run of non-answered responses since the last answer. |
| `consecutive_same_response` | telephony | Int64 | dial_attempt | - | null when never attempted. | Trailing run length of the latest network_response value. |
| `days_since_first_attempt` | telephony | Int64 | dial_attempt | - | null when never attempted. | Days from first visible attempt to as_of (date-based, UTC). |
| `days_since_last_attempt` | telephony | Int64 | dial_attempt | - | null when never attempted. | Days from last visible attempt to as_of. |
| `days_since_last_answer` | telephony | Int64 | dial_attempt | - | null when never answered. | Days from last answered attempt to as_of. |
| `days_since_last_rpc` | telephony | Int64 | disposition | - | null when no RPC disposition is visible. | Days from last RPC disposition to as_of. |
| `mean_gap_between_attempts_days` | telephony | Float64 | dial_attempt | - | null with fewer than 2 attempts. | Mean gap in days between consecutive visible attempts. |
| `system_fail_rate_on_last_attempt_day` | telephony | Float64 | dial_attempt | - | null when never attempted. | Portfolio-wide failure share on the IST date of this contact point's last attempt, so models can discount dialer outages. |
| `n_rpc` | disposition | Int64 | disposition | - | 0 when no visible dispositions. | Visible dispositions with value rpc (all-time). |
| `n_wrong_number` | disposition | Int64 | disposition | - | 0 when no visible dispositions. | Visible dispositions with value wrong_number (all-time). |
| `n_third_party` | disposition | Int64 | disposition | - | 0 when no visible dispositions. | Visible dispositions with value third_party (all-time). |
| `n_dispute` | disposition | Int64 | disposition | - | 0 when no visible dispositions. | Visible dispositions with value dispute (all-time). |
| `n_promise_to_pay` | disposition | Int64 | disposition | - | 0 when no visible dispositions. | Visible dispositions with value promise_to_pay (all-time). |
| `n_callback` | disposition | Int64 | disposition | - | 0 when no visible dispositions. | Visible dispositions with value callback (all-time). |
| `n_switched_off` | disposition | Int64 | disposition | - | 0 when no visible dispositions. | Visible dispositions with value switched_off (all-time). |
| `n_not_reachable` | disposition | Int64 | disposition | - | 0 when no visible dispositions. | Visible dispositions with value not_reachable (all-time). |
| `wrong_number_rate` | disposition | Float64 | disposition | - | null when no visible dispositions. | n_wrong_number / all visible dispositions. |
| `last_disposition` | disposition | string | disposition | - | null when no visible dispositions. | Most recent disposition value (by occurred_at). |
| `days_since_last_disposition` | disposition | Int64 | disposition | - | null when no visible dispositions. | Days from last visible disposition to as_of. |
| `remark_switchedoff_cue_count` | text | Int64 | disposition | - | 0 when no remarks mention it (never null when dispositions exist; 0 also when no dispositions). | Remarks matching switched-off phrasing (Hinglish patterns). |
| `remark_wrongnumber_cue_count` | text | Int64 | disposition | - | 0 when no match. | Remarks matching wrong-number phrasing. |
| `remark_thirdparty_cue_count` | text | Int64 | disposition | - | 0 when no match. | Remarks matching third-party-answer phrasing. |
| `remark_avoidance_cue_count` | text | Int64 | disposition | - | 0 when no match. | Remarks matching observable avoidance phrasing. |
| `switched_off_months_max` | text | Int64 | disposition | - | null when no duration phrase is found. | Max months extracted from phrases like 'number band hai 2 mahine se'. |
| `n_borrowers_sharing_cp` | shared | Int64 | contact_point_update | - | Always >= 1 for rows in the universe. | Distinct borrowers sharing this contact_point_ref within the same lender. |
| `is_shared` | shared | boolean | contact_point_update | - | Never null. | True when >1 borrower shares this ref within the lender. |
| `n_phone_cps_for_borrower` | shared | Int64 | contact_point_update | - | Always >= 1 for phone rows. | Phone contact points of this borrower known at as_of. |
| `cp_rank_within_borrower` | shared | Int64 | contact_point_update | - | 1-based; never null. | Rank of this contact point within the borrower by earliest-known time (ties broken by ref). |
| `is_primary` | shared | boolean | contact_point_update | - | Never null. | Latest contact_point_update is_primary when visible, else the contact_points table flag. |
| `connected_component_size` | shared | Int64 | contact_point_update | - | Always >= 1. | Size of the borrower's lender-local sharing component (borrowers linked by shared contact points). |
| `source` | record | string | contact_point_update | - | Never null; 'unknown' when neither table nor update gives one. | Latest update source when visible, else the contact_points table value. |
| `record_age_days` | record | Int64 | contact_point_update | - | Never null (falls back to first-seen time). | Days from contact-point creation (or first-seen) to as_of. |
| `days_since_last_update` | record | Int64 | contact_point_update | - | null when no update event is visible. | Days from last contact_point_update to as_of. |
| `n_updates` | record | Int64 | contact_point_update | - | 0 when no update event is visible. | Visible contact_point_update events (all-time). |
| `confirmed_by_payment` | record | boolean | payment, dial_attempt, disposition | - | False when there is no confirming evidence (never null). | True when a visible payment occurred within payment_confirmation_days after an answered call or RPC on this contact point. |
| `days_since_confirmed` | record | Int64 | payment | - | null when never confirmed. | Days from the latest confirming payment to as_of. |
| `other_lines_attempts_1d` | crossline | Int64 | dial_attempt | 1d | 0 when the borrower has no other phone lines with attempts. | Attempts on the borrower's OTHER phone contact points in 1d. |
| `other_lines_answered_1d` | crossline | Int64 | dial_attempt | 1d | 0 when none. | Answered attempts on the borrower's other phone lines in 1d. |
| `other_lines_answer_rate_1d` | crossline | Float64 | dial_attempt | 1d | null when other lines have no attempts in window. | Answer rate on the borrower's other phone lines over 1d. |
| `n_payments_1d` | crossline | Int64 | payment | 1d | 0 when no visible payments in window. | Borrower-level visible payments with occurred_at in 1d. |
| `other_lines_attempts_3d` | crossline | Int64 | dial_attempt | 3d | 0 when the borrower has no other phone lines with attempts. | Attempts on the borrower's OTHER phone contact points in 3d. |
| `other_lines_answered_3d` | crossline | Int64 | dial_attempt | 3d | 0 when none. | Answered attempts on the borrower's other phone lines in 3d. |
| `other_lines_answer_rate_3d` | crossline | Float64 | dial_attempt | 3d | null when other lines have no attempts in window. | Answer rate on the borrower's other phone lines over 3d. |
| `n_payments_3d` | crossline | Int64 | payment | 3d | 0 when no visible payments in window. | Borrower-level visible payments with occurred_at in 3d. |
| `other_lines_attempts_7d` | crossline | Int64 | dial_attempt | 7d | 0 when the borrower has no other phone lines with attempts. | Attempts on the borrower's OTHER phone contact points in 7d. |
| `other_lines_answered_7d` | crossline | Int64 | dial_attempt | 7d | 0 when none. | Answered attempts on the borrower's other phone lines in 7d. |
| `other_lines_answer_rate_7d` | crossline | Float64 | dial_attempt | 7d | null when other lines have no attempts in window. | Answer rate on the borrower's other phone lines over 7d. |
| `n_payments_7d` | crossline | Int64 | payment | 7d | 0 when no visible payments in window. | Borrower-level visible payments with occurred_at in 7d. |
| `other_lines_attempts_14d` | crossline | Int64 | dial_attempt | 14d | 0 when the borrower has no other phone lines with attempts. | Attempts on the borrower's OTHER phone contact points in 14d. |
| `other_lines_answered_14d` | crossline | Int64 | dial_attempt | 14d | 0 when none. | Answered attempts on the borrower's other phone lines in 14d. |
| `other_lines_answer_rate_14d` | crossline | Float64 | dial_attempt | 14d | null when other lines have no attempts in window. | Answer rate on the borrower's other phone lines over 14d. |
| `n_payments_14d` | crossline | Int64 | payment | 14d | 0 when no visible payments in window. | Borrower-level visible payments with occurred_at in 14d. |
| `other_lines_attempts_30d` | crossline | Int64 | dial_attempt | 30d | 0 when the borrower has no other phone lines with attempts. | Attempts on the borrower's OTHER phone contact points in 30d. |
| `other_lines_answered_30d` | crossline | Int64 | dial_attempt | 30d | 0 when none. | Answered attempts on the borrower's other phone lines in 30d. |
| `other_lines_answer_rate_30d` | crossline | Float64 | dial_attempt | 30d | null when other lines have no attempts in window. | Answer rate on the borrower's other phone lines over 30d. |
| `n_payments_30d` | crossline | Int64 | payment | 30d | 0 when no visible payments in window. | Borrower-level visible payments with occurred_at in 30d. |
| `days_since_last_payment` | crossline | Int64 | payment | - | null when no visible payment. | Days from last visible borrower payment to as_of. |
| `days_since_last_other_line_answer` | crossline | Int64 | dial_attempt | - | null when no other line was ever answered. | Days from the last answered attempt on any OTHER phone line to as_of. |
| `n_visits_locked_premises` | field | Int64 | field_visit | - | null for phone contact points; 0 for addresses with no such outcome. | Visible field visits with outcome=locked_premises. |
| `n_visits_nobody_of_that_name` | field | Int64 | field_visit | - | null for phone contact points; 0 for addresses with no such outcome. | Visible field visits with outcome=nobody_of_that_name. |
| `n_visits_met_borrower` | field | Int64 | field_visit | - | null for phone contact points; 0 for addresses with no such outcome. | Visible field visits with outcome=met_borrower. |
| `n_visits_met_third_party` | field | Int64 | field_visit | - | null for phone contact points; 0 for addresses with no such outcome. | Visible field visits with outcome=met_third_party. |
| `n_visits_address_not_found` | field | Int64 | field_visit | - | null for phone contact points; 0 for addresses with no such outcome. | Visible field visits with outcome=address_not_found. |
| `n_visits` | field | Int64 | field_visit | - | null for phone contact points. | Visible field visits (all-time). |
| `last_visit_outcome` | field | string | field_visit | - | null for phones or when never visited. | Most recent visit outcome (by occurred_at). |
| `days_since_last_visit` | field | Int64 | field_visit | - | null for phones or when never visited. | Days from last visible visit to as_of. |
| `gps_dwell_mean_seconds` | field | Float64 | field_visit | - | null for phones or when no dwell recorded. | Mean dwell_seconds across visible visits. |
| `visit_hour_mean` | field | Float64 | field_visit | - | null for phones or when never visited. | Mean visit hour in IST (circular mean is NOT used; plain mean, documented as approximate). |
| `dpd_bucket` | account | string | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (dpd_bucket). |
| `outstanding` | account | Float64 | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (outstanding). |
| `product` | account | string | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (product). |
| `bureau_score_band` | account | string | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (bureau_score_band). |
| `income_type` | account | string | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (income_type). |
| `preferred_language` | account | string | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (preferred_language). |
| `town_id` | account | string | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (town_id). |
| `dpd_start` | account | Int64 | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (dpd_start). |
| `overdue_start` | account | Float64 | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (overdue_start). |
| `emi_amount` | account | Float64 | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (emi_amount). |
| `other_active_loans` | account | Int64 | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (other_active_loans). |
| `paid_other_lenders_30d` | account | boolean | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (paid_other_lenders_30d). |
| `last_bounce_reason` | account | string | tables/calendar | - | Null when the borrower row is absent or the source column is. | Account context passthrough (last_bounce_reason). |
| `has_any_attempt` | core | boolean | dial_attempt | - | Never null. | True when any dial attempt is visible for this contact point. Distinguishes 'no evidence' from measured zeros. |
| `contact_point_type` | core | string | contact_point_update | - | Never null. | phone or address for this contact point. |
| `asof_weekday` | calendar | Int64 | tables/calendar | - | Never null. | as_of weekday in IST (Monday=0). |
| `asof_day_of_month` | calendar | Int64 | tables/calendar | - | Never null. | as_of day of month in IST. |
| `is_holiday` | calendar | boolean | tables/calendar | - | Never null. | True when the as_of IST date is in configs holidays. |
| `agent_wrong_number_rate` | agent | Float64 | disposition | - | null when the last disposition has no agent or the agent has no history. | Wrong-number share of the agent who recorded the latest disposition (feature of disposition reliability; only present when agent_id is observed). |
<!-- REGISTRY:END -->
