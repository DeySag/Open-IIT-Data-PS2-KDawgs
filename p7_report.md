# P7 report: decision layer on real outputs with real costs

Frozen policy: avoidance_thr 0.5, third_party_thr 0.5, min VOI/rupee 2.0, gain x0.34 (self-cure 35/53), recycled ratio 100. Guardrails evaluated FIRST (engine construction); models narrow only.

## Action mix (all four actions, no collapse; no dead ends)
TRAIN (n=3360): {'trace': 0.564, 'switch_channel': 0.344, 'continue': 0.091, 'switch_contact_point': 0.001} | VAL (n=1080): {'trace': 0.594, 'switch_channel': 0.334, 'continue': 0.069, 'switch_contact_point': 0.002} | TEST (n=360): {'trace': 0.575, 'switch_channel': 0.358, 'continue': 0.067}
switch_contact_point ~0.1-0.3% is structural (fires only when dead-best coexists with a healthy alternative, which is otherwise ranked first).
Top reasons TRAIN: {'INVALID_TRACE': 1862, 'LOW_VOI_DEPRIORITISED': 1011, 'VALID_CONTINUE': 283, 'GUARDRAIL_TRACE_PENDING': 66, 'GUARDRAIL_SUPPRESSED': 50, 'SWITCHED_OFF_MOVE_OR_TRACE': 36, 'AVOIDING_SWITCH_CHANNEL': 27, 'DEFERRED_TRACE_WAIT': 17}

## VOI vs 766 real trace outcomes
Queue enrichment (TRAIN): queued-trace precision 0.32 vs trigger-rule base 0.219 (1.46x). Ranking among triggered: null - per-trace VOI/rupee AUC 0.506 (n=761; TRAIN 0.510, VAL 0.459, TEST 0.527); outstanding 0.509, p_dead 0.500. Yield is unpredictable from dial-history inputs on the trigger-selected slice; policy value = cost-awareness + budget discipline + compliance, not yield prediction.
Observed cost per found contact: Rs 438 overall (L03 269 ... L04 836) - per-lender tables matter for efficiency.

## Calibration (P5)
ECE 0.3879 -> 0.0695, Brier 0.3864 -> 0.2398 (VAL 06-03, n labelled, lender segments; L04/L05 global fallback).
## Suppression
Weak-evidence review rate: TRAIN 0.039, VAL 0.0426, TEST 0.0278. Verified (measure-only): dead recall 0.0667, live false-suppress 0.0236. Fast-path wired (rules 1/2/4) + store idempotent + removal-requires-CN-signoff; unit tests green.

## Sensitivities
No-snapshot VAL: {'switch_channel': 0.914, 'continue': 0.083, 'switch_contact_point': 0.003} (nothing clears the TRAIN-set gate -> parked; gate binds). Out-of-hours: {'switch_channel': 0.906, 'continue': 0.092, 'switch_contact_point': 0.003}, traces=0, CONTACT_HOURS fired=True (guardrails-first proven).

## Unevaluable (logged, not assumed)
consent/DND/dispute/deceased columns absent: flags default False; those guardrails unevaluable here (logged, not assumed); account snapshot as-of unconfirmed: primary WITH snapshot (flagged) + without-snapshot sensitivity; no whatsapp opt-in column: False (conservative); address_state stub: None

## Files
p7_report.json, p7_eval.json, configs/costs.yaml (per_lender), src/rpc/decision/voi.py (resolve_costs_for_lender, scale_recovery_gain), tests/test_decision_p7.py (6 passed).