# P4 report: state-tracker refit at account grain

Primary (purged) validation = 3 rolling origins (06-03/10/17), 14d train lookback for baselines, 3d embargo, 7d horizon, VAL-split accounts only (360 accts, 878 refs), dialled-only labels (rpc = answered OR rpc-family disposition). EM fit TRAIN-split only (1680 accts, <=2026-05-26).

## Validation discrimination (primary)

| model | n | auc | pr_auc | brier | ece | rpc/100 |
|---|---|---|---|---|---|---|
| state_tracker | 603 | 0.5778 | 0.5417 | 0.3919 | 0.3891 | 44610.0 |
| contact_gbm | 603 | 0.6507 | 0.5873 | 0.2355 | 0.0672 | 44620.0 |
| incumbent | 603 | 0.5936 | 0.4952 | 0.2777 | 0.1828 | 44440.0 |
| account_gbm | 603 | 0.5619 | 0.5037 | 0.2469 | 0.0892 | 44550.0 |

RESULT: contact_gbm 0.6507 > incumbent 0.5936 > state_tracker 0.5778 > account_gbm 0.5619.
Tracker beats 1 of 3 on loose RPC AUC; does NOT meet 'beats all three' on this metric. See verified-dead win below (0.5853 vs 0.5466/0.4812/0.3904) and attribution.

## Test (final only, single origin 06-17, dialled-only)

| model | n | auc |
|---|---|---|
| state_tracker | 170 | 0.59 |
| contact_gbm | 170 | 0.6208 |
| incumbent | 170 | 0.6074 |
| account_gbm | 170 | 0.5251 |

Test ordering matches validation: contact 0.6208 > incumbent 0.6074 > tracker 0.59 > account 0.5251.

## Verified gold (final only, PIT 06-29, n=157: 30 dead vs 127 borrower_number)

tracker dead(rec+inv+off) 0.5853 > account_gbm 0.5466 > contact_gbm 0.4812 > incumbent 0.3904. Tracker WINS dead-state detection on held-out gold.

## Avoiding-vs-invalid: thin support on issued extracts (proxy n=14; tracker auc 0.36, no separating power after unconstrained EM collapsed avoidance persistence).
Mechanism verified on micro-fixtures instead (tests green): silent+answering-sibling -> dead 0.66 vs all-silent 0.08; latent-off ablation diff 0.001; PIT bit-identical. Constrained-EM sensitivity (T_A frozen at prior): validation AUC 0.537 < 0.578 shipped; avoidance preserved structurally but ranking worse - documented, not shipped.

## Attribution (dAUC vs full selected base/k1.5/reset0.9):
latent-off (pooling+silence+cross-resets) -0.0116; no-DPD -0.0043; no-silence -0.0031; no-resets -0.0016. Largest lever = borrower-latent pooling block; top scoring knob = DPD multipliers, then silence kappa, then reset strength.

## Refit record

config_hash 52ffe911ad31d894, iters 12, timing 101.5s, window 2026-04-01 02:32:07+00:00..2026-05-25 17:01:06+00:00 (fit cap 2026-05-26), n_events 35695, borrowers(account grain) 1680.
Prior-vs-data MAE: T_S 0.062, T_A 0.396 (collapsed - see sensitivity note), E_net 0.122, E_disp 0.109.

## Registration
try_register_eval True; scorer 'state_tracker' (protocol columns incl. 7 posteriors + recycled_risk + confidence). Ablation + PIT tests green (21 passed).

## Files
p4_tracker_final/ (selected cfg + TRAIN-fit params), p4_selected_config.yaml, p4_refit_record.json, p4_report.json, p4_verified_scores.pkl