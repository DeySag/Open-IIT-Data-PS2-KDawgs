# Evaluation harness (simulation-only)

All numbers produced here come from synthetic data. Never describe any result
as real-world performance. Every report table carries the
`SIMULATION-ONLY - synthetic data, not real-world performance` label.

## Entry point

```sh
make eval          # python -m src.rpc.eval.run --config configs/eval.yaml
```

Output: `reports/eval_<timestamp>.md` + `.json`, one comparison table per
metric family (baselines vs any registered model), with n and bootstrap CIs.

## Interfaces (frozen for other workstreams)

- `src/rpc/eval/protocol.py` — `Scorer` protocol:
  `score(as_of, contact_point_refs) -> DataFrame[contact_point_ref, p_rpc,
  optional state posteriors, recycled_risk, confidence]`.
- `src/rpc/eval/registry.py` — `register_scorer(name, factory)`,
  `get_scorer(name)`, `list_scorers()`. Other agents plug models in by name;
  `run.py` scores every registered baseline plus `oracle`/`random` diagnostics.
- Features: `get_feature_builder()` in `_minifeatures.py` returns an adapter
  around the real `src.rpc.features.build_features` (in-memory event source,
  PIT-correct; mini columns backfilled only where the real output lacks them
  for the baselines contract); until the real layer landed it was the TEMPORARY
  DuckDB mini-features (attempt counts, answer rate, recency, network-response
  counts, source/primary) built PIT-correct (`received_at <= as_of`).

## Splits, labels, metrics

- Rolling-origin splits by `as_of` with embargo gap (`configs/eval.yaml`);
  never random k-fold. `check_splits` enforces embargo + advancing origins.
- Labels: observed `rpc_next_7d` among DIALLED contact points (`censored`
  otherwise, always reported dialled-only) vs oracle (true state/reachable,
  eval-only read of `data/ground_truth.parquet`).
- Metrics (`metrics.py`): AUC, PR-AUC, Brier, log-loss; reliability table + ECE
  overall (segment split helper `ece_by_segment`); recycled precision/recall at
  operating points + cost-weighted loss (cost ratio from guardrails);
  RPC/1,000 dials, wasted attempts on dead lines, detection-within-k,
  coverage/orphaned; avoiding-vs-invalid confusion + AUC on silent lines plus
  cross-line subset; IPW second view via `policy_log.parquet` when present.
- Only `src/rpc/eval` may read `ground_truth.parquet` / `policy_log.parquet`
  (leakage test enforces this at file level; baselines are scanned too).

## Current status (2026-10-04)

- Sim v0 `generate.py` crashes (`timedelta` + `numpy.int64`, reported to sim
  workstream, not patched: outside ownership). Dev data for the first green
  run was produced by a throwaway script mirroring sim v0 schema (30-day
  horizon, 5k borrowers / 13k contact points / 66.5k events); `data/` is
  gitignored. No `ground_truth.parquet` / `policy_log.parquet` exist yet, so
  oracle and IPW tables are skipped with explicit notes.
- Incumbent policy params live in `configs/eval.yaml` (`incumbent:`) because
  `configs/sim.yaml` has no `policy:` section in sim v0 (assumed values).
