# Decision log

## 2026-10-04 — State tracker v0: joint (A,S) filter + pooling + silence shift (B-workstream)

- Decision: per-line joint 12-state (A,S) forward filter; borrower sharing via
  (i) reset events, (ii) naive-Bayes A-pooling with prior correction,
  (iii) asymmetric-silence explained-away shift (σ = 1−exp(−W/κ), κ=3.0).
- Reason: the filter's unconditional network emission cannot express
  "silent while the borrower is active elsewhere"; without (iii) a silent line
  with an answering sibling scores valid_reachable 0.70 (verified). With (iii):
  dead 0.66 vs 0.08 all-silent; ablation (latent off) diff 0.001.
- Alternatives: full joint forward over A×S₁×…×S_K (exact, but 2·6^K states —
  intractable past K=3 at 100k-borrower scale); two-round variational loop
  (rejected: does not fix the emission-contrast problem either).
- Leakage: config priors are coarse qualitative numbers, deliberately
  different from configs/sim.yaml; EM reads events only (ground-truth columns
  dropped defensively in parse_events).

## 2026-10-04 — Blockers outside ownership (reported, not fixed)

- `pyproject.toml:36` (`per-file-ignores = {"tests/*": ...}`) is rejected by
  this env's tomllib, so `pytest`/`make test` fail at config parse. Ran tests
  with `pytest -c NUL -p no:cacheprovider`. Needs coordinator/workstream-F fix.
- `src/rpc/sim/generate.py:107` crashes on numpy 2.x
  (`timedelta(days=rng.integers(...))` → TypeError), so no dev/full simulator
  data could be generated; dev-scale timing used the self-contained mini
  generator instead. Needs workstream-A fix.
- `git push origin main` → 403 (permission denied); commits are local only.
  `src/rpc/models` is covered by the `models/` gitignore rule (model
  artifacts); files were added with `git add -f`. Suggest narrowing to
  `/models/` — needs coordinator approval.
- No `src/rpc/eval/` harness on main yet: `try_register_eval()` returns False
  without crashing; slow eval test skips until simulator ground-truth outputs
  exist.
