# Decision log

## 2026-10-04 — eval: incumbent policy params live in configs/eval.yaml
- Reason: task pointed at a `policy:` section in `configs/sim.yaml`; sim v0 has none. Editing sim.yaml is outside eval ownership, so assumed values (k=3, trace-after=6, primary-first) live under `incumbent:` in `configs/eval.yaml`.
- Alternatives: patch sim.yaml (rejected: other workstream owns it).

## 2026-10-04 — eval: temporary DuckDB mini-features in src/rpc/eval/_minifeatures.py
- Reason: `src/rpc/features` (build_features/spec.py) has not landed. Clearly marked temporary; `get_feature_builder()` auto-switches to the real layer when importable.
- Alternatives: block until features land (rejected: parallel-work instruction says stub, don't wait).

## 2026-10-04 — eval: oracle/random scorers live in src/rpc/eval/reference.py, not baselines/
- Reason: keeps the leakage guard ("baselines never read ground truth") a trivial file-content check.
- Alternatives: put them beside baselines with an allowlist (rejected: weaker guarantee).

## 2026-10-04 — eval: dev data from throwaway generator, not sim v0
- Reason: sim v0 `generate.py` crashes (numpy.int64 in timedelta). Reported to sim workstream; data/ is gitignored so a same-schema stand-in unblocks the green run.
- Alternatives: patch sim (rejected: outside ownership).

## 2026-10-04 — eval: pre-existing pyproject.toml breaks bare pytest collection
- Reason: toml parse error at line 36 (unrelated to eval). Tests verified with `python -m pytest -c NUL -p no:cacheprovider`. Flagged to coordinator; not patched (outside ownership).

## 2026-10-04 — eval: root .gitignore `eval/` + `models/` also ignore src/rpc/eval and src/rpc/models
- Reason: patterns meant for top-level artifact dirs match our source dirs. Did NOT edit .gitignore (outside ownership); staged owned files with `git add -f`. Coordinator should scope those patterns (e.g. `/eval/` `/models/`).
- Alternatives: leave files uncommitted (rejected: deliverable must be in git).
