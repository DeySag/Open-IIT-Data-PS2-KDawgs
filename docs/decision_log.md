# Decision log

## 2026-10-04 — Decision layer v0 (workstream D, day 1)

- Decision: guardrails evaluated first and restrict-only; hard blocks park on
  switch_channel/field and never trace. Reason: compliance hard requirement;
  four-action contract forces a next action, so parking (not executed
  downstream) is the fail-safe. Alternative (empty action) rejected: violates
  no-dead-ends rule.
- Decision: recycled cost-ratio cutoff 1/(1+100) applied to the classifier
  score, not background posterior mass. Reason: preserves dialling on healthy
  lines while keeping fast-path suppression aggressive where it matters.
- Decision: VOI conditions on P(dead), avoiding accounts get ~zero VOI and
  never queue. Reason: tracing an avoider finds nothing new (incremental value).
- Decision: greedy-by-ratio knapsack with documented exactness note over an
  exact DP. Reason: day-1 scale and auditability; DP is a day-3 option.
- Decision: GUARDRAIL_* reason codes kept in decision layer, coerced at the
  boundary; contracts untouched. Reason: frozen-contract rule; flagged the
  enum addition for coordinator approval instead of stopping (outputs valid).
- Decision: stub evidence UUID5 placeholders; `no_viable_contact_point`
  placeholder for contract min_length=1. Reason: run without model/feedback
  layers; serving layer must reconcile. All simulation-only.
