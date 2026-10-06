# State tracker: contact-point HMM + borrower avoidance latent

All numbers below state the model design; every fitted parameter must be
estimated on issued extracts before use.

## 1. Problem

A borrower who will not answer and a dead number both look like "no answer"
but need opposite actions (switch channel vs trace). The shared borrower latent
separates them: if the borrower answers or pays through another route while one
line stays silent, suspicion moves to that line; silence on every active line
points to avoidance.

## 2. Model

- Per phone line k, hidden persistent state S in {valid, temp_unreachable,
  switched_off_long, recycled, third_party, invalid}, daily transitions T_S.
  `invalid` is absorbing; `recycled` can only go to `recycled`/`invalid`
  (a recycled number never becomes the borrower's live line again); dormancy
  precedes recycling (`switched_off_long -> recycled` allowed).
- Per borrower, hidden avoidance A in {0, 1} with daily transitions T_A
  (episodes persist for weeks), entry rate optionally scaled by DPD bucket
  (monotone coarse multipliers in config, weak magnitude).
- Each line is filtered as a JOINT 12-state chain over (A, S) with factorised
  transitions P(a'|a)·P(s'|s). Observations per line per day: network responses
  (7 codes), dispositions after answered calls (8 codes), bot-transcript cue
  hits (v0 keyword hook), borrower-level payments (strong A=0 evidence).
  Emissions depend on (S, A): e.g. `immediate_hangup`/`no_answer` inflate when
  (valid, avoiding); `does_not_exist` lives on recycled/invalid; `answered` is
  common when (valid, calm) and rare when avoiding.
- Reported 7-key posterior: avoiding = P(S=valid, A=1),
  valid_reachable = P(S=valid, A=0), other states marginalised over A.
- p_rpc = P(valid, A=0) × answer_given_valid_calm × slot_multiplier (hook for
  the slot-GBM workstream; default 1.0). recycled_risk = P(recycled).
- Confidence = (1 − H/Hmax) × exp(−gap/τ): transitions diffuse the posterior
  while no evidence arrives (no evidence is not negative evidence); a gap with
  no events raises entropy and lowers confidence.

## 3. Cross-line sharing (the avoiding-vs-invalid separator)

Three mechanisms, all borrower-level:

1. **Resets.** An rpc/promise_to_pay disposition on any line, or a payment,
   moves most A=1 mass to A=0 on every sibling line's filter (strengths
   `avoid_reset` in config). Own-line rpc additionally collapses that line to
   S=valid.
2. **A-pooling.** At scoring, sibling lines' A marginals are combined by
   naive-Bayes with prior correction; each line's S|A conditionals are then
   re-weighted by the pooled A. Soft evidence (e.g. a sibling that keeps
   getting answered) therefore pulls a silent line's silence toward a
   line-explanation rather than avoidance.
3. **Asymmetric-silence (explained-away) shift.** The filter's network emission
   is unconditional, so it cannot express this on its own: a line that rings
   `no_answer` is *compatible* with (valid, calm). But a line dialled and
   unanswered on days the borrower is demonstrably active elsewhere (rpc,
   payment, or an answered call on any line) is suspicious. For each such
   silent-on-active-day dial weight W, fraction σ = 1−exp(−W/κ) of the line's
   valid mass moves to dead states pro-rata (κ = `silence_shift_kappa`).
   Days with no events anywhere never trigger it.

Justification for (3) being a rule rather than learned: it is the conditional
likelihood P(line silent | S, borrower active elsewhere), which the
unconditional per-line emission cannot represent; learning it would need
joint multi-line EM. It is documented, configured (not hardcoded), and covered
by the ablation test. With `use_borrower_latent: false`, (2) and (3) are off
and only own-line resets plus borrower payments are shared: silent lines then
score identically regardless of siblings (verified in tests).

## 4. Hard-evidence overrides (rules before the probabilistic update)

- `does_not_exist`: shift 55% of live-state mass to invalid (60%) / recycled
  (40%). Rationale: the network asserts the number does not exist; agent-free
  signal, high precision.
- Explicit `wrong_number` disposition: shift 45% toward recycled/invalid.
  Weaker than the network signal because dispositions are noisy (agent error,
  borrower misdirection).
- rpc/promise_to_pay answer: collapse own line to valid; borrower-level reset.
- Payment: borrower-level reset (0.95). A payment proves the borrower is alive
  and reachable through some route.

## 5. Priors (configs/state_tracker.yaml) — qualitative, leakage-free

Every number is a round, coarse judgement, deliberately NOT copied from any
data-generating process (copying data-derived numbers would be leakage and
would overstate real-world performance). Dirichlet strengths are weak
(transition 20, emission 10, initial 5) so data dominates quickly.

| Choice | Value shape | Domain reason |
|---|---|---|
| initial valid 0.60 | majority live at first sight | most records are usable |
| initial invalid 0.10, recycled 0.05 | visible dead-on-arrival minority | stale files, churn |
| T_S diagonals ≥0.80–0.93 | states persist day to day | numbers don't change state often |
| invalid absorbing | dead stays dead | — |
| recycled→invalid 0.07 only | near-absorbing | re-classification, not recovery |
| T_A [[0.97,0.03],[0.05,0.95]] | avoidance episodes last weeks | behavioural persistence |
| does_not_exist on invalid 0.65 / recycled 0.40 | network truthfulness | — |
| answered on (valid, calm) 0.40 vs (valid, avoiding) 0.03 | screening | the core signal |
| wrong_number on recycled 0.55 / invalid 0.45 | new subscriber / dead record says so | — |
| source dead-multiplier bureau 1.4, borrower_on_call 0.4 | provenance trust | confirmed-by-contact records are cleaner |

## 6. Learning: EM (Baum-Welch) on events only

Forward-backward per line on the 12-state joint chain; M-step recovers T_A
(marginalising over S), T_S (marginalising over A), E_net, E_disp, pi with
fixed Dirichlet priors from §5. `e_pay`, overrides, and the p_rpc head stay
fixed. Approximations: (a) lines are independent sequences in the E-step (no
sibling pooling while learning); (b) override/reset non-linearity is applied
in the forward pass but ignored in the backward pass; (c) day gaps are folded
into J^gap (gap capped at 60: beyond that the chain is ~stationary);
(d) payments attach to every line's filter (mild double counting across
siblings, accepted for tractability). Discrete-time daily grid; the hazard
fallback was not needed (EM converges on dev-scale data; see timings below).

## 7. Point-in-time correctness

`score(as_of)` filters to `received_at <= as_of` and diffuses each filter from
its last event day to `as_of`. `received_at` (knowledge time) is the cutoff
per A6. Tested: identical params + extra future events ⇒ bit-identical scores
at T; future events still move later scores.

## 8. Addresses (thin stub, phase 2)

Address contact points return the prior posterior with confidence 0 and a
small stub p_rpc. No address states are inferred in v0.

## 9. Interfaces

- `StateTracker.fit(events_df, config)`, `.score(as_of, contact_point_refs)`,
  `.save(path)/.load(path)` (params.npz + events.parquet/pickle + meta.json).
- `StateTrackerScorer.score_df(as_of, refs)` → DataFrame(columns:
  contact_point_ref, p_rpc, state_posterior, recycled_risk, confidence,
  sp_<7 states>).
- `try_register_eval()` registers with `src.rpc.eval.registry` if present
  (absent on main at time of writing → returns False; no crash).

## 10. Verification status

- Unit/micro-fixture tests: 11 passed (see REPORT below).
- Parameter recovery on a self-contained mini generator: T_S MAE < 0.15,
  T_A MAE < 0.10, E_net MAE < 0.15, decode accuracy above chance + 0.25.
- Timings on dev-scale invented fixture data (240k dial obs, 6k borrowers,
  self-contained mini generator): EM fit 3 iters 39 s (≈2.5 min for the
  configured 12 iters); scoring 10k lines 24 s. `max_fit_borrowers: 5000`
  caps fit cost on larger data.
- Comparison vs baselines on the official extracts: pending mapping work;
  the slow eval test is written and skips cleanly until annotated extracts
  are present (the 250-row verified set is the current gold candidate).
- Unverified: calibration on real-shaped traffic, DPD-multiplier magnitudes,
  transcript-cue weights (placeholders), slot-multiplier hook (default 1.0).
