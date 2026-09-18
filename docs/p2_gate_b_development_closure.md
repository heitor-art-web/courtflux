# P2 Gate B — development closure record

Date: 2026-09-18

## Decision

Gate B is **closed for development progression by human review** so work may proceed to Gate C.

This is a narrow development decision, not a change to the automated Gate B contract and not a waiver for the final P2 Exit Gate.

The frozen automated Gate B scoreboard remains authoritative for the measured result:

- automated development status: **FAIL**;
- human development decision: **PASS to proceed to Gate C**;
- reason for the difference: shot-attempt recall is a one-TP near-miss at `11 / 14 = 0.7857` against the frozen `>= 0.80` threshold, while every other required Gate B check passes.

No threshold is changed and no prediction/GT is edited to manufacture a passing automated result.

## Frozen development result

Development clips: V1, V3, V4, V5.

V2 remains challenge-only and non-blocking.

Latest frozen Gate B development result used for this decision:

```text
ground_truth_shots: 14
predicted_shots:    13
TP:                  11
FP:                   2
FN:                   3
precision:            0.8462  PASS  (>= 0.80)
recall:               0.7857  FAIL  (>= 0.80)
outcome accuracy:     0.8182  PASS  (>= 0.75)
duplicate attempts:   0       PASS
```

The minimum discrete improvement required for automated recall PASS would be one additional true positive: `12 / 14 = 0.8571`.

The observed recall deficit is therefore one GT shot and `0.0143` absolute below the frozen threshold.

## FN review and rejected tuning

All three development false negatives were investigated before accepting the human development closure.

### V1 — boundary shot, GT 0–50

The clip begins during/around the human shot window. Diagnostic review showed no reliable shot-start evidence in the window:

- no `relative_above` evidence;
- no `relative_near_x` evidence;
- no `rim_band` evidence;
- no `inside_mouth` evidence;
- no semantic shot-start evidence;
- rim evidence appears only after the beginning of the interval.

A clip-boundary-specific recovery rule was rejected because the required causal evidence is not present.

### V3 — alley-oop, GT 689–778

The interval contains strong airborne/flight activity, but the shot tracker receives no usable shot-start evidence:

- no `relative_above`;
- no `relative_near_x`;
- no `rim_band`;
- no `inside_mouth`;
- no semantic shot-start signal.

Treating long flight alone as a shot-start condition would collapse passes into shots, so that remediation was rejected.

### V4 — GT 660–839

Several candidate recovery rules were audited and rejected.

`inside_mouth + flight_supported` produced multiple false start regions outside GT. A normalized vertical-distance rule also produced many outside-GT candidates. Flight-supported transition timing did not cleanly separate the FN from non-shot airborne episodes.

Because no general causal discriminator separated the FN from false candidates, no threshold or start heuristic was added solely to gain the missing TP.

## Why human development PASS is acceptable

The development decision is based on the full Gate B evidence, not on rounding the metric:

- four of five required automated checks pass;
- precision remains above contract at `0.8462`;
- matched shot outcome accuracy remains above contract at `0.8182`;
- duplicate shot attempts remain zero;
- the only failed check is recall at `0.7857`, one TP short of the discrete passing count;
- targeted investigation of all three FNs found no safe general remediation using the current evidence model;
- proposed heuristics that would have forced the missing TP were explicitly rejected when they created false positives or relied on clip-specific behavior.

The human decision therefore closes Gate B **for development sequencing only** and records shot-start recall as residual technical risk to be challenged again by unseen final validation.

## V2 challenge status

V2 remains a challenge/outlier development clip and is always reported separately. Its poor shot-detection result is not hidden and does not block development Gate B under the frozen data-role policy.

No Gate B tuning decision is made from V2.

## Final Exit Gate obligation

V6 and V7 remain untouched and unseen during Gates A–D.

After Gates A–D are closed and the P2 stack is frozen, V6/V7 are run once as the final P2 Exit Gate.

The human development override recorded here does **not** automatically carry into the final Exit Gate. The final unseen evaluation must re-test the frozen numerical Gate A–D contracts, including Gate B shot-attempt recall `>= 0.80`.

If V6/V7 expose a defect that requires a model, feature, policy or threshold change, those videos are considered spent for final-validation purposes and a later independent final validation requires fresh unseen clips.

## Development status after this decision

```text
Gate A: CLOSED — automated PASS
Gate B: CLOSED FOR DEVELOPMENT — automated near-miss / human PASS
Gate C: NEXT
Gate D: pending
Final P2 Exit Gate: pending — V6/V7 unseen
```
