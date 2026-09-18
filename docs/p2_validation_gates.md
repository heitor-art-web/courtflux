# P2 validation gates — P3 readiness contract

## Purpose

P2 exists to provide a basketball timeline that P3 can aggregate, search, clip and explain. Validation therefore measures whether P2 understands the semantic sequence of the play, not whether every human interval boundary is reproduced frame-perfectly.

The data contract is:

- V1–V5: development/training corpus for AirborneClassifierV1;
- grouped leave-one-video-out cross-validation is used inside V1–V5;
- the final AirborneClassifierV1 artefact is fitted on all V1–V5 only after development choices are frozen;
- V6–V7: new, unseen validation videos for the final P2 gates.

V6–V7 must not be used to select features, weights, model family, confidence threshold or gate thresholds.

The validation order remains A → B → C → D.

---

## Gate A — Timeline readiness

### Product question

> Can P3 trust the P2 control timeline to find possessions, passes and shots in the right semantic order without systematic collapse on a normal clip?

### Primary scoring unit

Human `control_intervals`, matched one-to-one to predicted control-state intervals. Human boundaries are windows, not millisecond release truth. A prediction of the same semantic state is a detection when its interval overlaps the human window or is within the frozen causal tolerance.

Frame accuracy, dominant-frame accuracy and exact boundary overlap are diagnostics only.

### Required checks

Frozen before V6–V7 validation:

- GT-window prediction coverage: aggregate >= 0.98; every clip >= 0.95.
- Core interval detection recall (`possessed`, `pass_flight`, `shot_flight`): aggregate >= 0.70.
- Possessed interval detection recall: aggregate >= 0.70.
- Pass interval detection recall: aggregate >= 0.60.
- Shot interval detection recall: aggregate >= 0.70.
- Semantic transition **order accuracy**, conditioned on both adjacent GT episodes being correctly detected: aggregate >= 0.90.
- Core unresolved interval rate: aggregate <= 0.25.
- PASS truth resolved as SHOT: <= 0.15.
- SHOT truth resolved as PASS: <= 0.15.
- Per-clip core interval detection recall >= 0.40.
- Per-clip core unresolved interval rate <= 0.45.

`loose` remains reported but is not a required Gate-A check until sufficient independent human loose-ball intervals exist.

### Interval resolution contract

For each GT core interval:

1. match one predicted interval of the same state when overlap/distance is within tolerance;
2. otherwise identify the nearest competing core-state prediction within tolerance;
3. if a competing state exists, mark the GT interval as `wrong_type`;
4. otherwise mark it `unresolved`.

Same-state matching is one-to-one so one long prediction cannot satisfy several independent GT episodes.

### Sequence contract

Interval recall already penalizes missing episodes. Sequence scoring therefore does **not** penalize the same missing interval again.

For every pair of adjacent GT core intervals:

1. if either endpoint was not correctly detected, the transition is not eligible for order scoring;
2. if both endpoints were correctly detected, compare the matched prediction order;
3. `semantic_transition_order_accuracy = correctly_ordered / eligible_detected`.

The evaluator also reports `detected_transition_coverage = eligible_detected / total_gt_transitions` as a diagnostic. It is not a second Gate blocker.

### Diagnostics, not blockers

- frame accuracy;
- dominant predicted state inside a GT interval;
- mean frame matching rate;
- start/end boundary errors;
- interval IoU/overlap ratio;
- detected-transition coverage.

---

## AirborneClassifierV1 development contract

The classifier is causal and separate from the P2 event engine.

It is trained as two stages:

```text
active P2 flight hypothesis
        ↓
Stage 1: AIRBORNE vs CONTINUITY
        ↓ AIRBORNE
Stage 2: PASS vs SHOT
        ↓
PASS_FLIGHT / SHOT_FLIGHT
```

Stage 1 exists because the dominant development error is not only PASS-vs-SHOT: human airborne episodes are often published as `POSSESSED` continuity.

Stage 2 trains only on GT `pass_flight` and `shot_flight` rows. A low-confidence Stage-2 result may abstain to `unresolved`.

Development uses V1–V5 with leave-one-video-out grouped cross-validation. Each fold trains on four videos and evaluates on the fifth. Training weighting gives comparable mass to human intervals and source videos so long clips or long intervals do not dominate the model.

After development choices are frozen, the final artefact is fit on all V1–V5 and evaluated once on V6–V7.

---

## Gate B — Shot/event correctness

Primary scoring is event-level, using human event windows and causal tolerance.

Required contract when GT is exhaustive for the evaluated event type:

- shot-attempt event recall >= 0.80;
- shot-attempt precision >= 0.80;
- shot outcome accuracy >= 0.75 on human-labeled matched attempts;
- no systematic rapid duplicate shot attempts from one physical attempt;
- rebound evaluation activates only when sufficient independent rebound GT exists.

Unknown outcome is preferable to a fabricated made/missed result and is scored separately from shot existence.

---

## Gate C — Team attribution

Required contract:

- team attribution coverage on attributable core events >= 0.80;
- team attribution accuracy >= 0.90 among attributed events;
- cross-team possession-change/pass/shot contamination <= 0.10;
- every evaluation clip must retain >= 0.70 attribution accuracy.

Unattributed is reported separately from wrongly attributed.

---

## Gate D — Game-player continuity

Required contract on reviewed same-player GT pairs/episodes:

- same-player continuity recall >= 0.85;
- false merges between different players <= 0.05;
- event-to-game-player attribution coverage >= 0.80 where the human actor is visually resolvable;
- per-clip continuity recall >= 0.70.

Track IDs may change. The evaluated identity is the stable game-player entity, not a raw tracker ID.

---

## Gate policy

- Gate thresholds are frozen before V6–V7 evaluation.
- Development and final validation videos are disjoint.
- V1–V5 may be used for AirborneClassifierV1 development, grouped CV and final fitting.
- V6–V7 are final validation-only and must remain untouched until model and thresholds are frozen.
- A failed gate triggers remediation of the responsible layer; thresholds are not changed in response to observed validation scores.
- P2 exits validation when Gates A–D pass their applicable contracts with reviewed GT.
