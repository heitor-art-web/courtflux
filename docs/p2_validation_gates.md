# P2 validation gates — P3 readiness contract

## Purpose

P2 exists to provide a basketball timeline that P3 can aggregate, search, clip and explain. Validation therefore measures whether P2 understands the semantic sequence of the play, not whether every human interval boundary is reproduced frame-perfectly.

The data contract is now:

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

Human `control_intervals`, matched to predicted control-state intervals. Human boundaries are windows, not millisecond release truth. A prediction of the same semantic state is a detection when its interval overlaps the human window or is within the frozen causal tolerance.

Frame accuracy, dominant-frame accuracy and exact boundary overlap are diagnostics only.

### Required checks

Frozen before V6–V7 validation:

- GT-window prediction coverage: aggregate >= 0.98; every clip >= 0.95.
- Core interval detection recall (`possessed`, `pass_flight`, `shot_flight`): aggregate >= 0.70.
- Possessed interval detection recall: aggregate >= 0.70.
- Pass interval detection recall: aggregate >= 0.60.
- Shot interval detection recall: aggregate >= 0.70.
- Semantic transition recall: aggregate >= 0.65.
- Core unresolved interval rate: aggregate <= 0.25.
- PASS truth resolved as SHOT: <= 0.15.
- SHOT truth resolved as PASS: <= 0.15.
- Per-clip core interval detection recall >= 0.40.
- Per-clip core unresolved interval rate <= 0.45.

`loose` remains reported but is not a required Gate-A check until at least two independent human loose-ball intervals exist.

### Interval resolution contract

For each GT core interval:

1. match one predicted interval of the same state when overlap/distance is within tolerance;
2. otherwise identify the nearest competing core-state prediction within tolerance;
3. if a competing state exists, mark the GT interval as `wrong_type`;
4. otherwise mark it `unresolved`.

Same-state matching is one-to-one so one long prediction cannot satisfy several independent GT episodes.

### Sequence contract

Adjacent GT core intervals form semantic transitions. A transition is correct when both intervals have correct same-state matches and the matched prediction intervals preserve temporal order. Exact start/end agreement is not required.

### Diagnostics, not blockers

- frame accuracy;
- dominant predicted state inside a GT interval;
- mean frame matching rate;
- start/end boundary errors;
- interval IoU/overlap ratio.

These remain useful for later quality improvement but do not by themselves prevent P3 from using a correctly detected episode.

---

## Gate B — Shot/event correctness

### Product question

> Can P3 create shot/rebound review collections without inventing or losing a material fraction of shot events?

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

### Product question

> When team evidence is known, can P3 aggregate events by team without systematic cross-team contamination?

Run only on known-team GT.

Required contract:

- team attribution coverage on attributable core events >= 0.80;
- team attribution accuracy >= 0.90 among attributed events;
- cross-team possession-change/pass/shot contamination <= 0.10;
- every evaluation clip must retain >= 0.70 attribution accuracy.

Unattributed is reported separately from wrongly attributed.

---

## Gate D — Game-player continuity

### Product question

> Can P3 aggregate a game's events under stable game-player entities without requiring real names, jersey OCR or cross-game identity?

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
