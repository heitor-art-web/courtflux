# P2 validation gates — P3 readiness contract

## Purpose

P2 exists to provide a basketball timeline that P3 can aggregate, search, clip and explain. Validation therefore measures whether P2 understands the semantic sequence of the play, not whether every human interval boundary is reproduced frame-perfectly.

The data contract is:

- V1, V3, V4 and V5: normal development corpus for Gates A–D and AirborneClassifierV1 development;
- V2: development challenge/outlier clip, always reported as a stress/regression test but not a blocker for closing development Gates A–D;
- grouped leave-one-video-out cross-validation for normal development decisions uses V1, V3, V4 and V5;
- the final AirborneClassifierV1 artefact is fitted only after development choices are frozen;
- V6–V7: new, unseen videos reserved exclusively for the final P2 Exit Gate.

V6–V7 must remain completely untouched while Gates A–D are being developed and closed. They must not be used for feature selection, model-family selection, weights, confidence thresholds, gate thresholds, iterative diagnostics or tuning.

The development validation order remains A → B → C → D.

After Gates A–D are closed and the P2 model, features, weights, confidence thresholds and gate thresholds are frozen, V6–V7 are evaluated once as the final P2 Exit Gate.

---

## Gate A — Timeline readiness

### Product question

> Can P3 trust the P2 control timeline to find possessions, passes and shots in the right semantic order without systematic collapse on a normal clip?

### Primary scoring unit

Human `control_intervals`, matched one-to-one to predicted control-state intervals. Human boundaries are windows, not millisecond release truth. A prediction of the same semantic state is a detection when its interval overlaps the human window or is within the frozen causal tolerance.

Frame accuracy, dominant-frame accuracy and exact boundary overlap are diagnostics only.

### Required checks

Frozen before final Exit Gate validation:

- GT-window prediction coverage: aggregate >= 0.98; every clip >= 0.95.
- Core interval detection recall (`possessed`, `pass_flight`, `shot_flight`): aggregate >= 0.70.
- Possessed interval detection recall: aggregate >= 0.70.
- Pass interval detection recall: aggregate >= 0.60.
- Shot interval detection recall >= 0.70.
- Semantic transition **order accuracy**, conditioned on both adjacent GT episodes being correctly detected: aggregate >= 0.90.
- Core unresolved interval rate: aggregate <= 0.25.
- PASS truth resolved as SHOT: <= 0.15.
- SHOT truth resolved as PASS: <= 0.15.
- Per-clip core interval detection recall >= 0.40.
- Per-clip core unresolved interval rate <= 0.45.

`loose` remains reported but is not a required Gate-A check until sufficient independent human loose-ball intervals exist.

During development, required Gate-A checks are evaluated on the normal development set V1/V3/V4/V5. V2 is reported separately as a challenge/outlier regression clip and does not block development Gate A.

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

Normal development uses V1, V3, V4 and V5 with grouped leave-one-video-out cross-validation. Each fold trains on the remaining normal development videos and evaluates on the held-out video. Training weighting gives comparable mass to human intervals and source videos so long clips or long intervals do not dominate the model.

V2 remains available throughout development as a challenge/outlier stress and regression clip, but its score is reported separately and does not block development Gates A–D.

After development choices are frozen, the final artefact is fit on the frozen development corpus. V6–V7 remain untouched until all Gates A–D are closed and the final P2 Exit Gate is run.

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

### Team identity and kit-context contract

Gate C evaluates **team identity**, not colour labels. Human scoring uses stable `team_a` / `team_b` slots inside the declared runtime context; colour names are evidence used to configure the visual team classifier and must never become the semantic team identity by themselves.

Each V1–V5 GT file may contain a prediction-blind `human_kit_audit` with:

- source-time segment boundaries;
- team slot (`team_a` / `team_b`);
- jersey-body colours;
- shorts colours;
- lettering/number colours;
- real team name only when the reviewer explicitly supplied it.

For the current TeamOnlyAnalyzer contract, **jersey-body colours are the configurable classifier evidence**. Shorts and lettering are retained as human context and must not be silently injected as torso colours.

When a source clip changes matchup or equipment context, Gate C must not force one static colour configuration over the whole clip. Colour-stable runtime segments are evaluated separately and aggregated back to the original video for the per-clip Gate-C check. V4 therefore uses three runtime segments (0–15 s, 15–27 s, 27–30 s) while remaining one `V4` result for thresholding.

V3 is a player-highlight clip whose Team-A torso varies Black/Red → Red → Black while Team B remains White. Its development runtime config uses the prediction-blind union of the reviewed Team-A torso colours, while the exact temporal variants remain recorded in `v3.json`.

Human kit context and runtime colour support are separate concerns. A colour written in GT (for example Orange) records what the reviewer observed; it does **not** imply that the current classifier implementation already models that colour correctly. Colour-capability validation occurs before Gate C is closed and before the final P2 freeze.

---

## Gate D — Game-player continuity

Required contract on reviewed same-player GT pairs/episodes:

- same-player continuity recall >= 0.85;
- false merges between different players <= 0.05;
- event-to-game-player attribution coverage >= 0.80 where the human actor is visually resolvable;
- per-clip continuity recall >= 0.70.

Track IDs may change. The evaluated identity is the stable game-player entity, not a raw tracker ID.

### Gate D identity boundary

Gate D is deliberately **anonymous and game-local**. It answers whether two observations belong to the same player inside the game; it does not answer who that real player is.

The following are prohibited as Gate-D resolver/association inputs and may not be reintroduced as an indirect "fingerprint":

- jersey number / `shirt_num`;
- Jersey OCR output or digit posteriors;
- roster or `player_name` context;
- embeddings/features from a model trained specifically to recognize jersey numbers;
- human jersey labels as runtime identity evidence.

Human jersey labels may be used only to construct and score reviewed SAME/DIFFERENT ground truth after the anonymous association has already been produced. Jersey/roster/real-world identity belongs to **P5 — Persistent Player Identity / Jersey Evidence**.

If Gate D cannot meet its thresholds without P5 evidence, Gate D remains open. P5 evidence must not be moved backward into P2 merely to make Gate D pass.

The detailed phase boundary is defined in `docs/p1_p2_identity_boundary.md`.

---

## Final P2 Exit Gate

The final P2 Exit Gate is evaluated only after development Gates A–D are closed and all P2 development decisions are frozen.

- V6–V7 are the only final unseen Exit Gate videos.
- V6–V7 are evaluated once with the frozen P2 stack and frozen Gate A–D contracts.
- V6–V7 must not be used for iterative tuning or threshold selection.
- If the Exit Gate exposes a defect that requires model, feature, policy or threshold changes, the affected layer is remediated using development data; a subsequent independent final validation requires fresh unseen videos rather than reusing V6–V7 as if they were still unseen.

---

## Gate policy

- Gate thresholds are frozen before the final V6–V7 Exit Gate.
- Normal development and final validation videos are disjoint.
- V1/V3/V4/V5 are the normal development set for Gates A–D.
- V2 is a development challenge/outlier set: always reported, never hidden, but non-blocking for development Gate closure.
- V6–V7 are final Exit Gate-only and must remain untouched until Gates A–D, model, features, weights and thresholds are frozen.
- A failed development gate triggers remediation of the responsible layer; thresholds are not changed in response to observed scores.
- P2 exits validation only after Gates A–D pass their applicable development contracts and the frozen stack passes the final V6–V7 Exit Gate with reviewed GT.
