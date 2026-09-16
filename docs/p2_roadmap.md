# P2 — Basketball Understanding Roadmap

P2 converts trusted P1 state into confidence-aware basketball event primitives.

```text
PlayerState[] + BallState + CourtState
                ↓
          P2 evidence fusion
                ↓
control state + possession / pass / turnover / loose / shot / rebound primitives
                ↓
          P3 aggregation later
```

## Current status

**P2 implementation/reconstruction is closed; P2 validation/calibration is active.**

The reconstructed P2 runtime was merged into `main` through PR #48. The `p2-rebuilt` branch remains active for continued P2 validation and contract hardening.

P2 is not yet promoted as complete. The current gate is basketball correctness on frozen real-video evaluation with Jersey OFF, followed by human GT and controlled calibration.

The basketball motor is frozen against intuition-driven tuning: possession/shot state-machine thresholds should not be changed because of one visually surprising clip. Contract fixes, observability fixes and boundary corrections are allowed when they repair a demonstrated architectural problem; motor calibration must be driven by the frozen evaluation set and GT.

## Baseline and phase boundary

P2 is built on the P1 baseline merged by PR #43. P1 perception/state remains authoritative and P2 must not mutate it.

During P2:

- stable track/entity IDs are the operational identity source;
- `shirt_num` and `player_name` are optional, confidence-aware labels and may remain `UNKNOWN`;
- Jersey recognition is evidence/diagnostics only and final correctness is deferred to the end of P2;
- Smart Ball remains OFF by default;
- the current basketball-validation battery runs with Jersey OFF.

Missing or contradictory evidence must remain unknown or a candidate rather than being forced into a basketball event.

## Canonical runtime architecture

The old alternate analytics runtime and detection-only runtime split are retired. `services/pipeline.py` is the canonical state path.

```text
video
  ↓
GameplayFilter
  └── high-recall prefilter evidence; fail-open by default
  ↓
Identity-v2 detector / tracker / team evidence
  ↓
CourtState
  ├── gameplay_present       → continuity / P2 scope
  └── analysis_eligible      → strict metric-analysis scope
  ↓
PlayerState[] + BallState + CourtState
  ↓
P2EngineV1
  ↓
frame_state.p2_events + diagnostics["p2"]
```

`services.p2.P2Pipeline` is a compatibility wrapper that enables the canonical P2 stage; it must not run a second event engine.

The Lab contract, worker request, canonical analyzer and real-video probe no longer expose or pass the retired `detection_only` mode. Canonical run metadata must describe the runtime as `canonical_p1_p2` and must not emit a stale top-level `detection_only` field.

## Gameplay presence vs analysis quality

The runtime now separates gameplay continuity from strict analysis quality.

The historical image-only `GameplayFilter` decision remains available as diagnostics (`likely_gameplay`), but by default it does not veto valid frames before detector/tracker/Identity inference. This prevents weak HSV/court-colour evidence, painted areas, tight broadcast views or camera composition changes from creating artificial temporal gaps.

Post-detector `CourtState` exposes three practical scopes:

```text
NON_GAMEPLAY
GAMEPLAY_LIMITED
GAMEPLAY_ANALYSIS
```

Semantics:

```text
GAMEPLAY_LIMITED
tracking / Identity / BallState / P2 continuity   ✅
strict metric analysis                            ❌ / abstain

GAMEPLAY_ANALYSIS
tracking / Identity / BallState / P2 continuity   ✅
strict metric analysis                            ✅
```

`CourtState.gameplay_present` controls continuity eligibility. `CourtState.analysis_eligible` remains conservative and controls strict metric-scoped analysis.

This distinction is part of the P1/P2 boundary contract. P2 should not reset simply because a real basketball frame is too tight or visually weak for wide-view analysis.

## P2 event contract

`core/basketball_event.py` defines timestamped primitives with event type, candidate/confirmed decision, frame/timestamp bounds, confidence/uncertainty, stable track IDs, optional team/court location, evidence, provenance and attributes.

Current event types:

```text
POSSESSION_ACQUIRED
POSSESSION_CHANGE
PASS_CANDIDATE
TURNOVER_CANDIDATE
LOOSE_BALL_STARTED
LOOSE_BALL_RECOVERED
LOOSE_BALL_ENDED
SHOT_ATTEMPT
REBOUND_CANDIDATE
```

`BallControlState` provides the temporal interpretation:

```text
UNKNOWN
POSSESSED
LOOSE
PASS_FLIGHT
SHOT_FLIGHT
```

`SHOT_ATTEMPT` carries `outcome = made | missed | unknown`. P2 does not turn these primitives into P3 statistics.

## Implemented basketball capabilities

```text
explicit ball-control state
possession acquisition/change
robust loose-ball lifecycle
same-handler loose recovery
explicit loose-ball terminal closure
pass-flight evidence
pass / turnover transition attribution
shot-flight isolation from possession
robust shot re-arm / de-dup
made / missed / unknown shot outcome evidence
rebound candidate chaining
frame/timestamp discontinuity reset
per-event-type actor/team attribution
human-GT evaluator + annotation template
```

### Possession / pass / turnover

`PossessionTrackerV1` ranks canonical player/ball evidence with temporal hysteresis. Predicted ball evidence is discounted, short evidence gaps have grace behavior and weak evidence may abstain.

Same-team handler transitions become `PASS_CANDIDATE` only when direct ball-flight evidence supports the transition. Cross-team transitions may become `TURNOVER_CANDIDATE` with either flight evidence or a direct control change. Unknown team evidence cannot force pass/turnover classification.

Pass/turnover actor semantics are origin-first: the passer or player losing control is `actor_track_id`; the receiver/new handler is `secondary_track_id`. `POSSESSION_CHANGE` remains new-handler-first because it describes the resulting possession state.

### Loose ball

Loose intervals emit `LOOSE_BALL_STARTED` and close with either `LOOSE_BALL_RECOVERED` or `LOOSE_BALL_ENDED`.

A `LOOSE -> POSSESSED` transition closes the interval even when the same handler recovers the ball. Same-handler recovery must not create a fake possession change, pass or turnover.

Terminal closure is explicit when a loose interval ends because of another state transition or temporal reset.

### Shot and rebound

`ShotEventTrackerV1` uses a temporal state machine. Semantic detections are supporting evidence only and cannot create event truth by themselves.

Missing outcome evidence resolves to `unknown`, not `missed`.

After shot resolution, re-arm requires multiple directly observed frames that spatially clear the previous rim/start zone plus a minimum temporal separation. Predicted or missing ball frames do not count as direct clearance evidence.

When a supported shot becomes active, possession moves into `SHOT_FLIGHT` and airborne ball evidence is not reinterpreted as pass/turnover. The stable handler immediately before release — or a recent handler inside the attribution window — may be retained as shot actor evidence.

After a supported missed shot, a subsequent supported possession/recovery inside the rebound window may emit `REBOUND_CANDIDATE`.

## Temporal reset / segment boundaries

P2 resets possession, loose/pass flight, shot/re-arm state and pending rebound state when:

- a persistent **non-gameplay** gap is established by the canonical post-detector gameplay-presence contract;
- frame IDs become non-monotonic or jump beyond the accepted discontinuity;
- timestamps regress or jump beyond the accepted temporal discontinuity.

A frame that is `GAMEPLAY_LIMITED` but still has `gameplay_present=True` remains eligible for P2 continuity even when `analysis_eligible=False`.

P2 intentionally consumes the canonical upstream gameplay boundary instead of implementing a second pixel-level scene detector.

## Metrics and diagnostics

`P2MetricsCollector` currently emits the v5 scope contract.

The important denominator split is:

```text
metric_eligible_frames
    strict CourtState.analysis_eligible scope

engine_eligible_frames
    P2 continuity / gameplay_present scope

limited_gameplay_frames
    gameplay retained for P2 but excluded from strict metric analysis
```

Possession/control rates use the P2 engine scope. Strict P1 metric counts remain tied to the conservative analysis scope.

The collector also reports:

- control-state counts/rates;
- actual `POSSESSED` coverage separately from retained handler memory;
- event counts and decision types;
- per-event actor/team attribution;
- pass-flight and turnover evidence splits;
- shot outcomes and rapid-retrigger diagnostics;
- loose-ball closure and terminal reasons;
- temporal reset reasons;
- P1 immutability rate.

`p2_states.jsonl` records both analysis and gameplay-continuity scope so a future failure can be attributed to the correct gate rather than collapsed into one `valid_gameplay` boolean.

## Real-video probe and ground truth

`evaluation/p2_real_video_probe.py` runs P2 through the canonical analyzer with Smart Ball OFF and Jersey optionally disabled. Current P2 validation uses `--no-jersey`.

The probe emits:

```text
p2_metrics.json
p2_events.jsonl
p2_states.jsonl
```

`evaluation/p2_ground_truth.py` evaluates predictions against manually authored `courtflux_p2_gt_v1` labels and reports per-event TP/FP/FN, precision, recall, F1 and temporal error, plus control-state frame accuracy/confusion.

`evaluation/p2_ground_truth_template.json` is the annotation starting point. Predictions must never be copied back into GT; uncertain identity/team fields remain null.

## Validation gates before Jersey ON

### P2.1 — Gameplay/continuity boundary

Verify that true basketball frames survive tight views and weak court-colour evidence without artificially breaking tracker/Identity/P2 continuity. Measure:

- inferred-frame coverage;
- `GAMEPLAY_ANALYSIS` vs `GAMEPLAY_LIMITED` vs `NON_GAMEPLAY` counts;
- P2 engine eligibility;
- reset reasons;
- false continuation into genuine replays/graphics/non-gameplay.

### P2.2 — Control / possession GT

Human-label stable possession, dribble gaps/occlusion, pass flight, loose balls, steals/deflections, shot flight and rebounds. Measure control-state accuracy/confusion, possession coverage, false-lock rate, time-to-lock, switch precision and unknown rate.

### P2.3 — Pass / turnover GT

Evaluate precision/recall and temporal tolerance against human labels. A pass requires supported flight evidence. Team-classification errors must not be counted as valid turnover inference.

### P2.4 — Shot GT

Label shot start, release/attempt window, rim interaction and outcome separately. Measure attempt precision/recall, duplicate-shot rate, made/missed precision and coverage, unknown rate and temporal error. Rapid UNKNOWN→new-shot retrigger should remain zero unless human GT confirms a genuine second attempt.

### P2.5 — Rebound GT

Evaluate rebound candidates only after shot-event quality is acceptable. Offensive/defensive classification requires independently reliable team evidence.

### P2.6 — Runtime / ablation

Ablate supporting signals independently: ball trajectory, rim geometry, semantic shot-action evidence, ball-in-basket evidence and team evidence. Promotion requires measurable quality gain versus runtime cost.

## Current validation sequence

1. Run the frozen V1–V5 battery with the current gameplay-presence split and Jersey OFF.
2. Re-run the exact Porto–Ovarense `0–60 s` canonical probe with Jersey OFF.
3. Verify canonical metadata has no retired top-level `detection_only` field.
4. Compare `metric_eligible_frames`, `engine_eligible_frames` and `limited_gameplay_frames`; do not collapse these scopes.
5. Inspect continuity/reset reasons, possession/control states, loose-ball closure, pass/turnover evidence, shot de-duplication and per-event attribution.
6. If contract invariants remain clean, human-label the frozen clips and run the GT evaluator.
7. Do not tune the basketball motor by intuition before GT.
8. Calibrate only against the frozen multi-video GT set.
9. Freeze the validated Jersey-OFF P2 baseline.
10. Run the deferred Jersey experiment: **soft-ranking + localized ROI + Identity-v2-only temporal consensus**.
11. Promote P2 only after the basketball GT gates and end-of-P2 Jersey gate are acceptable.

## Deferred Jersey gate

The P1 decision merged in PR #43 remains authoritative: stable track/entity IDs own P2 attribution and Jersey correctness is deferred.

The end-of-P2 Jersey experiment must preserve:

```text
soft frame ranking
    ↓
localized torso / number ROI
    ↓
specialized recognizer
    ↓
one raw visual observation or UNKNOWN
    ↓
roster validator-only
    ↓
Identity v2 as the single temporal consensus / locking layer
```

Do not add a second temporal consensus layer inside the recognizer adapter. `UNKNOWN` remains preferable to a false player identity.

## Retired legacy surfaces

The following remain intentionally retired:

- legacy analytics Engine/metrics runtime;
- legacy Streamlit app/UI and Streamlit-only dependencies;
- old renderer/session configuration surface;
- old P0 context pipeline and `GameContext`;
- old ball-context feature experiment surface;
- Smart Ball benchmark;
- obsolete local event-image/runtime artifacts;
- duplicate legacy root branding assets;
- obsolete Jersey ground-truth Streamlit tool.

The supported frontend remains the current `web/` Next.js application. Identity v2, the current Jersey challenger stack, canonical P1 state and the P2 event engine remain active components.
