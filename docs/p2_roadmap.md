# P2 — Basketball Understanding Roadmap

P2 converts trusted P1 state into confidence-aware basketball event primitives.

```text
PlayerState[] + BallState + CourtState
                ↓
          P2 evidence fusion
                ↓
control state + possession / pass / turnover / shot / rebound primitives
                ↓
          P3 aggregation later
```

## Baseline and phase boundary

P2 is built on the P1 baseline merged by PR #43. P1 perception/state is authoritative and P2 must not mutate it.

During P2, stable track/entity IDs are the operational identity source. `shirt_num` and `player_name` are optional, confidence-aware labels and may remain unknown. Jersey recognition is evidence/diagnostics only and its correctness gate is deferred to the end of P2.

Missing or contradictory evidence must remain unknown or a candidate rather than being forced into a basketball event.

## Rebuilt motor status

The `p2-rebuilt` motor implementation is now complete for the intended P2 scope. The remaining work before the deferred Jersey gate is **validation/calibration**, not adding another identity dependency.

Implemented motor capabilities:

```text
robust shot re-arm / de-dup
explicit ball-control state
loose-ball start + recovery
pass-flight evidence
pass / turnover transition attribution
shot-flight isolation from possession
camera/segment-boundary temporal reset
frame/timestamp discontinuity reset
rebound candidate chaining
real-video control/event metrics
human-GT evaluator + annotation template
```

Camera-cut handling is intentionally expressed as a temporal reset at upstream analysis/segment boundaries and frame/timestamp discontinuities. P2 does not add a second pixel-level scene detector; it consumes the canonical P1 eligibility boundary and resets all temporal basketball state when that boundary persists.

## P2 v1 event contract

`core/basketball_event.py` defines timestamped primitives with event type, candidate/confirmed decision, frame/timestamp bounds, confidence/uncertainty, stable track IDs, optional team/court location, evidence, provenance and attributes.

Current event types:

```text
POSSESSION_ACQUIRED
POSSESSION_CHANGE
PASS_CANDIDATE
TURNOVER_CANDIDATE
LOOSE_BALL_STARTED
LOOSE_BALL_RECOVERED
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

## Possession, loose ball and pass-flight

`PossessionTrackerV1` ranks ball/player evidence over canonical P1 state. Acquisition and switches use temporal hysteresis, predicted-ball positions are discounted, short evidence gaps have a grace window and weak evidence abstains.

The tracker now distinguishes controlled possession from loose/pass-flight/shot-flight intervals. Same-team handler transitions become `PASS_CANDIDATE` only when direct ball-flight evidence supports the transition. Cross-team transitions may become `TURNOVER_CANDIDATE` with either flight evidence or a direct control change. Unknown team evidence still cannot force pass/turnover classification.

Pass/turnover actor semantics are origin-first: the passer or player losing control is `actor_track_id`; the receiver/new handler is `secondary_track_id`. `POSSESSION_CHANGE` remains new-handler-first because it describes the resulting possession state.

Loose-ball intervals emit `LOOSE_BALL_STARTED` and, after a supported new lock, `LOOSE_BALL_RECOVERED`. A recovery can then support a pass/turnover interpretation without inventing continuity through the free-ball interval.

## Shot and rebound

`ShotEventTrackerV1` uses a temporal state machine. Semantic detections are supporting evidence only; they do not become event truth by themselves. Missing outcome evidence resolves to `unknown`, not `missed`.

Shot de-dup/re-arm no longer accepts a one-frame disappearance of the shot-start signal. After a resolution, re-arm requires multiple directly observed frames that spatially clear the previous rim/start zone plus a minimum temporal separation. Predicted or missing ball frames do not count as proof of clearance, and re-arm/start cannot happen on the same frame.

When a supported shot becomes active, possession moves into `SHOT_FLIGHT` and is not reinterpreted as pass/turnover. The stable handler immediately before release — or a recent handler within the configured attribution window — can be retained as shot attribution evidence. After a supported missed shot, a subsequent supported possession/recovery within the rebound window may emit `REBOUND_CANDIDATE`.

## Temporal reset / camera boundary

`P2EngineV1` resets possession, loose/pass flight, shot/re-arm state and pending rebound state when:

- an analysis-ineligible segment persists beyond the configured gap;
- frame IDs become non-monotonic or jump beyond the accepted discontinuity;
- timestamps regress or jump beyond the accepted temporal discontinuity.

This prevents candidates from crossing replay/camera/segment boundaries. The reset reason and count are included in P2 diagnostics and the real-video probe.

## Runtime integration

`p2-rebuilt` keeps the current `main` pipeline intact and provides `services.p2.P2Pipeline` as an opt-in adapter. It executes P2 immediately after canonical P1 state construction and before the existing analytics engine when that engine is active.

Enable with:

```bash
BASKETBALL_P2_ENABLED=1
```

P2 output is exposed through:

```text
frame_state.p2_events
frame_state.diagnostics["p2"]
```

The legacy analytics engine is compatibility code and must not be used as P2 ground truth or promotion labels.

## Real-video probe

`evaluation/p2_real_video_probe.py` runs P2 through the canonical analyzer without changing the normal Lab runtime. Smart Ball is explicitly OFF for the rebuilt baseline and Jersey may remain OFF until the end-of-P2 identity gate.

Probe output now includes:

```text
p2_metrics.json
p2_events.jsonl
p2_states.jsonl
```

Metrics include possession coverage/unknown rate, control-state distribution, event counts/confidence/attribution, pass-flight support, direct-vs-flight turnovers, loose-ball intervals, shot outcomes, rapid UNKNOWN→shot retriggers, temporal resets and the P1 immutability invariant.

## Human ground truth

`evaluation/p2_ground_truth.py` evaluates P2 predictions against manually authored `courtflux_p2_gt_v1` labels. It reports per-event TP/FP/FN, precision, recall, F1 and temporal error, plus control-state frame accuracy/confusion.

`evaluation/p2_ground_truth_template.json` is the annotation starting point. Labels must be human-authored; predictions must not be copied back into GT. Unknown/uncertain identity or team fields should remain null rather than guessed.

Example evaluation:

```bash
python evaluation/p2_ground_truth.py \
  --events evaluation/runs/<run_id>/p2_events.jsonl \
  --states evaluation/runs/<run_id>/p2_states.jsonl \
  --ground-truth evaluation/porto_ovarense_0_60_gt.json \
  --tolerance-frames 8
```

## Validation gates before Jersey ON

### P2.1 — Control / possession GT

Human-label stable possession, dribble gaps/occlusion, pass flight, loose balls, steals/deflections, shot flight and rebounds. Measure control-state accuracy/confusion, possession coverage, false-lock rate, time-to-lock, switch precision and unknown rate.

### P2.2 — Pass / turnover GT

Evaluate precision/recall and temporal tolerance against human labels. A pass requires supported flight evidence. Team-classification errors must not be counted as valid turnover inference.

### P2.3 — Shot GT

Label shot start, release/attempt window, rim interaction and outcome separately. Measure attempt precision/recall, duplicate-shot rate, made/missed precision and coverage, unknown rate, temporal error and failure rate by camera view/cut/occlusion. Rapid UNKNOWN→new-shot retrigger should remain zero on frozen validation clips unless human GT confirms a genuine second attempt.

### P2.4 — Rebound GT

Evaluate rebound candidates only after shot-event quality is acceptable. Offensive/defensive classification requires independently reliable team evidence.

### P2.5 — Runtime / ablation

Ablate supporting signals independently: ball trajectory, rim geometry, semantic shot-action evidence, ball-in-basket evidence and team evidence. Promotion requires measurable quality gain versus runtime cost.

## Closure sequence

1. Re-run the frozen Porto–Ovarense 0–60 s clip with Jersey OFF and the v2 probe.
2. Verify control-state distribution, zero rapid shot retrigger regressions, reset behavior and pass-flight/loose-ball event plausibility.
3. Human-label that frozen clip using `courtflux_p2_gt_v1` and run the evaluator.
4. Repeat the same battery on the remaining frozen videos/fixtures (including v1–v5 once their local paths are available).
5. Calibrate thresholds only against the frozen multi-video set; do not hand-tune to one game.
6. Freeze the Jersey-OFF P2 baseline once the GT gates are acceptable.
7. Run the deferred Jersey experiment/gate (`soft-ranking + localized ROI + Identity-v2-only temporal consensus`) without adding a second consensus layer.
8. Promote P2 into the default runtime only after the basketball GT gates and end-of-P2 Jersey gate pass.
