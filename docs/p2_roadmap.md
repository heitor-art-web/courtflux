# P2 — Basketball Understanding Roadmap

P2 converts trusted P1 state into confidence-aware basketball event primitives.

```text
PlayerState[] + BallState + CourtState
                ↓
          P2 evidence fusion
                ↓
possession / pass / turnover / shot / rebound primitives
                ↓
          P3 aggregation later
```

## Baseline and phase boundary

P2 is built on the P1 baseline merged by PR #43. P1 perception/state is authoritative and P2 must not mutate it.

During P2, stable track/entity IDs are the operational identity source. `shirt_num` and `player_name` are optional, confidence-aware labels and may remain unknown. Jersey recognition is evidence/diagnostics only and its correctness gate is deferred to the end of P2.

Missing or contradictory evidence must remain unknown or a candidate rather than being forced into a basketball event.

## P2 v1 event contract

`core/basketball_event.py` defines timestamped primitives with event type, candidate/confirmed decision, frame/timestamp bounds, confidence/uncertainty, stable track IDs, optional team/court location, evidence, provenance and attributes.

Current event types:

```text
POSSESSION_ACQUIRED
POSSESSION_CHANGE
PASS_CANDIDATE
TURNOVER_CANDIDATE
SHOT_ATTEMPT
REBOUND_CANDIDATE
```

`SHOT_ATTEMPT` carries `outcome = made | missed | unknown`. P2 does not turn those primitives into P3 statistics.

## Possession v1

`PossessionTrackerV1` ranks ball/player evidence over canonical P1 state. Acquisition and switches use temporal hysteresis, predicted-ball positions are discounted, short evidence gaps have a grace window and weak evidence abstains.

A handler switch can always emit `POSSESSION_CHANGE`. It becomes `PASS_CANDIDATE` only with sufficiently confident same-team evidence, or `TURNOVER_CANDIDATE` only with sufficiently confident cross-team evidence. Unknown team evidence must not force either classification.

## Shot and rebound v1

`ShotEventTrackerV1` uses a temporal state machine. Semantic detections are supporting evidence only; they do not become event truth by themselves. Missing outcome evidence resolves to `unknown`, not `missed`.

When a supported shot becomes active, possession is released without inventing a pass/turnover. The stable handler track immediately before release may be retained as shot attribution evidence. After a supported missed shot, a subsequent supported possession within the rebound window may emit `REBOUND_CANDIDATE`.

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

## Validation gates

### P2.1 — Possession GT

Build human-labelled clips covering stable possession, dribble gaps/occlusion, handoffs, passes in flight, loose balls, steals/deflections and rebounds. Measure precision/coverage, false-lock rate, time-to-lock, switch precision and unknown rate.

### P2.2 — Pass / turnover GT

Evaluate event precision/recall and temporal tolerance against human labels. Team-classification errors must not be counted as valid turnover inference.

### P2.3 — Shot GT

Label shot start, release/attempt window, rim interaction and outcome separately. Measure attempt precision/recall, duplicate-shot rate, made/missed precision and coverage, unknown rate, temporal error and failure rate by camera view/cut/occlusion.

### P2.4 — Rebound GT

Evaluate rebound candidates only after shot-event quality is acceptable. Offensive/defensive classification requires independently reliable team evidence.

### P2.5 — Runtime / ablation

Ablate supporting signals independently: ball trajectory, rim geometry, semantic shot-action evidence, ball-in-basket evidence and team evidence. Promotion requires measurable quality gain versus runtime cost.

## Next implementation steps

1. Persist a dedicated P2 event stream instead of relying only on per-frame diagnostics.
2. Add human-GT schemas/evaluators for possession and shots.
3. Calibrate thresholds on frozen videos rather than hand-tuning to one game.
4. Add explicit pass-flight evidence: handler release → airborne ball → receiver lock.
5. Add an explicit loose-ball state for contested/free-ball intervals.
6. Add camera-cut/reset handling for all temporal P2 trackers.
7. Keep actor attribution keyed by stable track/entity IDs; enrich with jersey/player labels only when their own evidence is reliable.
8. Run the deferred Jersey experiment/gate at the end of P2 without adding a second temporal consensus layer inside the recognizer.
9. Promote P2 into the default runtime only after the GT gates pass.
