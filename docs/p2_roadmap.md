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

## Reconstruction status — CLOSED

The implementation/reconstruction phase of `p2-rebuilt` is closed. The branch now reconstructs both the basketball motor and the intended P2 product/runtime architecture on top of the current PR #43 / Identity-v2 baseline.

From this point until human ground truth is evaluated, the basketball motor is frozen: do not tune thresholds or state-machine behavior by intuition from a single video. The immediate next gate is the frozen Jersey-OFF Porto–Ovarense 0–60 s run, followed directly by human GT if the closure invariants remain clean.

Implemented basketball capabilities:

```text
robust shot re-arm / de-dup
explicit ball-control state
loose-ball start + recovery, including same-handler recovery
pass-flight evidence
pass / turnover transition attribution
shot-flight isolation from possession
camera/segment-boundary temporal reset
frame/timestamp discontinuity reset
rebound candidate chaining
real-video control/event metrics v3
per-event-type actor/team attribution
human-GT evaluator + annotation template
```

Implemented reconstruction/retirement work:

```text
single canonical P1 -> P2 runtime
legacy analytics Engine/metrics removed
Lab/worker detection-only runtime split removed
canonical analyzer/probe detection_only contract removed
legacy Streamlit UI removed
legacy renderer/session/P0-context surfaces removed
GameContext/P0 experiment surface removed
Smart Ball benchmark removed; Smart Ball OFF by default
legacy duplicate branding/runtime artifacts removed
frozen real-video regression fixture restored
repo-wide Python Ruff/compileall gate restored
Supabase migration replay/lint + deterministic empty seed restored
web typecheck/lint/build/Playwright quality gate enabled for p2-rebuilt
```

The supported frontend remains the current `web/` Next.js application. Current web branding and PR #43 Identity/Jersey capabilities are retained; only obsolete duplicate/Streamlit-era surfaces were retired.

## Runtime architecture

The old alternate analytics runtime is gone. `services/analytics/engine.py` and `services/analytics/metrics.py` are retired, and `services/pipeline.py` is the single canonical state path.

The analyzer always builds Identity-v2/P1 canonical state and then runs P2 when `BASKETBALL_P2_ENABLED=1`:

```text
video
  ↓
Identity-v2 detector / team evidence
  ↓
PlayerState[] + BallState + CourtState
  ↓
P2EngineV1 (when enabled)
  ↓
frame_state.p2_events + diagnostics["p2"]
```

`services.p2.P2Pipeline` is a compatibility wrapper that enables this canonical P2 stage; it must not run a second P2 engine.

The Lab contract, worker request, current Lab UI, canonical analyzer and real-video probe no longer expose or pass a `detection_only` mode. Canonical run metadata must describe the runtime as `canonical_p1_p2` and must not emit a stale `detection_only` key.

Smart Ball is OFF by default. Jersey remains optional evidence and does not become required operational identity before the deferred end-of-P2 Jersey gate.

## Retired legacy surfaces

The following old-P2 cleanup is part of `p2-rebuilt` and protected by CI invariants:

- legacy analytics Engine/metrics;
- legacy Streamlit app/UI and Streamlit-only dependencies;
- old renderer/session configuration surface;
- old P0 context pipeline and `GameContext`;
- old ball-context feature experiment surface;
- Smart Ball benchmark;
- obsolete local event-image/runtime artifacts;
- duplicate legacy root branding assets;
- obsolete Jersey ground-truth Streamlit tool.

These removals do not remove the current `web/` frontend, current Identity v2, current Jersey challenger stack, canonical P1 state, or P2 event engine.

## Infrastructure and quality gates

### Frozen fixture

`evaluation/fixtures/g4UROqJoYBM.mp4` is restored byte-for-byte from the legacy P2 branch. `evaluation/p2_frozen_fixture_test.py` locks its size and Git blob identity so the fixture cannot silently drift.

### Python quality

`.github/workflows/repo-python-lint.yml` runs Ruff correctness rules plus `compileall` and asserts that retired legacy surfaces are not reintroduced.

### Supabase

`.github/workflows/supabase-lint.yml` starts local Supabase, replays migrations from scratch, runs database lint, verifies local migration history and then tears the local stack down. `supabase/seed.sql` is intentionally empty so CI does not depend on production data.

### Web

`.github/workflows/web-ci.yml` runs on `p2-rebuilt` web changes and executes Node 24 install, TypeScript typecheck, ESLint, Next build, Playwright public-flow tests and accessibility/auth/onboarding smoke suites. Browser/E2E failures remain a separate web-quality gate and do not change the basketball-motor freeze unless they are caused by a P2 runtime contract regression.

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

The tracker distinguishes controlled possession from loose/pass-flight/shot-flight intervals. Same-team handler transitions become `PASS_CANDIDATE` only when direct ball-flight evidence supports the transition. Cross-team transitions may become `TURNOVER_CANDIDATE` with either flight evidence or a direct control change. Unknown team evidence cannot force pass/turnover classification.

Pass/turnover actor semantics are origin-first: the passer or player losing control is `actor_track_id`; the receiver/new handler is `secondary_track_id`. `POSSESSION_CHANGE` remains new-handler-first because it describes the resulting possession state.

Loose-ball intervals emit `LOOSE_BALL_STARTED` and `LOOSE_BALL_RECOVERED`. A `LOOSE -> POSSESSED` transition now closes the interval even when the same handler recovers the ball; that recovery must not create a fake possession change, pass or turnover.

## Shot and rebound

`ShotEventTrackerV1` uses a temporal state machine. Semantic detections are supporting evidence only; they do not become event truth by themselves. Missing outcome evidence resolves to `unknown`, not `missed`.

After a shot resolution, re-arm requires multiple directly observed frames that spatially clear the previous rim/start zone plus a minimum temporal separation. Predicted or missing ball frames do not count as proof of clearance, and re-arm/start cannot happen on the same frame.

When a supported shot becomes active, possession moves into `SHOT_FLIGHT` and is not reinterpreted as pass/turnover. The stable handler immediately before release — or a recent handler within the configured attribution window — may be retained as shot attribution evidence. After a supported missed shot, a subsequent supported possession/recovery within the rebound window may emit `REBOUND_CANDIDATE`.

## Temporal reset / camera boundary

`P2EngineV1` resets possession, loose/pass flight, shot/re-arm state and pending rebound state when:

- an analysis-ineligible segment persists beyond the configured gap;
- frame IDs become non-monotonic or jump beyond the accepted discontinuity;
- timestamps regress or jump beyond the accepted temporal discontinuity.

P2 intentionally consumes the canonical upstream eligibility boundary instead of adding a second pixel-level scene detector.

## Real-video probe v3 and ground truth

`evaluation/p2_real_video_probe.py` runs P2 through the canonical analyzer with Smart Ball explicitly OFF and Jersey optionally OFF. It emits:

```text
p2_metrics.json
p2_events.jsonl
p2_states.jsonl
```

Probe v3 deliberately separates two concepts that were previously conflated:

- actual possession coverage is `control_state == POSSESSED`;
- handler-memory coverage means `current_track_id` is retained while the control state may be `LOOSE`, `PASS_FLIGHT` or another non-possession state.

It also reports actor/team attribution independently for every event type so global attribution cannot hide weak shot attribution or another event-specific failure.

`evaluation/p2_ground_truth.py` evaluates predictions against manually authored `courtflux_p2_gt_v1` labels and reports per-event TP/FP/FN, precision, recall, F1 and temporal error, plus control-state frame accuracy/confusion.

`evaluation/p2_ground_truth_template.json` is the annotation starting point. Predictions must never be copied back into GT; uncertain identity/team fields remain null.

## Validation gates before Jersey ON

### P2.1 — Control / possession GT

Human-label stable possession, dribble gaps/occlusion, pass flight, loose balls, steals/deflections, shot flight and rebounds. Measure control-state accuracy/confusion, possession coverage, false-lock rate, time-to-lock, switch precision and unknown rate.

### P2.2 — Pass / turnover GT

Evaluate precision/recall and temporal tolerance against human labels. A pass requires supported flight evidence. Team-classification errors must not be counted as valid turnover inference.

### P2.3 — Shot GT

Label shot start, release/attempt window, rim interaction and outcome separately. Measure attempt precision/recall, duplicate-shot rate, made/missed precision and coverage, unknown rate and temporal error. Rapid UNKNOWN→new-shot retrigger should remain zero unless human GT confirms a genuine second attempt.

### P2.4 — Rebound GT

Evaluate rebound candidates only after shot-event quality is acceptable. Offensive/defensive classification requires independently reliable team evidence.

### P2.5 — Runtime / ablation

Ablate supporting signals independently: ball trajectory, rim geometry, semantic shot-action evidence, ball-in-basket evidence and team evidence. Promotion requires measurable quality gain versus runtime cost.

## Closure gate and next sequence

Implementation is frozen. The next sequence is validation, not further motor development:

1. Re-run exactly Porto–Ovarense 0–60 s with Jersey OFF on the canonical runtime.
2. Verify the summary has `mode = canonical_p1_p2` and no top-level `detection_only` key.
3. Verify `p2_probe.version = p2_real_video_probe_v3`, actual possession vs handler-memory metrics, per-event attribution, same-handler loose recoveries, zero rapid shot-retrigger regression and 100% P1 immutability.
4. If those closure invariants are clean, human-label this exact 0–60 s clip immediately and run the GT evaluator. Do not tune the motor by intuition before GT.
5. Run the committed `g4UROqJoYBM.mp4` frozen fixture and the remaining v1–v5 videos through the same Jersey-OFF battery when available.
6. Calibrate only against the frozen multi-video GT set.
7. Freeze the validated Jersey-OFF P2 baseline.
8. Run the deferred Jersey experiment (`soft-ranking + localized ROI + Identity-v2-only temporal consensus`) without adding a second consensus layer.
9. Promote P2 only after the basketball GT gates and end-of-P2 Jersey gate pass.
