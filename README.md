# CourtFlux — Basketball Intelligence from Video

**CourtFlux** is a computer-vision system for turning standard basketball broadcast footage into structured player, ball, event and tactical intelligence.

The long-term objective is to reconstruct a basketball game from ordinary video well enough to answer, with measurable confidence:

```text
Where are the players?
Who probably has the ball?
What happened?
Who did it?
Where did it happen?
What does it mean tactically?
```

CourtFlux is in active development. The P0 vision/evaluation foundation is complete and P1 is now in final exit validation: the canonical runtime, CourtState, BallState, PlayerState, team evidence, visual QA and performance work are integrated on `main`; human-GT Jersey validation remains the principal open P1 gate.

---

## Current phase

```text
P0 — VISION FOUNDATION                         ✅ COMPLETE
P1 — PERCEPTION + GAME STATE                  ← CURRENT PHASE / EXIT VALIDATION
P2 — BASKETBALL UNDERSTANDING
P3 — STRUCTURED ANALYTICS + ACTIONS
P4 — TACTICAL INTELLIGENCE
```

P1 must tolerate missing/`UNKNOWN` state and must never force player, jersey or ball identity when evidence is weak.

Current P1 execution status:

```text
1. Player tracking baseline + evaluation                         ✅ complete
2. RF-DETR player-detector challenger                            ✅ closed / not promoted
3. Detector model_v1.1.0 supervised validation                  ✅ complete
4. Temporal Ball Tracker + trajectory consistency                ✅ complete / integrated
5. BallState                                                     ✅ complete / integrated
6. PlayerState                                                   ✅ complete / integrated
7. CourtState                                                    ✅ complete / integrated
8. Single-forward runtime + P1 correctness cleanup               ✅ merged / validated
9. Smart Ball ablation                                           ✅ complete → OFF by default
10. Team identity correctness                                    ✅ closed for P1
11. Integrated visual QA                                         ✅ passed
12. Jersey identification correctness                            🔄 human GT pending
13. P1 exit / v1.2.0 release gate                                ⏳ pending Jersey GT + final release validation

SAM2 → deferred / contingency backlog
```

See:

- `docs/p1_roadmap.md` — canonical P1 execution roadmap and promotion rules;
- `docs/p0_foundation_snapshot.md` — frozen P0 exit snapshot and P1 boundary;
- `docs/system_quality_gate.md` — detailed ball-quality evaluation contract.

---

## Product hierarchy

### P0 — Vision Foundation — COMPLETE

P0 establishes the measurable perception foundation:

- frozen evaluation contract;
- canonical detector baseline;
- ball candidate / hypothesis foundation;
- court geometry / homography;
- explicit uncertainty and diagnostics;
- repeatable evaluation tooling.

### P1 — Perception + Game State — CURRENT

P1 answers:

> Who/what is visible, where is it, and what identity/state evidence do we trust now?

P1 owns:

- player detection/tracking quality;
- jersey identification as track-level identity evidence;
- team evidence;
- temporal ball tracking;
- ball trajectory consistency;
- versioned detector validation;
- `BallState`;
- `PlayerState` + `CourtState`;
- persistent team + jersey evidence;
- confidence, provenance and `UNKNOWN` semantics.

P1 does **not** need to infer possession, passes, shots, rebounds or tactical meaning before it can be considered complete.

### P2 — Basketball Understanding

P2 answers:

> What basketball event is happening?

P2 consumes `PlayerState + BallState + CourtState` and builds confidence-aware basketball event primitives:

- possession states and transitions;
- pass candidates;
- turnover candidates;
- shot attempts and outcome evidence;
- rebound candidates;
- basketball-specific semantic evidence supporting those decisions;
- timestamped event primitives with provenance and uncertainty.

Semantic classes such as `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `ball-in-basket` and `rim` are supporting evidence only and must never force an event by themselves.

P2 decides the event. It does not yet turn those events into final statistics, observable action catalogues or tactical interpretation.

### P3 — Structured Analytics + Actions

P3 answers:

> What measurable basketball facts, actions and summaries can we derive from trusted P2 events and P1 state?

P3 owns three kinds of output.

**Statistical aggregation**

- possession counts and duration;
- FGA / FGM / shooting percentages when outcome evidence supports them;
- passes and turnovers;
- rebounds;
- touches;
- player and lineup summaries;
- action frequencies and rates.

**Observable basketball actions**

- screens;
- cuts;
- drives;
- handoffs;
- relocations;
- catches and releases;
- dribble sequences;
- closeout events defined from measurable defender motion;
- paint touches;
- transition-action candidates.

**Spatial measurements and derived structures**

- shot location and shot distance;
- nearest-defender distance;
- court zones;
- spacing measurements;
- paint / corner occupancy;
- player speed and distance;
- catch-to-shot time and dribbles before a shot;
- shot charts;
- heatmaps;
- pass networks;
- touch maps;
- transition maps.

P3 records observable facts. It may say that a screen occurred, that a defender closed 3.2 m toward a shooter or that average spacing was 4.8 m. It should not yet label the full tactical system or infer coaching intent.

### P4 — Tactical Intelligence

P4 answers:

> What tactical pattern explains the structured actions and measurements?

P4 combines multiple P3 actions, spatial features and temporal context to infer:

- pick-and-roll / pick-and-pop;
- Spain PnR;
- double drag and other multi-screen structures;
- Horns, Floppy, Zipper and other offensive sets/actions;
- defensive coverage such as drop, switch, hedge, blitz or ICE;
- help responsibilities and rotations;
- weak-side / low-man behaviour;
- offensive and defensive shape;
- transition principles and transition-defence quality;
- contextual shot quality;
- advantage creation such as paint touch → help → kickout;
- mismatch exploitation;
- tactical tendencies;
- coaching / tactical interpretation.

The P3/P4 boundary is intentional:

```text
P3: screen happened at the left slot
P4: that screen initiated a high pick-and-roll against drop coverage

P3: nearest defender at release = 2.4 m
P4: open catch-and-shoot generated by weak-side help

P3: average spacing = 4.8 m
P4: offense is maintaining 5-out spacing to open the paint
```

---

## Architecture

```text
Video Session
    ↓
Frame Loader
    ↓
Gameplay Filter
    ↓
Perception Layer
    ├── player detection / tracking
    ├── team evidence
    ├── jersey-number evidence
    ├── ball candidates
    ├── temporal ball tracking
    ├── trajectory consistency
    ├── court keypoints / homography
    └── diagnostics / confidence
    ↓
PlayerState + BallState + CourtState               ← P1
    ↓
Possession + Basketball Event Primitives           ← P2
    ↓
Structured Statistics + Observable Actions         ← P3
    ↓
Spatial Measurements + Networks / Maps             ← P3
    ↓
Tactical Pattern Recognition + Interpretation      ← P4
    ↓
Database / Renderer / UI
```

The central boundary is:

```text
P1: reliable state
    ↓
P2: basketball events
    ↓
P3: measurable facts + actions
    ↓
P4: tactical interpretation
```

---

# P0 — Vision Foundation — COMPLETE

P0 completed the reusable vision/evaluation foundation required for P1.

## Frozen P0 ball contract

```text
model               weights/model_v1.0.0.pt
imgsz               960
reacquisition        OFF
backend              YOLO
```

The frozen V1–V5 generalization set remains the controlled reference footage for system comparisons.

The ball subsystem remains confidence-aware. `UNKNOWN` or missing ball state is preferable to unsupported concrete output.

---

# P1 — Perception + Game State — CURRENT PHASE

P1 starts from the strongest foundations already available:

```text
player tracks
team assignments
court_xy
homography
frame timestamps
motion / temporal state
ball evidence when reliable
```

## Current P1 runtime baseline — 2026-09-14

The current integrated P1 runtime reference is the Porto/Ovarense `0–60 s` run `20260914_140057_772692`. It was produced from the final `feature/p1-correctness-cleanup` state immediately before PR #46 was merged to `main`; the merged runtime is represented by merge commit `f13126497a99edf15973461cbb7ed870540308fc`.

Configuration:

```text
model               weights/model_v1.1.0.pt
segment             0–60 s
player conf         0.40
inference conf      0.03
ball conf           0.075
imgsz               960
device              mps
tracker             botsort.yaml
detection-only      ON
single-forward      ON
Smart Ball          OFF
Jersey              ON
jersey interval     24 frames
court calibration   OFF
```

Measured baseline:

```text
metric eligible                 62.89%
players / gameplay frame         8.521
team classification rate        99.50%
jersey identification rate      27.40%
ball direct                      62.08%
ball predicted hold              27.44%
ball effective trajectory        89.51%
ball missing                     10.49%
YOLO full-frame                  26.930 ms/frame
entity tracker                    8.161 ms/frame
Jersey runtime                    5.524 ms/frame
team classification               3.527 ms/frame
detector total                   44.810 ms/frame
total pipeline                   49.430 ms/frame
analysis FPS                     18.604
wall time                         79.927 s
```

This is the preferred P1 runtime/performance baseline. Smart Ball is truly OFF in this run: there is no `Detector/smart_ball` stage in the profiler.

Team classification is considered **closed for P1** after visual QA. Residual misclassifications are known and accepted at this phase; `team_classification_rate` is coverage, not an accuracy claim.

Jersey is intentionally more conservative after the P1 correctness gate. In this run, the runtime recorded 305 Jersey attempts, 45 identified tracks and 170 `raw_roster_disagreement` rejections. The lower 27.4% identification rate is therefore not interpreted as a regression by itself: unsupported roster-conditioned rewrites are now rejected instead of being silently accepted. These Jersey values remain coverage/acceptance diagnostics, **not correctness claims**. Human GT is still required before promoting a Jersey recognizer or declaring P1 identity complete.

Smart Ball four-mode ablation on the same `0–60 s` contract selected `off` as the P1 default:

```text
mode      effective ball   missing   Smart cost   detector   FPS
off             89.51%      10.49%       0.00 ms    101.22 ms  9.126
verify          89.51%      10.49%      25.26 ms    152.03 ms  6.291
zoom            89.51%      10.49%      31.13 ms    167.87 ms  5.729
full            90.47%       9.53%      54.36 ms    193.16 ms  4.960
```

The ablation timings were sequential MPS runs and therefore include thermal/load drift, but the quality/cost conclusion is clear: `verify` adds no measured coverage, `zoom` mainly exchanges predicted holds for direct detections without increasing effective coverage, and `full` adds only `+0.96 pp` effective coverage at substantial extra crop-inference cost. Smart Ball remains available as an experimental/diagnostic layer but is not the default P1 path.

Generated run directories are disposable. The README is the durable record of this baseline; raw local `evaluation/runs/`, `evaluation/worker_jobs/` and `evaluation/ablations/` outputs do not need to be retained once the relevant decision has been recorded here.

## P1.1 — Player Tracking Quality — COMPLETE

The current integrated operating point uses:

```text
model            weights/model_v1.1.0.pt
player detector  CourtFlux YOLO
tracker          BoT-SORT
imgsz            960
player conf      0.40
```

The frozen P1.1 evaluator measures track assignment, duration, fragmentation proxies, interruptions, players per gameplay frame, `court_xy` continuity, per-video robustness and runtime. True physical-player ID switches or recall require appropriate GT and are not inferred from diagnostics alone.

## P1.2 — RF-DETR Challenger — CLOSED / NOT PROMOTED

RF-DETR was evaluated as a basketball player-detector challenger using the same downstream tracking contract. The investigation is closed and RF-DETR was not promoted.

## P1.3 — Jersey Identification — ACTIVE VALIDATION

Jersey identification is explicitly part of P1 because it is identity evidence required by downstream game state and event attribution.

Preferred direction:

```text
player track
    ↓
player / torso crop
    ↓
jersey-number recognizer
    ↓
confidence + uncertainty + abstention
    ↓
temporal aggregation
    ↓
team + jersey identity evidence
    ↓
roster candidate when sufficiently reliable
```

The important metric is **track-level jersey precision and coverage**, not only single-frame OCR accuracy.

Evaluation includes:

- readable-crop exact / top-k accuracy;
- accepted observation precision;
- observation coverage;
- unreadable false-accept rate;
- track-level jersey accuracy;
- track-level coverage;
- time to identity lock;
- false identity lock rate;
- runtime overhead.

Roster information may validate identity evidence but must never rewrite an unsupported OCR/JNR prediction into an apparently correct result.

The current Lab/runtime Jersey implementation uses the EfficientNet-B0 checkpoint path with a P1 raw-vs-roster consistency gate. The separate ViT-B uncertainty-aware challenger remains under validation. No Jersey winner is promoted without human GT.

## P1.4 — Temporal Ball Tracker v1 — COMPLETE / INTEGRATED

The temporal tracker associates direct ball detections, estimates motion from direct observations only and bridges short gaps conservatively.

Predicted positions never recursively update direct-observation state. After the prediction budget expires, the state becomes missing/`UNKNOWN`.

## P1.5 — Ball Trajectory Consistency v1 — COMPLETE / INTEGRATED

Trajectory consistency ranks ball candidates using detector confidence, position residual, velocity consistency, acceleration consistency and bounding-box size consistency.

The scorer supports the temporal tracker but does not own track state. Trajectory-inconsistent detections may be rejected even when detector confidence is high.

## P1.6 — Detector `model_v1.1.0.pt` Validation — COMPLETE

On the common supervised detector test, `model_v1.1.0.pt` materially outperformed `model_v1.0.0.pt` across the primary detector classes and is the current integrated P1 detector baseline. The older `model_v1.0.0.pt` configuration remains useful as an operational regression reference for historical ball-pipeline comparisons; old downstream thresholds are not assumed to transfer automatically to v1.1.

## P1.7 — BallState — COMPLETE / INTEGRATED

P1 exposes the explicit downstream ball-state contract:

```text
OBSERVED
PREDICTED
MISSING
UNKNOWN
```

`BallState` preserves source/provenance, confidence, uncertainty, age since last direct observation, position when supported and relevant temporal evidence.

## P1.8 — PlayerState + CourtState — COMPLETE / INTEGRATED

The canonical P1 runtime exposes `PlayerState[]`, `BallState` and `CourtState`. Fields may remain independently `UNKNOWN`. A reliable track does not imply a known jersey number, and a jersey candidate does not imply a fully resolved player identity.

CourtState has two explicit evidence modes:

- calibrated geometry may assert `ONCOURT` or `OFFCOURT` from projected FIBA bounds;
- uncalibrated image-space fallback may assert positive court-surface evidence, but weak/negative HSV evidence abstains to `UNKNOWN` instead of forcing `OFFCOURT`.

Visual QA on the integrated Porto/Ovarense regression run passed the intended P1 behavior for on-court players and referee/cameraman off-court handling.

## P1.9 — Persistent Team + Jersey Evidence — TEAM CLOSED / JERSEY GT PENDING

P1 owns enough identity persistence to support later basketball reasoning:

```text
track
  ↓
team evidence
  ↓
jersey evidence
  ↓
temporal confidence / lock
  ↓
roster candidate when supported
```

Perfect cross-camera or cross-game person ReID is not a P1 requirement. Difficult identity merges may remain `UNKNOWN`.

Team evidence is temporally persisted and re-scored conservatively. The current Black-vs-Blue identity failure mode was reduced by making dark pixels mutually exclusive with chromatic team masks, requiring accepted frame-level team evidence before temporal accumulation and making active kits explicit. Visual QA still found some incorrect team labels, but the current behavior is accepted as sufficient for P1 and Team classification is closed.

Jersey remains ON in the integrated product configuration. The current correctness wrapper rejects roster-conditioned candidates when raw canonical OCR disagrees instead of forcing the roster number. Jersey correctness/promotion remains gated on human GT.

## P1.10 — Exit Contract

P1 is complete when CourtFlux exposes reproducible, confidence-aware and temporally coherent:

```text
PlayerState[]
BallState
CourtState
team + jersey identity evidence
timestamps + provenance + uncertainty
```

Current exit assessment:

```text
canonical P1 runtime/state contract     ✅
player / referee / CourtState visual QA ✅
ball runtime + continuity               ✅
team identity                           ✅ closed for P1
runtime performance                     ✅
Jersey human-GT correctness             🔄 remaining gate
final v1.2.0 release validation         ⏳ after Jersey GT
```

P1 is therefore **near exit but not yet formally complete**. The remaining blocking work is human-GT Jersey correctness and the final release gate; further Team or Smart Ball tuning is not on the P1 critical path unless a new regression is found.

---

# P2 — Basketball Understanding

P2 converts reliable P1 state into confidence-aware basketball event primitives.

```text
PlayerState + BallState + CourtState
                ↓
            possession
                ↓
 semantic basketball evidence
                ↓
 pass / turnover / shot / rebound primitives
                ↓
        timestamped event stream
```

## Planned P2 capabilities

- possession states: `POSSESSED`, `LOOSE`, `IN_FLIGHT`, `UNKNOWN`;
- possession transitions;
- pass candidates;
- turnover / steal candidates;
- shot-attempt candidates;
- made/miss evidence when sufficiently supported;
- offensive / defensive rebound candidates;
- semantic detections such as jump-shot, layup/dunk, possession, shot-block and ball-in-basket;
- event timestamps and windows;
- confidence, provenance, involved tracks/teams and uncertainty.

P2 must combine multiple evidence sources. A single semantic detection is never event truth by itself.

A key contract is that outcome decisions happen here when the evidence is perceptual/event-level. For example, P2 may emit:

```json
{
  "event": "SHOT_ATTEMPT",
  "outcome": "MADE",
  "confidence": 0.91
}
```

P3 may aggregate that event into `FGA=1` and `FGM=1`, but should not re-run visual made/miss inference.

---

# P3 — Structured Analytics + Actions

P3 converts trusted P2 events and P1 state into measurable basketball facts, observable actions and reusable analytics structures.

## P3.1 — Statistical Aggregation

- possession counts and duration;
- FGA / FGM / shooting percentages;
- passes;
- turnovers;
- rebounds;
- touches;
- action counts and rates;
- player summaries;
- lineup summaries.

## P3.2 — Observable Basketball Actions

- screens;
- cuts;
- drives;
- handoffs;
- relocations;
- catches;
- releases;
- dribble sequences;
- closeout events;
- paint touches;
- transition-action candidates.

These actions should have explicit measurable definitions. For example, P3 can identify that a defender rapidly closed distance toward a shooter; whether that closeout belongs to a specific defensive rotation is a P4 question.

## P3.3 — Spatial Measurements

- player spacing distances;
- shot distance;
- nearest-defender distance;
- court zones;
- paint and corner occupancy;
- ball-side / weak-side position;
- speed and distance travelled;
- catch-to-shot time;
- dribbles before shot;
- event origin/destination coordinates.

## P3.4 — Derived Analytics Structures

- shot charts;
- player heatmaps;
- touch maps;
- possession-zone maps;
- pass networks;
- transition maps;
- action-location maps;
- player and lineup spatial profiles.

P3 deliberately describes **what happened in measurable terms**. It does not label complex tactical systems merely because individual actions resemble them.

---

# P4 — Tactical Intelligence

P4 combines multiple P3 actions, spatial measurements and temporal sequences to infer tactical patterns and context.

## P4.1 — Offensive Structure

- pick-and-roll;
- pick-and-pop;
- Spain PnR;
- double drag;
- Horns;
- Floppy;
- Zipper;
- pindown and related screen structures;
- offensive-set recognition.

## P4.2 — Defensive Behaviour

- drop coverage;
- switching;
- hedge / show;
- blitz / trap;
- ICE;
- help responsibilities;
- low-man / weak-side help;
- tags;
- X-out and related rotations;
- defensive-shape interpretation.

## P4.3 — Transition Intelligence

P3 may measure:

```text
possession starts after defensive rebound
3 attackers cross half-court in 2.1 s
2 defenders remain behind the ball
shot occurs 5.4 s after possession start
```

P4 may interpret the sequence as:

- fast break;
- early offense;
- secondary transition;
- poor / effective transition defence.

## P4.4 — Shot Context + Quality

P3 provides measurable features such as:

- shot location;
- defender distance;
- shooter speed;
- catch-to-shot time;
- dribbles before shot;
- possession context.

P4 combines those features with tactical context to infer concepts such as:

- open catch-and-shoot;
- contested pull-up;
- rim attempt generated by drive-and-kick;
- shot generated against drop coverage;
- contextual shot quality.

## P4.5 — Advantage + Tendencies

- paint touch → help → kickout chains;
- mismatch exploitation;
- weak-side advantage creation;
- recurring offensive actions;
- recurring defensive coverages;
- lineup-specific tactical tendencies;
- opponent tendencies;
- coaching / tactical insights.

P4 should explain the tactical pattern behind P3 facts rather than duplicate P3 event/action detection.

---

## SAM2 status

SAM2 remains explicitly deferred from the active P1 critical path.

It may be revisited for severe track fragmentation, hard occlusions, cleaner player crops, mask-based court points or cross-track continuity if simpler approaches prove insufficient under measured evaluation.

---

## Current system status

### Working / established

- YouTube and local-video ingestion;
- local video caching;
- gameplay / non-game filtering;
- `weights/model_v1.1.0.pt` integrated detector baseline;
- player detection + BoT-SORT / ByteTrack paths;
- single-forward full-frame inference path;
- team classification infrastructure + temporal persistence, closed for P1;
- Jersey runtime path with Jersey ON in integrated validation;
- MPS / CUDA / CPU execution paths;
- detection-only performance pipeline;
- Beta Lab local + web analysis contract;
- persistence and diagnostics;
- frozen V1–V5 evaluation foundation;
- confidence-aware ball candidates + temporal association;
- BallState, PlayerState and CourtState canonical runtime;
- conservative fallback CourtState abstention;
- court geometry / homography and player `court_xy` projection;
- integrated P1 visual QA for player/court/referee behavior;
- Smart Ball OFF default with single-forward runtime validated at 18.604 FPS on the Porto/Ovarense 0–60 s reference run.

### Working / under validation

- jersey-number correctness and model promotion;
- track-level jersey identity policy;
- long-sequence homography stability.

### Not yet production-ready

- robust possession inference;
- pass / shot / turnover / rebound primitives;
- basketball semantic evidence integration;
- observable P3 action detection;
- structured spatial measurements and networks;
- structured game analytics;
- tactical pattern recognition;
- production real-time performance across target hardware.

---

## Diagnostics workflow

CourtFlux development follows controlled experiments:

```text
change one variable
    ↓
run the same video segment
    ↓
record perception + performance metrics
    ↓
compare against baseline
    ↓
keep / reject the change
```

Important principle:

> Do not hide weak perception behind downstream heuristics. Measure the layer that failed before deciding what to change.

Promotion rules:

```text
measure current baseline
        ↓
benchmark one challenger at a time
        ↓
keep only measured improvements
        ↓
prefer UNKNOWN over forced state
        ↓
build basketball meaning from multiple evidence sources
```

A feature is successful only when it improves the final basketball signal **and** its uncertainty, cost and failure modes remain visible.

---

## Running CourtFlux locally

### Initial setup

```bash
git clone <repository-url>
cd courtflux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For normal P1 visual validation, use the **Worker + web Lab** flow below. The old Streamlit Beta UI command is kept at the end of this section only for legacy/debug use.

### 1. Select the branch you want to test

The current P1 consolidation work is merged. Use `main`:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
source .venv/bin/activate
```

### 2. Run the P1 smoke tests before a visual run

```bash
PYTHONPATH=. python evaluation/court_state_v1_test.py
PYTHONPATH=. python evaluation/p1_correctness_cleanup_test.py
PYTHONPATH=. python evaluation/p1_runtime_consolidation_test.py
PYTHONPATH=. python evaluation/p1_identity_runtime_optimization_test.py
PYTHONPATH=. python evaluation/p1_team_identity_correctness_test.py
PYTHONPATH=. python evaluation/p1_jersey_acceptance_test.py
PYTHONPATH=. python evaluation/p1_gameplay_referee_qa_test.py
PYTHONPATH=. python evaluation/ball_candidate_pipeline_test.py
```

Optional syntax check for the current P1 runtime path:

```bash
python -m py_compile \
  core/frame_state.py \
  services/pipeline.py \
  services/diagnostics.py \
  services/vision/player_state_v1.py \
  services/vision/ball_state_v1.py \
  services/vision/court_state_v1.py \
  services/vision/single_forward_tracker.py \
  services/vision/shared_yolo_ball_backend.py \
  services/vision/single_forward_p0_detector.py \
  services/vision/smart_p0_detector.py \
  services/vision/hypothesis_p0_detector.py \
  services/vision/diagnostic_preview.py
```

No output from `py_compile` means the syntax check passed.

### 3. Start the analysis worker

The current P1 local validation defaults are:

```text
model               weights/model_v1.1.0.pt
player confidence   0.40
inference confidence 0.03
ball confidence     0.075
imgsz               960
tracker             botsort.yaml
single full-frame YOLO forward  ON
Smart Ball mode     off
Smart Ball crop imgsz 640 (used only by Smart Ball experimental modes)
Jersey              ON
jersey interval     24 frames
```

The development worker listens on `127.0.0.1:8787`. On the current development Mac it may be managed by the `com.courtflux.worker` launchd service; if so, restart it with:

```bash
launchctl kickstart -k gui/$(id -u)/com.courtflux.worker
```

For a manual worker process:

```bash
source .venv/bin/activate

export WORKER_API_TOKEN="$(cat ~/.config/courtflux/worker-api-token)"
export WORKER_HOST="127.0.0.1"
export WORKER_PORT="8787"

python3 -m services.worker_api
```

A healthy startup listens on:

```text
http://127.0.0.1:8787
```

The ngrok tunnel used by the current beta setup remains pointed at local port `8787`.

### 4. Start the web Lab

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Then open:

```text
http://localhost:3000/lab
```

The Lab submits jobs to the worker on port `8787`, polls status/logs, and exposes the latest annotated image and annotated MP4 preview.

### 5. Recommended P1 Lab configuration

For the current Porto/Ovarense visual regression run use:

```text
Video               data/OwiQgAMqyUs.mp4 (or the equivalent YouTube URL)
Start                0 s
End                  60 s
Model                weights/model_v1.1.0.pt
Player confidence    0.40
Inference confidence 0.03
Ball confidence      0.075
Image size           960
Device               mps on Apple Silicon
Tracker              botsort.yaml
Detection-only       ON
Live preview         ON
Jersey               ON
```

When testing P1 perception, compare the annotated MP4 and the run JSON together. In particular inspect:

- `COURT ... metrics:YES/NO` against real gameplay, replays and close-ups;
- fallback `COURT?`/`UNKNOWN` behavior in painted court areas and explicit `OFFCOURT` only when calibrated geometry supports it;
- referee/player role separation;
- canonical `PlayerState` count;
- canonical `BallState` status: `OBSERVED`, `PREDICTED`, or `MISSING`;
- rim/backboard basket geometry evidence;
- `metric_eligible_frames` rather than assuming every inferred frame is gameplay;
- player/team counts and identity diagnostics;
- ball coverage and continuity;
- detector timing and total analysis FPS.

With the current single-forward + Smart Ball OFF path, the profiler should structurally show:

```text
yolo_predict_full_frame
entity_tracker
team_classification
jersey_recognition
```

and should **not** spend time in `smart_ball` unless an experimental Smart Ball mode is explicitly enabled.

### 6. Smart Ball A/B modes

The P1 default is:

```bash
export BASKETBALL_SMART_BALL_MODE=off
```

Smart Ball remains available for explicit ablation/diagnostic work:

```bash
export BASKETBALL_SMART_BALL_MODE=verify
export BASKETBALL_SMART_BALL_MODE=zoom
export BASKETBALL_SMART_BALL_MODE=full
```

Use one mode per worker process. Restart the worker after changing the environment variable.

The single-forward detector itself can be rolled back for comparison with:

```bash
export BASKETBALL_SINGLE_FORWARD=0
```

Restart the worker after changing it.

### 7. Where to find results

Completed local analysis runs are written under:

```text
evaluation/runs/<run_id>/
```

Worker job state/log/preview artifacts are written under:

```text
evaluation/worker_jobs/<job_id>/
```

Smart Ball ablation reports, when generated, are written under:

```text
evaluation/ablations/
```

These are local disposable outputs. Once a decision becomes a baseline, record the configuration and measured result in this README and clean the generated directories instead of treating them as long-term backups.

### 8. Legacy Streamlit Beta UI

The legacy/debug UI can still be started with:

```bash
python3 -m streamlit run ui/beta.py --server.port 8523
```

For current P1 Lab validation, prefer the Worker + `/lab` flow above so the same runtime contract, diagnostics and annotated video path are exercised.

Benchmark claims must always record the exact model, thresholds, tracker, device, preprocessing and evaluation footage used.

---

## Status

✅ **P0 Vision Foundation Complete**

🚧 **Current phase: P1 — Perception + Game State / exit validation**

Current priority:

> Finish human-GT Jersey correctness validation, then run the final P1 release gate and prepare v1.2.0 before entering P2 basketball semantics.
