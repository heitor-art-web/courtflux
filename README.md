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

CourtFlux is in active development. The P0 vision/evaluation foundation is complete, P1 is building reliable perception/game-state interfaces, and ball quality continues to improve under an explicit confidence-aware contract.

---

## Current phase

```text
P0 — VISION FOUNDATION                         ✅ COMPLETE
P1 — PERCEPTION + GAME STATE                  ← CURRENT PHASE
P2 — BASKETBALL UNDERSTANDING
P3 — STRUCTURED ANALYTICS + ACTIONS
P4 — TACTICAL INTELLIGENCE
```

P1 must tolerate missing/`UNKNOWN` state and must never force player, jersey or ball identity when evidence is weak.

Current P1 execution order:

```text
1. Player tracking baseline + evaluation                         ✅ complete
2. RF-DETR player-detector challenger                            ✅ closed / not promoted
3. Jersey identification                                         🔄 active validation
4. Temporal Ball Tracker v1                                      ⏳ implemented / validate
5. Ball Trajectory Consistency v1                                ⏳ implemented / validate
6. CourtFlux detector model_v1.1.0 validation                    ⏳ after training
7. BallState                                                     ⏳ pending
8. PlayerState + CourtState                                      ⏳ pending
9. Persistent team + jersey identity evidence                    ⏳ consolidate
10. P1 integration / exit validation                             ⏳ pending

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

## P1.1 — Player Tracking Quality — COMPLETE

Canonical baseline:

```text
model            weights/model_v1.0.0.pt
player detector  CourtFlux YOLO
tracker          BoT-SORT
imgsz            960
player conf      0.10
```

The frozen P1.1 evaluator measures track assignment, duration, fragmentation proxies, interruptions, players per gameplay frame, `court_xy` continuity, per-video robustness and runtime. True physical-player ID switches or recall require appropriate GT and are not inferred from diagnostics alone.

## P1.2 — RF-DETR Challenger — CLOSED / NOT PROMOTED

RF-DETR was evaluated as a basketball player-detector challenger using the same downstream tracking contract. The investigation is closed and RF-DETR was not promoted.

The canonical player detector therefore remains `weights/model_v1.0.0.pt` until a later CourtFlux detector version demonstrates a clear measured improvement.

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

## P1.4 — Temporal Ball Tracker v1 — IMPLEMENTED / VALIDATION PENDING

The temporal tracker associates direct ball detections, estimates motion from direct observations only and bridges short gaps conservatively.

Predicted positions never recursively update direct-observation state. After the prediction budget expires, the state becomes missing/`UNKNOWN`.

## P1.5 — Ball Trajectory Consistency v1 — IMPLEMENTED / VALIDATION PENDING

Trajectory consistency ranks ball candidates using detector confidence, position residual, velocity consistency, acceleration consistency and bounding-box size consistency.

The scorer supports the temporal tracker but does not own track state. Trajectory-inconsistent detections may be rejected even when detector confidence is high.

## P1.6 — Detector `model_v1.1.0.pt` Validation — PLANNED

A new CourtFlux detector is being trained separately from the current frozen baseline. When `model_v1.1.0.pt` is available, it will be compared directly against `model_v1.0.0.pt` using the same controlled P1.1 evaluation contract.

The comparison should keep constant, as far as possible:

```text
V1–V5 evaluation footage
same gameplay windows
imgsz=960 baseline contract
same BoT-SORT downstream tracker
same normalized tracking metrics
same failure attribution
```

`model_v1.1.0.pt` becomes canonical only if it demonstrates a clear system-level improvement without unacceptable regressions.

## P1.7 — BallState

P1 will expose an explicit downstream ball-state contract, for example:

```text
OBSERVED
PREDICTED
MISSING
UNKNOWN
```

`BallState` must preserve source/provenance, confidence, uncertainty, age since last direct observation, position when supported and relevant temporal evidence.

## P1.8 — PlayerState + CourtState

Example target representation:

```json
{
  "track_id": 17,
  "team": "FC Porto",
  "jersey_number": 14,
  "identity_confidence": 0.91,
  "court_xy": [12.4, 6.8],
  "velocity": [0.12, -0.04],
  "active_on_court": true,
  "tracking_confidence": 0.93
}
```

Fields may remain independently `UNKNOWN`. A reliable track does not imply a known jersey number, and a jersey candidate does not imply a fully resolved player identity.

## P1.9 — Persistent Team + Jersey Evidence

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

## P1.10 — Exit Contract

P1 is complete when CourtFlux exposes reproducible, confidence-aware and temporally coherent:

```text
PlayerState[]
BallState
CourtState
team + jersey identity evidence
timestamps + provenance + uncertainty
```

The exit criteria include measured jersey identity behavior, conservative ball-state semantics, validated temporal/trajectory logic, no forced low-confidence state and reproducible diagnostics.

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
- canonical `weights/model_v1.0.0.pt` detector;
- player detection;
- BoT-SORT / ByteTrack paths;
- team classification infrastructure;
- jersey OCR/JNR and temporal identity infrastructure;
- MPS / CUDA / CPU execution paths;
- detection-only performance pipeline;
- Beta Lab local + web analysis contract;
- persistence and diagnostics;
- frozen V1–V5 evaluation foundation;
- confidence-aware ball candidates;
- court geometry / homography;
- player `court_xy` projection.

### Working / under validation

- jersey-number evidence quality;
- track-level jersey identity policy;
- temporal ball tracking;
- trajectory consistency;
- team-label stability;
- ball identity across varied footage;
- long-sequence homography stability.

### Not yet production-ready

- consolidated `BallState`;
- consolidated `PlayerState`;
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

```bash
git clone <repository-url>
cd courtflux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run ui/beta.py --server.port 8523
```

Benchmark claims must always record the exact model, thresholds, tracker, device, preprocessing and evaluation footage used.

---

## Status

✅ **P0 Vision Foundation Complete**

🚧 **Current phase: P1 — Perception + Game State**

Current priority:

> Finish track-level jersey validation, validate Temporal Ball Tracker and Trajectory Consistency, benchmark `model_v1.1.0.pt` when training completes, then consolidate `BallState`, `PlayerState` and P1 identity evidence before entering P2 basketball semantics.
