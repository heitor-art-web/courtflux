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
P3 — STRUCTURED BASKETBALL ANALYTICS
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

> What basketball action or game event is happening?

P2 consumes `PlayerState + BallState + CourtState` and builds:

- possession states and transitions;
- pass candidates;
- turnover candidates;
- shot attempts and outcome evidence;
- rebound candidates;
- basketball-specific semantic evidence;
- spatial analytics;
- confidence-aware event timeline.

Semantic classes such as `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `ball-in-basket` and `rim` are supporting evidence only and must never force an event by themselves.

### P3 — Structured Basketball Analytics

P3 derives structured statistics and summaries from trusted P2 events:

- possessions;
- shots / makes / misses when supported;
- passes and pass networks;
- turnovers;
- rebounds;
- possession duration;
- touches;
- drives;
- paint touches;
- transition possessions;
- player and lineup summaries.

### P4 — Tactical Intelligence

P4 interprets the structured basketball state tactically:

- offensive / defensive spacing;
- defensive shape;
- screens / actions;
- pick-and-roll recognition;
- help / rotations / closeouts;
- offensive sets;
- defensive schemes;
- shot quality;
- coaching interpretation.

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
Possession + Basketball Event Primitives            ← P2
    ↓
Spatial interpretation + Event Timeline             ← P2
    ↓
Structured Analytics                                ← P3
    ↓
Tactical Intelligence                               ← P4
    ↓
Database / Renderer / UI
```

The central boundary is:

```text
P1: reliable state
    ↓
P2: basketball meaning
    ↓
P3: statistics
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

Jersey identification is now explicitly part of P1 because it is identity evidence required by downstream game state and event attribution.

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

P2 moves basketball primitives, semantic evidence and spatial interpretation out of the perception phase.

```text
PlayerState + BallState + CourtState
                ↓
            possession
                ↓
 semantic basketball evidence
                ↓
 pass / turnover / shot / rebound primitives
                ↓
       spatial interpretation
                ↓
           event timeline
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
- spatial zones, shot origins, paint/corner occupancy, transition positioning and spacing primitives;
- event timeline with confidence, provenance, involved tracks/teams, court location and uncertainty.

P2 must combine multiple evidence sources. A single semantic detection is never event truth by itself.

---

# P3 — Structured Basketball Analytics

Once P2 events are trustworthy, CourtFlux derives statistics and summaries such as:

- possession counts and duration;
- shots / makes / misses;
- passes and pass networks;
- turnovers;
- rebounds;
- touches;
- drives;
- paint touches;
- transition metrics;
- player and lineup summaries.

---

# P4 — Tactical Intelligence

Potential capabilities include:

- spacing;
- defensive shape;
- screens and actions;
- pick-and-roll recognition;
- help rotations;
- closeouts;
- offensive sets;
- defensive schemes;
- shot quality;
- coaching / tactical insights.

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
- spatial event analytics;
- event timeline;
- structured game analytics;
- tactical intelligence;
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
