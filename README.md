# CourtFlux — Basketball Intelligence from Video

**CourtFlux** is a computer-vision system for turning standard basketball broadcast footage into structured player, ball, event and eventually tactical intelligence.

The long-term objective is to reconstruct a basketball game from ordinary video well enough to answer, with measurable confidence:

```text
Where is the ball?
Who probably has it?
What happened?
Who exactly did it?
What does it mean tactically?
```

CourtFlux is currently in an active beta/diagnostics phase. Development is deliberately dependency-driven: downstream basketball logic is not treated as reliable until the perception layer it depends on passes explicit quality gates.

---

## Product hierarchy

The official development hierarchy is:

```text
P0 — BALL QUALITY
     direct detection
     temporal association / tracking
     confidence + explicit ball-state model

            ↓

P1 — POSSESSION + EVENT PRIMITIVES
     likely possessor
     possession segments / changes
     pass / shot / turnover / rebound candidates

            ↓

P2 — PLAYER IDENTITY + ATTRIBUTION
     player vs referee / other
     team
     jersey
     persistent player identity
     event → team / player attribution

            ↓

P3 — STRUCTURED BASKETBALL ANALYTICS
     possessions
     shots
     passes / pass networks
     turnovers
     rebounds
     touches
     drives
     paint touches
     transition metrics
     lineup / player summaries

            ↓

P4 — TACTICAL INTELLIGENCE
     spacing
     defensive shape
     screens / actions
     pick-and-roll recognition
     help / rotations / closeouts
     offensive sets
     defensive schemes
     shot quality
     coaching interpretation
```

This order is intentional. Player/team/jersey work can progress in parallel, but **ball quality is the current gating dependency** for possession, events and tactical analytics.

---

# P0 — Ball Quality — CURRENT PRIORITY

Ball quality is a first-class product requirement. If the ball state is unreliable, possession segmentation, passes, shots, rebounds, turnovers and tactical interpretation become brittle regardless of player-tracking quality.

P0 is split into three separate engineering tracks rather than one generic “ball tracking” task.

## P0.1 — Direct ball detection

Goal: increase direct detector recall while protecting precision.

Current work / evaluation focuses on:

- small-ball examples
- motion blur
- partial occlusion
- difficult under-the-rim situations
- rim / backboard confusion
- crowd / background ball-like objects
- low-resolution broadcast frames
- fast passes
- distant camera views
- ball close to a player's body
- hard negatives

Direct detection must be evaluated with more than frame coverage. Required metrics include:

- precision
- recall
- direct detection frame coverage
- false positives
- performance by ball size
- performance by court region
- camera angle / zoom
- occlusion state
- distance to nearest player
- basketball event context

### Current resolution evidence

Controlled experiments on the same 30-second segment showed a strong resolution effect:

| Input size | Raw ball candidate frame rate | Analysis FPS on MPS |
| ---: | ---: | ---: |
| 640 | 26.10% | 18.06 |
| 768 | 40.43% | 17.36 |
| 960 | 50.47% | 16.19 |
| 1280 | 63.19% | 13.09 |

Lowering inference / ball confidence from `0.10` to `0.03` at 640 produced almost no useful gain (`26.10% → 26.24%`), indicating that threshold suppression was not the main bottleneck in that test.

`960` remains the current balanced development resolution; `1280` is a higher-accuracy / higher-cost operating point under evaluation.

## P0.2 — Temporal association / tracking

Independent per-frame detections are not sufficient. Higher resolution also increases candidate ambiguity, so CourtFlux needs to select the temporally plausible ball rather than simply the highest-confidence candidate.

Current association uses signals such as:

```text
detector confidence
+ predicted position
+ short-term velocity
+ trajectory consistency
+ ball-size consistency
```

The next iterations should progressively include:

- acceleration / motion plausibility
- gap-aware gating
- court constraints
- proximity to likely handler / shooter
- recent ball-state history
- better reacquisition logic

Success is judged by:

- effective trajectory coverage
- incorrect-switch / catastrophic-switch rate
- reacquisition frequency
- median / P90 / maximum trajectory gaps
- spatial trajectory error
- direct-vs-predicted provenance

A current 960 temporal benchmark produced:

```text
raw ball candidate rate        50.47%
direct associated rate         37.22%
predicted hold rate            24.77%
effective trajectory coverage  61.98%
large spatial jump rate         0.00%
```

This demonstrated that temporal association can remove implausible jumps and bridge short gaps, but smoothness alone is not proof of correctness. Visual and annotated ground-truth validation are still required.

## P0.3 — Ball confidence + explicit state model

CourtFlux will expose an explicit state and confidence for every ball frame so downstream systems know whether a location is directly observed or inferred.

Target state model:

```text
DETECTED
   ↓
TRACKED
   ↓
PREDICTED
   ↓
OCCLUDED
   ↓
LOST
   ↓
REACQUIRING
```

Each frame should eventually expose information such as:

```json
{
  "ball_state": "TRACKED",
  "source": "detector_associated",
  "confidence": 0.91,
  "position_confidence": 0.94,
  "trajectory_confidence": 0.88,
  "frames_since_direct_detection": 0
}
```

This prevents uncertainty from silently propagating into possession or event inference.

## P0 milestone gate

Before CourtFlux moves serious product effort into P1, the target is approximately:

```text
direct ball detection coverage      ≥ 70%
effective trajectory coverage       ≥ 90%
P90 trajectory gap                  < ~8 frames
reacquisitions                       materially reduced
catastrophic ball switches           near zero
precision / recall                   explicitly measured
confidence calibration               validated
```

These gates must hold across multiple games and camera conditions, not only one benchmark clip.

---

# P1 — Possession + Event Primitives

P1 converts trusted ball trajectory into basketball semantics.

Target chain:

```text
ball trajectory
    ↓
likely possessor
    ↓
possession segment
    ↓
possession change / loose ball
    ↓
event primitives
```

Planned capabilities:

- ball-to-player association
- likely ball carrier
- handler confidence
- possession start / end
- loose-ball state
- possession changes
- pass candidates
- shot-release candidates
- rebound candidates
- turnover candidates

P1 should work primarily from stable `track_id` + team information and must not wait for perfect jersey recognition.

---

# P2 — Player Identity + Attribution

P2 answers **who exactly performed the event**.

Target chain:

```text
human detection
    ↓
player / referee / coach / other
    ↓
team
    ↓
jersey
    ↓
persistent player identity
    ↓
event attribution
```

Current status:

- persistent player track IDs are working
- team classification is working but requires accuracy validation
- jersey checkpoint loads with exact architecture coverage
- jersey inference runs on MPS
- roster-constrained decoding still needs hardening
- referees can currently leak into the player pipeline

The system must prefer `UNKNOWN` over forcing a weak jersey prediction into the roster.

Player identity can continue improving in parallel with P0/P1, but it is not the current ball-quality gate.

---

# P3 — Structured Basketball Analytics

Once possession, events and attribution are trustworthy, CourtFlux can derive structured basketball data such as:

- shots by player / team
- passes
- pass networks
- turnovers
- rebounds
- possession duration
- touches per player
- transition possessions
- drives
- paint touches
- usage-like metrics
- assist / secondary-assist candidates
- player / lineup summaries

P3 is intended to represent observable basketball facts and derived event statistics, not higher-level tactical interpretation.

---

# P4 — Tactical Intelligence

P4 interprets the structured game state tactically.

Potential capabilities include:

- offensive / defensive spacing
- defensive shape
- screen and action recognition
- pick-and-roll recognition
- help rotations
- closeouts
- transition-defense quality
- offensive-set recognition
- defensive-scheme classification
- expected shot quality
- coaching / tactical insights

This work remains downstream of reliable P0–P3 data.

---

## Current system status

### ✅ Working

- YouTube and local-video ingestion
- local video caching for repeatable experiments
- gameplay / non-game frame filtering before expensive inference
- custom basketball YOLO model
- player detection
- ball detection
- BoT-SORT / ByteTrack integration
- persistent player track IDs
- team classification with active-kit aware colour scoring
- MPS inference on Apple Silicon
- CUDA / CPU support paths
- detection-only performance pipeline
- buffered SQLite persistence
- per-stage profiler
- per-frame diagnostics (`frames.jsonl`)
- run summaries (`summary.json`)
- Streamlit beta UI + diagnostic preview
- Next.js / Vercel control-plane beta
- separate inference / player / ball confidence thresholds
- temporal ball candidate association
- conservative short-gap ball prediction (`predicted_hold`)
- direct-vs-predicted diagnostics
- effective trajectory diagnostics

### 🟡 Working but under validation

- direct ball quality across varied footage
- temporal ball correctness during passes / occlusions
- longer-gap ball reacquisition
- team-classification accuracy
- jersey recognition
- persistent player identity

### 🔴 Not yet production-ready

- explicit ball state / confidence machine
- annotated ball precision / recall evaluation set
- reliable player / referee separation
- possession inference
- basketball event inference
- event-to-player attribution
- structured game analytics
- tactical intelligence
- production real-time performance across target hardware

---

## Gameplay filtering

A lightweight gameplay filter runs before YOLO inference and can reject non-game frames such as replays, close-ups, graphics and broadcast interruptions.

```text
Frame
  ↓
Gameplay Filter
  ├── non-game → skip expensive detector work
  └── gameplay → detector / tracking / analytics
```

This remains an important compute-saving stage, especially for full games and higher resolutions.

---

## Architecture

```text
Video Session
    ↓
Frame Loader
    ↓
Gameplay Filter
    ↓
Detector / Tracker
    ├── player detection / tracking
    ├── ball candidates
    ├── temporal ball association
    ├── team classification
    └── jersey / identity
    ↓
Ball State / Confidence Layer        ← P0 target
    ↓
Possession + Event Primitive Layer   ← P1
    ↓
Identity / Attribution Layer         ← P2
    ↓
Structured Analytics                 ← P3
    ↓
Tactical Intelligence                ← P4
    ↓
Database / Renderer / UI
```

Detection-only beta mode deliberately skips camera calibration, homography and tactical analytics while perception quality and performance are being measured.

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

Every run creates:

```text
evaluation/runs/<run_id>/
    run_config.json
    summary.json
    frames.jsonl
    console.log
```

Important principle:

> Do not hide weak perception behind downstream heuristics. Measure the layer that failed before deciding what to change.

Run IDs and full configuration should always accompany benchmark claims so results from different resolutions, thresholds or branches are not mixed.

---

## Running CourtFlux locally

```bash
git clone <repository-url>
cd BasketballAIAnalytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/beta.py --server.port 8523
```

Current ball-development baseline:

```text
imgsz              960
YOLO inference     0.10
player confidence  0.10
ball confidence    0.10
tracker             botsort.yaml
detection-only      ON
jersey              OFF
preview             ON
```

Jersey is intentionally disabled in isolated ball benchmarks so identity cost does not contaminate P0 comparisons.

---

## Performance notes

Approximate detection-only performance on the current Apple Silicon development machine has been:

```text
640   ~18 FPS
768   ~17 FPS
960   ~15–16 FPS
1280  ~13 FPS
```

YOLO inference remains the dominant runtime cost. Parsing / device-to-CPU synchronization is also a future optimization target.

Deployment direction remains:

```text
CourtFlux web UI / control plane
        ↓
remote inference worker
        ↓
CUDA + detection + tracking
        ↓
structured CourtFlux results
```

---

## Development philosophy

CourtFlux is being built results-first:

```text
measure → diagnose → change → rerun → compare
```

A feature is considered successful only when it improves the final basketball signal **and** its uncertainty, cost and failure modes remain visible.

The current dependency rule is:

```text
P0 quality gates P1
P1 enables P2 event attribution
P1 + P2 enable P3
P3 enables P4
```

Parallel work is allowed, but downstream features are not used to declare upstream perception solved.

---

## Status

🚧 **CourtFlux active development — P0 Ball Quality**

Current priority:

> Build a precise, high-recall, confidence-aware and temporally reliable ball subsystem before investing seriously in possession and event inference.
