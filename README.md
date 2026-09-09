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

CourtFlux is in active development. The P0 vision/evaluation foundation is complete enough to support higher-level basketball analytics, while ball quality continues to improve as a parallel product-maturity track.

---

## Current phase

**P0 FOUNDATION COMPLETE**

The active phase is now:

```text
P1 — GAME STATE + EVENT PRIMITIVES
P1.1 — PLAYER TRACKING QUALITY
```

P0 completion does **not** mean ball tracking is solved or that the Ball Quality Maturity Gate has passed. P1 must tolerate missing/`UNKNOWN` ball state and use ball evidence only when it is sufficiently reliable.

Current P1 execution priority:

```text
1. Player tracking baseline + evaluation
2. RF-DETR as a player-detector challenger
3. Jersey-number recognition improvement
4. PlayerState + court state
5. Possession + event primitives
6. Basketball-specific semantic evidence as supporting signals
7. Spatial analytics + event timeline

SAM2 → deferred / contingency backlog
```

See:

- `docs/p1_roadmap.md` — canonical P1 execution roadmap and promotion rules;
- `docs/p0_foundation_snapshot.md` — frozen P0 exit snapshot and P1 boundary;
- `docs/system_quality_gate.md` — detailed ball-quality evaluation contract.

---

## Product hierarchy

```text
P0 — VISION FOUNDATION                              ✅ COMPLETE
     frozen evaluation contract
     ball candidate / hypothesis foundation
     court geometry / homography
     explicit uncertainty / diagnostics

            ↓

P1 — GAME-STATE + EVENT PRIMITIVES                 ← CURRENT PHASE
     player tracking quality
     detector/tracker challengers
     jersey-number evidence improvement
     PlayerState + court state
     possession / pass / turnover / shot / rebound candidates
     basketball-specific semantic evidence
     spatial analytics
     event timeline primitives

            ↓

P2 — PLAYER IDENTITY + ATTRIBUTION
     player vs referee / other
     persistent team + jersey identity
     roster resolution
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

The key P0→P1 architectural rule remains:

```text
players + teams + court + temporal state
                 ↓
          possession / events
                 ↑
       ball when sufficiently reliable
```

---

# P0 — Vision Foundation — COMPLETE

P0 established the measurable perception foundation required for P1.

## Frozen P0 ball contract

```text
model               weights/model_v1.0.0.pt
ball confidence     0.10
imgsz                960
official GT radius  <=40 px
reacquisition        OFF
backend              YOLO
```

The frozen V1–V5 generalization set remains unchanged.

Current Smart-v1 direct visible candidate recall:

```text
micro        56.54%
macro        58.74%
worst video  50.69%
```

Oracle-validated short-gap bridging reaches **72.81% micro at gap <=8**. This is an upper bound, not production tracker quality.

The ball subsystem remains:

```text
EXPERIMENTAL / CONFIDENCE-AWARE SUPPORTING SIGNAL
```

The Ball Quality Maturity Gate continues in parallel with P1.

---

# P1 — Game State + Event Primitives — CURRENT PHASE

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

## P1.1 — Player Tracking Quality

The first objective is to **measure before replacing**.

Canonical baseline:

```text
model            weights/model_v1.0.0.pt
player detector  CourtFlux YOLO
tracker          BoT-SORT
imgsz            960
player conf      0.10
```

Primary metrics:

- player recall / players visible vs detected;
- false player detections and referee leakage;
- track fragmentation;
- ID switches;
- track duration;
- missing-player frames;
- duplicate tracks;
- players per gameplay frame;
- on-court track persistence;
- `court_xy` continuity;
- team-label stability per track;
- per-video robustness;
- processing FPS / cost.

The target is **stable enough player tracks for game-state reasoning**, not perfect human identity.

### RF-DETR challenger

The Roboflow basketball detection/identification notebook is now an explicit P1 technical reference:

https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb

RF-DETR is evaluated as a **player-detector challenger**, not adopted automatically:

```text
CourtFlux YOLO + same downstream tracking
                vs
RF-DETR player detection + same downstream tracking
```

A challenger is promoted only if it materially improves player/track quality at acceptable runtime cost.

No RF-DETR experiment changes the frozen P0 ball baseline.

---

## P1 identity-support priority — Jersey Number Recognition

Jersey-number recognition now takes priority over SAM2 once the P1.1 baseline/evaluator is established.

This does **not** move final player attribution out of P2. It improves identity evidence early because better track-level jersey evidence helps validation and later event attribution.

Preferred direction:

```text
player track
    ↓
player bbox
    ↓
jersey-number region / candidate
    ↓
OCR or digit classifier
    ↓
confidence-aware observation
    ↓
temporal voting on track_id
    ↓
confirmed team + jersey number
    ↓
roster identity when reliable
```

Research reference:

https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr

Current EasyOCR remains the baseline. Potential challengers include jersey-region-specific OCR, a fine-tuned digit classifier and Roboflow dataset/model-derived approaches.

The important metric is **track-level jersey identity accuracy**, not only single-frame OCR accuracy.

Useful jersey metrics:

- accepted observation precision;
- observation coverage;
- track-level jersey accuracy;
- time to identity lock;
- false identity lock rate;
- usable observations per track;
- runtime overhead.

### SAM2 status

SAM2 is explicitly **deferred from the active P1 critical path**.

It remains a contingency challenger for severe track fragmentation or hard-occlusion recovery if the measured YOLO/RF-DETR + tracker paths are insufficient.

```text
baseline evaluation
      ↓
RF-DETR / tracker improvements
      ↓
tracks sufficient? ── yes → continue P1
      │
      no
      ↓
consider SAM2
```

---

## P1.2 — PlayerState + Court State

Build a stable per-frame representation such as:

```json
{
  "track_id": 17,
  "team": "FC Porto",
  "court_xy": [12.4, 6.8],
  "velocity": [0.12, -0.04],
  "active_on_court": true,
  "confidence": 0.93
}
```

This becomes the interface between perception and basketball reasoning.

---

## P1.3 — Possession

Possession combines multiple signals rather than depending on perfect ball tracking.

Potential evidence:

```text
high-confidence ball → player proximity
player motion
team continuity
previous possessor
court position
basketball-specific semantic evidence
```

Target states:

```text
POSSESSED
LOOSE
IN_FLIGHT
UNKNOWN
```

`UNKNOWN` is valid and preferable to forcing a false possessor.

---

## P1.4 — Pass Candidates

```text
Player A possession
        ↓
transition / loose interval
        ↓
Player B possession
```

If `A.team == B.team`, the transition can become a pass candidate. Reliable ball trajectory increases confidence but is not required through the full event.

---

## P1.5 — Turnover Candidates

```text
Team A possession
        ↓
transition
        ↓
Team B possession
```

This produces turnover / steal candidates. Fine-grained attribution comes later.

---

## P1.6 — Shot Attempts + Basketball Semantic Evidence

Basketball-specific semantic classes are a P1 **supporting-evidence** lane.

Candidate signals include:

```text
player-in-possession
player-jump-shot
player-layup-dunk
player-shot-block
ball-in-basket
rim
```

These signals may support event inference, but a semantic detection alone must never force an event.

Shot-attempt inference may combine:

- possessor state before release;
- `jump-shot` / `layup-dunk` visual evidence;
- reliable ball motion when available;
- hoop proximity;
- court location;
- possession transition;
- temporal consistency.

The first milestone remains `SHOT_ATTEMPT`.

---

## P1.7 — Rebounds

```text
SHOT
 ↓
loose / uncertain interval
 ↓
new possession
```

Same-team recovery suggests an offensive rebound; opponent recovery suggests a defensive rebound.

---

## P1.8 — Spatial Analytics

Homography and stable `PlayerState` enable:

- player heatmaps;
- possession zones;
- spacing;
- paint occupancy;
- corner occupancy;
- transition positioning;
- touch / shot-origin zones when event confidence supports them.

---

## P1.9 — Event Timeline

P1 culminates in confidence-aware timestamped event primitives:

```text
12:32  FC Porto possession
12:28  track17 → track8 pass candidate
12:25  shot attempt
12:23  Ovarense rebound candidate
```

Each event preserves:

- confidence;
- supporting evidence / provenance;
- involved tracks and teams;
- court location;
- frame/time window;
- uncertainty / abstention state.

For the detailed P1 contract, see `docs/p1_roadmap.md`.

---

# P2 — Player Identity + Attribution

P2 answers **who exactly performed the event**.

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

P1 may improve jersey evidence early, but robust persistent human identity and final event attribution remain P2 responsibilities.

The system must prefer `UNKNOWN` over forcing weak identity predictions.

---

# P3 — Structured Basketball Analytics

Once game-state primitives and attribution are trustworthy, CourtFlux can derive:

- possessions;
- shots;
- passes and pass networks;
- turnovers;
- rebounds;
- possession duration;
- touches;
- drives;
- paint touches;
- transition possessions;
- player / lineup summaries.

---

# P4 — Tactical Intelligence

Potential capabilities include:

- offensive / defensive spacing;
- defensive shape;
- screens / actions;
- pick-and-roll recognition;
- help rotations;
- closeouts;
- transition-defense quality;
- offensive-set recognition;
- defensive-scheme classification;
- expected shot quality;
- coaching / tactical insights.

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
    │      ├── CourtFlux YOLO baseline
    │      └── RF-DETR challenger
    ├── team classification
    ├── jersey-number evidence
    ├── basketball semantic evidence
    ├── ball candidates / hypotheses
    ├── court keypoints / homography
    └── diagnostics / confidence
    ↓
PlayerState + Court State               ← P1.1 / P1.2
    ↓
Possession + Event Primitive Layer      ← P1.3–P1.9
    ↓
Identity / Attribution Layer            ← P2
    ↓
Structured Analytics                    ← P3
    ↓
Tactical Intelligence                   ← P4
    ↓
Database / Renderer / UI

Ball Quality Track runs in parallel and feeds P1 only when reliable.
SAM2 remains outside the active critical path unless track-quality evidence justifies it.
```

---

## Current system status

### ✅ Working / established foundation

- YouTube and local-video ingestion;
- local video caching;
- gameplay / non-game filtering;
- canonical `weights/model_v1.0.0.pt` model;
- player detection;
- BoT-SORT / ByteTrack player tracking paths;
- persistent track IDs;
- team classification;
- jersey OCR / temporal identity infrastructure;
- MPS / CUDA / CPU paths;
- detection-only performance pipeline;
- Beta Lab local + web analysis contract;
- buffered persistence and diagnostics;
- frozen V1–V5 ball GT evaluation;
- confidence-aware ball candidate / hypothesis foundation;
- court geometry / homography;
- player `court_xy` projection.

### 🟡 Working but under validation

- player tracking quality across cameras;
- team-label stability;
- jersey-number evidence quality;
- persistent player identity;
- ball identity across varied footage;
- ball-quality maturity gate;
- long-sequence homography stability.

### 🔴 Not yet production-ready

- robust possession inference;
- pass / shot / turnover / rebound primitives;
- reliable player / referee separation;
- event-to-player attribution;
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

Every diagnostic run creates evidence under:

```text
evaluation/runs/<run_id>/
```

Important principle:

> Do not hide weak perception behind downstream heuristics. Measure the layer that failed before deciding what to change.

For P1, the same principle applies to YOLO vs RF-DETR, tracker changes, jersey OCR and semantic models: **baseline first, challenger second, promotion only with measured evidence**.

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

Frozen P0 ball benchmark baseline:

```text
model               weights/model_v1.0.0.pt
imgsz               960
YOLO inference      0.10
player confidence   0.10
ball confidence     0.10
reacquisition       OFF
```

The Beta Lab may use separate experimental thresholds; benchmark claims must always record the exact run configuration.

---

## Development philosophy

```text
measure → diagnose → change → rerun → compare
```

P1 promotion rules:

```text
measure current player baseline
        ↓
benchmark one challenger at a time
        ↓
keep only measured improvements
        ↓
prefer UNKNOWN over forced state
        ↓
build events from multiple evidence sources
```

A feature is successful only when it improves the final basketball signal **and** its uncertainty, cost and failure modes remain visible.

---

## Status

✅ **P0 Vision Foundation Complete**

🚧 **Current phase: P1.1 — Player Tracking Quality**

Current priority:

> Establish the player-tracking evaluation baseline, benchmark RF-DETR, improve track-level jersey-number evidence, then build PlayerState and possession/event primitives. SAM2 remains deferred unless measured track continuity makes it necessary.
