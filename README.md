# Basketball AI Analytics

Basketball AI Analytics is a computer-vision system for turning standard basketball broadcast footage into structured, player-level and event-level data.

The core objective is to build a practical analytics pipeline that can identify players, track the ball continuously enough to infer basketball events, and eventually attribute those events to the correct team and player — without requiring wearable sensors or dedicated optical-tracking hardware.

The project is currently in an active beta/diagnostics phase. The focus is deliberately on getting the perception layer reliable and measurable before adding more tactical analytics.

---

## Product objective

The target pipeline is:

```text
broadcast video
    ↓
gameplay / non-game filtering
    ↓
player + ball detection
    ↓
player tracking
    ↓
team classification
    ↓
jersey / player identity
    ↓
continuous ball trajectory
    ↓
ball ↔ player possession association
    ↓
shots / passes / turnovers / rebounds
    ↓
structured basketball analytics
```

The key product principle is simple: event analytics are only useful if the underlying perception is trustworthy.

For that reason, the current development priority is not advanced tactical modelling. It is the reliable reconstruction of **who is on the court, where the ball is, and who the ball belongs to over time**.

---

## Current status

### ✅ Working

- YouTube and local-video ingestion
- Local video caching for repeatable experiments
- Gameplay / non-game frame filtering before expensive inference
- Custom basketball YOLO model
- Player detection
- Ball detection
- BoT-SORT / ByteTrack integration
- Persistent player track IDs
- Team classification with active-kit aware colour scoring
- MPS inference on Apple Silicon
- CUDA / CPU support paths
- Detection-only high-performance beta pipeline
- Buffered SQLite persistence
- Per-stage profiler and benchmark reports
- Per-frame diagnostics (`frames.jsonl`)
- Run summaries (`summary.json`)
- Local Streamlit beta UI with live diagnostic preview
- Separate confidence thresholds for YOLO inference, players, and ball
- Temporal ball candidate association
- Short-gap ball prediction (`predicted_hold`)
- Direct-vs-predicted ball diagnostics
- Effective trajectory coverage diagnostics

### 🟡 Working but still under validation

- Jersey-number recognition
- Persistent player identity resolution
- Team classification accuracy across referees / non-player humans
- Ball trajectory correctness during occlusions and fast passes
- Ball reacquisition after longer gaps

### 🔴 Not yet production-ready

- Reliable referee / player separation
- Reliable possession attribution
- Shot detection
- Pass detection
- Turnover detection
- Rebound attribution
- Event-to-player attribution
- Full-game identity reliability
- Production real-time performance on all hardware

---

## Ball tracking is currently the main engineering priority

Ball continuity is a first-class requirement because downstream events depend on it.

A system that detects players well but loses the ball cannot reliably infer:

- shots
- passes
- turnovers
- rebounds
- possession changes
- event attribution

Recent controlled experiments on the same 30-second segment show that inference resolution has a major effect on direct ball detection:

| Input size | Direct ball candidate frame rate | Analysis FPS on MPS |
| ---: | ---: | ---: |
| 640 | 26.10% | 18.06 |
| 768 | 40.43% | 17.36 |
| 960 | 50.47% | 16.19 |
| 1280 | 63.19% | 13.09 |

Lowering the inference / ball confidence threshold from `0.10` to `0.03` at 640 produced almost no improvement (`26.10% → 26.24%`), showing that resolution — not threshold suppression — is currently the stronger lever.

For development, `960` is the current balanced operating point. `1280` provides stronger raw ball coverage but carries a much larger performance cost.

---

## Temporal ball association

Independent per-frame ball detections are not enough for basketball events. Multiple ball-like candidates can exist in a frame, especially at higher resolution.

The current temporal ball tracker scores candidates using:

```text
detector confidence
+ proximity to predicted position
+ short-term velocity
+ trajectory consistency
+ ball-size consistency
```

If no plausible direct candidate exists, the tracker can bridge a short gap with a conservative constant-velocity prediction for up to 5 frames.

The important diagnostic distinction is:

```text
raw candidate       YOLO produced a possible ball

direct associated  tracker accepted a real detector observation

predicted hold      short temporal prediction, not a detector hit

missing             no usable ball trajectory for that frame
```

On the current 960 beta run, temporal association changed the behaviour from noisy frame-independent detections to a smoother trajectory:

```text
raw ball candidate rate        50.47%
direct associated rate         37.22%
predicted hold rate            24.77%
effective trajectory coverage  61.99%
large spatial jump rate         0.00%
```

The tracker intentionally rejects some raw YOLO candidates to avoid implausible jumps.

This is still under visual validation: a mathematically smooth trajectory is not automatically a correct basketball trajectory.

---

## Player identity

The intended player identity chain is:

```text
YOLO player detection
    ↓
stable track_id
    ↓
team classification
    ↓
jersey recognition
    ↓
roster lookup
    ↓
persistent player identity
```

The local jersey checkpoint is now loaded with exact architecture coverage:

```text
checkpoint keys matched: 364 / 364
missing keys:             0
unexpected keys:          0
architecture:             single-frame EfficientNet-B0 + projection + digit heads
```

Jersey inference runs successfully on MPS, but the current roster-constrained decoding still requires additional validation. The system must prefer `UNKNOWN` over forcing an unsupported jersey number into the known roster.

The current roster configuration also contains players whose jersey numbers are still unknown (`?_N` entries), so those players cannot yet be resolved from jersey number to name.

---

## Known issue: referees and other humans

The current detector treats YOLO `person` / `player` detections as player candidates before team classification.

That means referees can currently enter the player pipeline and may eventually receive a team label from accumulated colour evidence.

A future classification stage will explicitly separate:

```text
human
  ↓
player / referee / coach / other
  ↓
team classification only for players
```

This is important before possession and event attribution become production features.

---

## Gameplay filtering

The pipeline includes a lightweight gameplay filter before YOLO inference.

It uses court appearance, motion, scene change and edge information to reject obvious non-game frames such as replays, close-ups, graphics and other broadcast interruptions.

```text
Frame
  ↓
Gameplay Filter
  ├── non-game → skip expensive detector work
  └── gameplay → run YOLO / tracking / analytics
```

This optimization becomes especially valuable on full-game video and at larger inference resolutions.

---

## Architecture

The project is structured around a shared frame state and modular processing stages:

```text
Video Session
    ↓
Frame Loader
    ↓
Gameplay Filter
    ↓
Detector / Tracker
    ├── player detection
    ├── ball candidates
    ├── team classification
    ├── jersey / identity
    └── temporal ball association
    ↓
Analytics Engine
    ↓
Database Service
    ↓
Renderer / Diagnostics / UI
```

Detection-only beta mode deliberately skips camera calibration, homography and tactical analytics so that detector quality and performance can be measured independently.

---

## Beta diagnostics workflow

The main development loop is based on controlled experiments:

```text
change one variable
    ↓
run the same video segment
    ↓
record performance + perception metrics
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
    console.log        # when launched through the Streamlit beta UI
```

`summary.json` includes:

- player counts
- team classification rates
- jersey identity diagnostics
- raw ball candidate rate
- direct associated ball rate
- predicted-hold rate
- effective trajectory coverage
- ball streak / gap statistics
- candidate ambiguity
- spatial-jump diagnostics
- model / threshold / resolution configuration
- per-stage timing
- overall FPS and realtime factor

The guiding rule is:

> Do not retrain a model until diagnostics show that the model itself is the bottleneck.

---

## Running the beta locally

### Install

```bash
git clone <repository-url>
cd BasketballAIAnalytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Launch the beta UI

```bash
streamlit run ui/beta.py --server.port 8523
```

The beta UI supports:

- YouTube or local video
- start / end timestamps
- model checkpoint selection
- detector confidence
- YOLO inference confidence
- player confidence
- ball confidence
- image size
- MPS / CPU / CUDA
- BoT-SORT / ByteTrack
- detection-only mode
- jersey recognition toggle
- rolling diagnostic preview

### Current recommended ball-development configuration

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

Jersey is intentionally disabled during isolated ball benchmarks so that recognition cost does not contaminate the comparison.

---

## Performance notes

On the current Apple Silicon development machine, approximate detection-only performance is:

```text
640   ~18 FPS
768   ~17 FPS
960   ~16 FPS
1280  ~13 FPS
```

The main runtime cost remains YOLO inference. Parsing also has a measurable cost and currently performs per-box device-to-CPU access, which is a future optimization target.

The project is designed so that local MPS development can later move to a CUDA worker without changing the high-level product architecture.

Target deployment direction:

```text
Web UI / control plane
        ↓
remote inference worker
        ↓
CUDA + YOLO + tracking
        ↓
structured results
```

---

## Near-term roadmap

### 1. Validate effective ball trajectory

- repeat the 960 temporal run with the new trajectory metrics
- measure effective streaks and true missing gaps
- visually inspect `BALL` vs `BALL PRED`
- test temporal association at 1280
- choose the final accuracy / performance operating point

### 2. Ball-to-player possession association

- determine the likely ball carrier
- use spatial proximity + trajectory + track continuity
- handle dribble separation and brief occlusion
- detect possession changes conservatively

### 3. Event primitives

Once ball continuity and possession are reliable:

- pass candidate detection
- shot-attempt candidate detection
- turnover candidate detection
- rebound / loose-ball sequences

### 4. Human-role separation

- distinguish player vs referee vs coach / sideline personnel
- prevent referees from entering team / identity statistics

### 5. Player identity hardening

- make roster constraints non-forcing
- improve jersey crop quality and confidence gating
- validate player identity visually
- fill missing roster jersey numbers
- add stronger temporal / appearance identity continuity

### 6. Performance optimization

- reduce per-box MPS → CPU synchronization
- optimize parsing / batching
- benchmark CUDA worker performance
- preserve the gameplay filter as an early compute-saving stage

### 7. Basketball analytics

Only after the perception layer is sufficiently reliable:

- shots by player
- passes and pass networks
- turnovers
- possessions
- rebounds
- transition events
- player movement / spacing
- tactical and coaching analytics

---

## Development philosophy

This project is being built results-first:

```text
measure → diagnose → change → rerun → compare
```

The goal is not to hide imperfect perception behind heuristics. Raw detector behaviour, temporal reconstruction and downstream analytics are kept separate so each layer can be measured objectively.

A beta feature is considered successful when it improves the final basketball signal **and** its cost / failure modes are visible in the diagnostics.

---

## Status

🚧 **Active development — perception and ball-trajectory beta**

Current priority:

> Build a reliable continuous ball trajectory and stable player identity layer before implementing shot, pass and turnover analytics.
