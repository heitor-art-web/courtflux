# Basketball AI Analytics

🏀 **Basketball AI Analytics**

Basketball AI Analytics is a computer vision project designed to automatically extract tactical and performance insights from basketball game footage.

Using a custom basketball YOLO model, multi-object tracking, homography-based FIBA court calibration, and AI-powered analytics, the system transforms broadcast video into structured basketball data that can assist coaches, analysts, and player development staff.

## Product direction

The long-term objective is an automated basketball tactical intelligence platform capable of producing advanced metrics directly from normal game/broadcast footage without wearable sensors or dedicated tracking cameras.

Current / planned capabilities include:

- YouTube and local-video ingestion
- Player detection and multi-object tracking
- Ball detection
- Jersey/team classification
- Temporal jersey-number recognition
- Dynamic court calibration and homography
- Player positioning in real-world FIBA court coordinates
- Player speed estimation
- Offensive and defensive spacing
- Closest-defender pressure metrics
- Possession and turnover detection
- Transition defense timing
- Shot/event detection
- SQLite data storage
- Tactical HUD / radar visualization
- Heatmaps
- Offensive set recognition
- Defensive scheme classification
- Player trajectory prediction
- Pass network analysis
- Expected Shot Quality (xShot)
- AI-assisted coaching recommendations

## Architecture

The project is progressively evolving from a monolithic MVP into a modular processing pipeline:

```text
Video Session
    ↓
Frame Loader
    ↓
Gameplay Filter
    ↓
Camera Tracker / Homography
    ↓
Detector / Tracker
    ↓
Analytics Engine
    ↓
Database Service
    ↓
Renderer / UI
```

Each stage operates on a shared `FrameState` object.

---

# Beta Diagnostics Lab

The `beta-diagnostics` branch contains the first repeatable local beta workflow for debugging player and ball detection before adding more tactical features.

The reference video currently pre-filled in the UI is:

```text
https://www.youtube.com/watch?v=OwiQgAMqyUs
```

## Install

```bash
git checkout beta-diagnostics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the court keypoint service is available, place the Roboflow key in `.env`:

```text
ROBOFLOW_API_KEY=...
```

The beta no longer crashes without this key; it falls back to optical flow, but real court-coordinate analytics should only be evaluated when a valid non-identity homography is available.

## Launch the local site

From the repository root:

```bash
streamlit run ui/beta.py
```

The local site lets you select:

- YouTube URL or local video
- start/end timestamps
- model checkpoint
- detector confidence
- inference image size
- MPS / CPU / CUDA
- BoT-SORT or ByteTrack

The YouTube file is cached locally by video ID after the first successful download, so repeated benchmark runs use the same input video.

## Recommended first benchmark

Use exactly the same video interval for every comparison and change one variable at a time.

Suggested sequence:

```text
Run A — baseline
conf=0.40
imgsz=640
tracker=botsort.yaml

Run B — confidence test
conf=0.10
imgsz=640
tracker=botsort.yaml

Run C — resolution test
conf=0.10
imgsz=768
tracker=botsort.yaml

Run D — higher-resolution test
conf=0.10
imgsz=960
tracker=botsort.yaml
```

Start with a short 30–60 second interval that contains continuous live gameplay.

## Diagnostic output

Every run creates:

```text
evaluation/runs/<run_id>/
    run_config.json
    summary.json
    frames.jsonl
    console.log        # when launched through ui/beta.py
```

`frames.jsonl` records per-frame observations such as:

- raw class counts returned by YOLO/tracking
- raw player candidates
- raw ball candidates
- maximum raw ball confidence
- detections rejected by the court polygon
- parsed/final player count
- selected ball confidence
- whether the final ball is detector output, held/predicted, or missing
- whether the court keypoint model is active
- whether the homography is a real calibration or identity fallback
- detected events

`summary.json` aggregates these fields so multiple runs can be compared directly inside the Beta Lab.

## Important beta principle

Do not retrain the model based only on end-to-end misses. First determine whether the object was lost at:

```text
raw detector → court filter → tracker/parser → analytics
```

Only retrain once the diagnostic data shows that raw detector recall is the limiting step.

---

## Existing interfaces

Run the existing analytics CLI directly:

```bash
python3 services/analytics/analyzer.py --video <file-or-youtube-url> --start 00:00 --end 01:00
```

Run the original dashboard:

```bash
streamlit run ui/dashboard.py
```

Status: 🚧 Active Development
