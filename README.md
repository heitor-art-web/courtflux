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

CourtFlux is in active development. The vision/evaluation foundation is now complete enough to support higher-level basketball analytics, while ball quality continues to improve as a parallel product-maturity track.

---

## Current phase

**P0 FOUNDATION COMPLETE**

P0 completion means the perception foundation is now:

- measurable against frozen ground truth;
- explicit about uncertainty and failure modes;
- modular enough to support higher-level analytics;
- safe against known dangerous ball-reacquisition behaviour;
- equipped with court geometry / homography and player court coordinates;
- able to use ball evidence when reliable and abstain when it is not.

It does **not** mean that ball tracking is solved or that the ball-quality maturity gate has passed.

The next active phase is:

```text
P1.1 — PLAYER TRACKING QUALITY
```

Ball quality continues in parallel and is not a blocking dependency for entering P1.

See:

- `docs/p0_foundation_snapshot.md` — frozen P0 exit snapshot and P1 boundary;
- `docs/system_quality_gate.md` — detailed ball-quality evaluation contract.

---

## Product hierarchy

The current development hierarchy is:

```text
P0 — VISION FOUNDATION                              ✅ COMPLETE
     evaluation / frozen GT
     ball candidate pipeline
     safe hypothesis lifecycle
     court geometry / homography
     explicit uncertainty / diagnostics

            ↓

P1 — GAME-STATE + EVENT PRIMITIVES                 ← CURRENT PHASE
     player tracking quality
     PlayerState + court state
     likely possessor / possession state
     pass / shot / turnover / rebound candidates
     event timeline primitives

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

The key architectural change at the P0→P1 boundary is that **ball quality is no longer a hard prerequisite for possession/events**. P1 must tolerate `UNKNOWN` / abstaining ball state and use ball evidence only when sufficiently reliable.

Intended direction:

```text
players + teams + court + temporal state
                 ↓
          possession / events
                 ↑
       ball when sufficiently reliable
```

---

# P0 — Vision Foundation — COMPLETE

P0 established the technical foundation required to build and evaluate the rest of CourtFlux.

## P0.1 — Frozen evaluation contract

The current frozen ball benchmark contract is:

```text
model               weights/model_v1.0.0.pt
ball confidence     0.10
imgsz                960
official GT radius  <=40 px
reacquisition        OFF
backend              YOLO
```

The frozen V1–V5 generalization set remains unchanged.

Current Smart-v1 direct visible candidate recall is:

```text
micro        56.54%
macro        58.74%
worst video  50.69%
```

Oracle-validated short-gap linear bridging reaches **72.81% micro at gap <=8**. This is an upper bound, not production tracker quality.

## P0.2 — Ball hypothesis foundation

CourtFlux now has an experimental multi-hypothesis ball layer using:

```text
OC-SORT + CIoU
```

with explicit lifecycle:

```text
TENTATIVE → TRACKED → LOST → REMOVED
```

Safety rule:

```text
LOST / TENTATIVE / REMOVED
→ never emit a concrete ball location
```

Hard reacquisition remains OFF by default.

The hypothesis layer is useful for preserving candidate identity over time, but it is not treated as a direct production ball-output policy.

## P0.3 — Temporal identity evidence

A multivariate temporal Ball Identity diagnostic was evaluated with leave-one-video-out validation.

The result showed that temporal signals can identify useful high-confidence subsets, but calibration still varies materially between cameras/videos. The temporal arbiter is therefore retained as a supporting/experimental signal rather than a universal production identity rule.

## P0.4 — Court geometry / homography

CourtFlux now includes:

- reusable FIBA court geometry primitives;
- versioned Roboflow basketball-court v19 33-landmark → FIBA mapping;
- Serverless court-keypoint inference through the lightweight SDK;
- RANSAC anchors / inliers / reprojection RMSE diagnostics;
- periodic keypoint refresh with optical-flow hold;
- player feet projected to `court_xy`;
- ball center projected only as **planar/apparent** court context.

A real v19 calibration was validated with:

```text
anchors       12
inliers        6
RMSE        1.5261 px
valid        True
```

Ball homography coordinates are soft context only because an airborne basketball does not lie on the court plane.

## P0.5 — Court/player context research

Court/player context was evaluated as a Ball Identity feature using held-out leave-one-video-out validation.

At the 80% training-precision target:

| Profile | Held-out AUC | Output precision | Visible coverage | NV false |
| --- | ---: | ---: | ---: | ---: |
| temporal-only | 0.7656 | 73.09% | 37.72% | 15.63% |
| temporal + court/player context | 0.7232 | 72.25% | 39.10% | 16.09% |

The generic context model was **not promoted** into the production Ball Identity policy because it did not improve cross-video reliability.

Court/player context remains valuable infrastructure for P1 spatial/game-state analytics.

## Ball Quality Track — continues in parallel

The Ball Quality Gate remains active as a product-maturity milestone, but it no longer blocks P1 entry.

Engineering targets:

| Metric | Target |
| --- | ---: |
| usable visible coverage micro | `>=80%` |
| usable visible coverage macro | `>=78%` |
| worst-video usable coverage | `>=70%` |
| concrete output precision | `>=80%` |
| not-visible false-state | `<=5%` |
| unsupported visible gap P90 | `<8 frames` |
| catastrophic errors (`>160 px`) | `0` |

Approximately **90%** remains the later product North Star.

The ball subsystem is currently classified as:

```text
EXPERIMENTAL / CONFIDENCE-AWARE SUPPORTING SIGNAL
```

P1 must be able to continue when ball state is missing or uncertain.

---

# P1 — Game State + Event Primitives — CURRENT PHASE

P1 now starts from the strongest foundations already available:

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

First objective: measure and stabilize player tracks before building possession/event logic on top of them.

Primary metrics:

- track fragmentation;
- ID switches;
- track duration;
- players per gameplay frame;
- court-position continuity;
- team-label stability;
- on-court track persistence;
- per-video robustness.

The target is not perfect jersey identity. P1.1 needs stable enough player tracks to support game-state reasoning.

## P1.2 — PlayerState + court state

Build a clean per-frame representation such as:

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

This state becomes the basis for possession, spacing and event primitives.

## P1.3 — Possession

Possession should combine multiple signals rather than depend on perfect ball tracking.

Potential evidence:

```text
high-confidence ball → nearest player
ball/player temporal proximity
player motion
team continuity
previous possessor
court position
```

Target state model:

```text
POSSESSED
LOOSE
IN_FLIGHT
UNKNOWN
```

`UNKNOWN` is a valid output and is preferable to forcing a false possessor.

## P1.4 — Pass candidates

A pass can often be inferred from possession transition:

```text
Player A possesses
        ↓
transition / loose interval
        ↓
Player B possesses
```

If `A.team == B.team`, this becomes a pass candidate. Ball trajectory can increase confidence but does not need to be perfect through the entire pass.

## P1.5 — Turnover candidates

A team-possession transition can produce turnover / steal candidates:

```text
Team A possession
        ↓
transition
        ↓
Team B possession
```

Fine-grained classification comes later.

## P1.6 — Shot attempts

Shot-attempt inference can combine:

- possession before release;
- reliable ball motion when available;
- proximity to hoop;
- release region;
- court location;
- possession transition.

The first milestone is `SHOT_ATTEMPT`, not immediately made/missed/2PT/3PT/assist attribution.

## P1.7 — Rebounds

After a shot attempt:

```text
SHOT
 ↓
loose / uncertain interval
 ↓
new possession
```

Same-team recovery suggests an offensive rebound; opponent recovery suggests a defensive rebound.

## P1.8 — Spatial analytics

Homography enables higher-value court-based outputs even while ball quality continues improving:

- player heatmaps;
- possession zones;
- spacing;
- paint occupancy;
- corner occupancy;
- transition positioning;
- touch / shot-origin zones when supported by event confidence.

## P1.9 — Event timeline

Unify game-state primitives into timestamped events with confidence and provenance:

```text
12:32  FC Porto possession
12:28  track17 → track8 pass candidate
12:25  shot attempt
12:23  Ovarense rebound candidate
```

Each event should preserve:

- confidence;
- supporting evidence;
- involved tracks/teams;
- court location;
- frame/time window;
- uncertainty / abstention state.

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

- persistent player track IDs exist;
- team classification exists but still requires accuracy validation;
- jersey checkpoint loads and runs;
- roster-constrained decoding still needs hardening;
- referee leakage into the player pipeline remains a known issue.

The system must prefer `UNKNOWN` over forcing weak identity predictions.

---

# P3 — Structured Basketball Analytics

Once game-state primitives and attribution are trustworthy, CourtFlux can derive:

- shots by player / team;
- passes;
- pass networks;
- turnovers;
- rebounds;
- possession duration;
- touches per player;
- transition possessions;
- drives;
- paint touches;
- usage-like metrics;
- assist / secondary-assist candidates;
- player / lineup summaries.

P3 represents observable basketball facts and derived event statistics rather than high-level tactical interpretation.

---

# P4 — Tactical Intelligence

Potential capabilities include:

- offensive / defensive spacing;
- defensive shape;
- screen and action recognition;
- pick-and-roll recognition;
- help rotations;
- closeouts;
- transition-defense quality;
- offensive-set recognition;
- defensive-scheme classification;
- expected shot quality;
- coaching / tactical insights.

This remains downstream of reliable P1–P3 game-state data.

---

## Current system status

### ✅ Working / established foundation

- YouTube and local-video ingestion;
- local video caching for repeatable experiments;
- gameplay / non-game frame filtering;
- custom basketball YOLO model;
- player detection;
- ball candidate detection;
- BoT-SORT / ByteTrack player tracking paths;
- persistent player track IDs;
- team classification;
- MPS / CUDA / CPU support paths;
- detection-only performance pipeline;
- buffered persistence;
- per-stage profiler;
- per-frame diagnostics (`frames.jsonl`);
- run summaries (`summary.json`);
- frozen V1–V5 ball GT evaluation;
- Smart ball candidate pipeline;
- conservative ball association / bounded predicted hold;
- OC-SORT + CIoU experimental hypothesis layer;
- explicit `TENTATIVE/TRACKED/LOST/REMOVED` ball lifecycle;
- court geometry / homography;
- Roboflow v19 33-landmark mapping;
- player `court_xy` projection;
- context-feature diagnostics;
- P0 Quality Gate tooling and failure attribution.

### 🟡 Working but under validation

- player tracking quality across cameras;
- team-classification stability;
- persistent player identity;
- jersey recognition;
- ball identity across varied footage;
- ball-quality maturity gate;
- real-world homography stability across long sequences.

### 🔴 Not yet production-ready

- robust possession inference;
- pass / shot / turnover / rebound primitives;
- reliable player / referee separation;
- event-to-player attribution;
- structured game analytics;
- tactical intelligence;
- production real-time performance across target hardware.

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
    ├── team classification
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
```

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

Every diagnostic run creates:

```text
evaluation/runs/<run_id>/
    run_config.json
    summary.json
    frames.jsonl
    console.log
```

Important principle:

> Do not hide weak perception behind downstream heuristics. Measure the layer that failed before deciding what to change.

Run IDs and full configuration should accompany benchmark claims so results from different models, resolutions, thresholds or branches are not mixed.

---

## Running CourtFlux locally

```bash
git clone <repository-url>
cd courtflux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/beta.py --server.port 8523
```

Frozen ball benchmark baseline:

```text
model               weights/model_v1.0.0.pt
imgsz               960
YOLO inference      0.10
player confidence   0.10
ball confidence     0.10
reacquisition       OFF
```

Optional Roboflow/court dependencies are installed separately where needed.

---

## Performance notes

YOLO inference remains a dominant runtime cost. Court keypoint inference is intentionally treated as periodic calibration rather than an every-frame requirement when using the hosted path.

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

CourtFlux is built results-first:

```text
measure → diagnose → change → rerun → compare
```

A feature is successful only when it improves the final basketball signal **and** its uncertainty, cost and failure modes remain visible.

The dependency rule after P0 is now:

```text
P0 foundation enables P1
Ball quality matures in parallel
P1 enables reliable event primitives
P1 + P2 enable structured attribution / P3
P3 enables P4 tactical interpretation
```

Downstream features are never used to claim that upstream perception is solved.

---

## Status

✅ **P0 Vision Foundation Complete**

🚧 **Current phase: P1.1 — Player Tracking Quality**

Current priority:

> Stabilize and measure player tracks and per-frame court state, then build possession and event primitives that can use reliable ball evidence without depending on perfect ball tracking.
