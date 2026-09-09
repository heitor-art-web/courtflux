# CourtFlux P1 Roadmap — Game State + Event Primitives

**Status:** CURRENT PHASE  
**Entry condition:** P0 Vision Foundation Complete  
**Primary objective:** build reliable player/game-state primitives and event evidence without requiring perfect ball tracking.

P1 follows the P0 product boundary:

```text
players + teams + court + temporal state
                 ↓
          possession / events
                 ↑
       ball when sufficiently reliable
```

`UNKNOWN` / abstention remains a valid output whenever evidence is insufficient.

---

## P1 priorities

The current execution order is:

```text
1. Player tracking baseline + evaluation
2. RF-DETR as a player-detector challenger
3. Jersey-number recognition improvement
4. PlayerState + court state
5. Possession + event primitives
6. Basketball-specific semantic evidence as supporting signals
7. Spatial analytics + event timeline

SAM2: deferred / contingency backlog
```

This ordering is intentional. SAM2 is not required to enter or complete the first P1 milestones. It should only be promoted if measured player-track continuity remains a material blocker after the baseline and detector/tracker work.

---

## Technical references

P1 uses the following Roboflow material as **research/reference architecture**, not code to copy wholesale:

- Basketball player detection/tracking/identification notebook:  
  https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb
- Basketball jersey-number OCR project:  
  https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr

The useful ideas are evaluated independently inside the CourtFlux architecture and must beat the existing baseline before promotion.

---

# P1.1 — Player Tracking Quality

P1.1 begins by freezing and measuring the current player pipeline before tuning or replacing anything.

## Baseline

Current canonical player baseline:

```text
model            weights/model_v1.0.0.pt
player detector  CourtFlux YOLO
tracker          BoT-SORT
imgsz            960
player conf      0.10
```

ByteTrack remains an available comparison path.

## Evaluation contract

Primary metrics:

- player recall / players visible vs detected;
- false player detections and referee leakage;
- track fragmentation;
- ID switches;
- average / median track duration;
- missing-player frames;
- duplicate tracks;
- players per gameplay frame;
- on-court track persistence;
- `court_xy` continuity;
- team-label stability per track;
- per-video robustness;
- processing FPS / cost.

The target is **stable enough track identity for game-state reasoning**, not perfect human identity.

## RF-DETR challenger

RF-DETR is the first major challenger to the existing player detector.

Intended experiment:

```text
CourtFlux YOLO + same downstream tracking
                vs
RF-DETR player detection + same downstream tracking
```

The comparison must isolate detector quality as much as possible. RF-DETR is promoted only if it produces a material improvement in player recall/track quality at acceptable runtime cost.

No ball baseline is replaced by this experiment.

---

# P1 identity-support track — Jersey Number Recognition

Jersey recognition now takes priority over SAM2 as the next identity-support improvement once the P1.1 player baseline/evaluator is established.

This does **not** move full player attribution out of P2. P1 may improve jersey evidence early because better track-level identity evidence helps debugging, validation and later event attribution.

Preferred architecture:

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
roster identity when sufficiently reliable
```

CourtFlux already has useful downstream primitives such as temporal observations, voting and identity locking. The main research target is improving the quality of the per-frame jersey-number evidence.

Evaluation should prioritize **track-level identity accuracy**, not only single-frame OCR accuracy.

Metrics should include:

- OCR / classifier candidate accuracy;
- accepted observation precision;
- observation coverage;
- track-level jersey accuracy;
- time-to-lock identity;
- false identity lock rate;
- number of usable observations per track;
- runtime overhead.

Potential challengers:

- current EasyOCR path — baseline;
- jersey-region-specific OCR;
- Roboflow jersey-number model / dataset-derived approach;
- fine-tuned digit classifier;
- confidence-weighted ensemble if justified by evidence.

---

# SAM2 — Deferred / contingency backlog

SAM2 is explicitly **out of the active P1 critical path**.

It remains a useful future challenger for:

- severe player-track fragmentation;
- recovery through hard occlusions;
- mask-based foot-point estimation;
- cleaner player crops for team/jersey analysis.

SAM2 should be revisited only if the measured baseline shows that detector/tracker improvements cannot provide sufficiently stable player tracks.

```text
baseline player evaluation
        ↓
RF-DETR / tracker improvements
        ↓
track quality sufficient? ── yes → continue P1
        │
        no
        ↓
consider SAM2 challenger
```

---

# P1.2 — PlayerState + Court State

Build a stable per-frame player representation:

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

This becomes the stable interface between perception and basketball reasoning.

---

# P1.3 — Possession

Possession combines multiple evidence sources rather than depending on perfect ball tracking.

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

Semantic classes such as `player-in-possession`, when available and validated, are **supporting evidence**, never ground truth by themselves.

---

# P1.4 — Pass Candidates

Infer pass candidates from possession transition:

```text
Player A possession
        ↓
transition / loose / uncertain interval
        ↓
Player B possession
```

Same-team transition increases pass likelihood. Reliable ball trajectory can increase confidence but is not mandatory through the full event.

---

# P1.5 — Turnover Candidates

Detect team-possession transitions:

```text
Team A possession
        ↓
transition
        ↓
Team B possession
```

Fine-grained steal/turnover attribution comes later.

---

# P1.6 — Shot Attempts + Semantic Basketball Evidence

Basketball-specific semantic detection is a P1 research lane and should feed the event layer as supporting evidence.

Candidate signals include:

```text
player-in-possession
player-jump-shot
player-layup-dunk
player-shot-block
ball-in-basket
rim
```

Shot-attempt inference may combine:

- possessor state before release;
- `jump-shot` / `layup-dunk` visual evidence;
- reliable ball motion when available;
- hoop proximity;
- court location;
- possession transition;
- temporal consistency.

The first milestone remains `SHOT_ATTEMPT`. A semantic class alone must never force an event.

---

# P1.7 — Rebounds

After a shot candidate:

```text
SHOT
 ↓
loose / uncertain interval
 ↓
new possession
```

Same-team recovery suggests an offensive rebound; opponent recovery suggests a defensive rebound.

---

# P1.8 — Spatial Analytics

Use homography and stable `PlayerState` for:

- player heatmaps;
- possession zones;
- spacing;
- paint occupancy;
- corner occupancy;
- transition positioning;
- touch / shot-origin zones when confidence supports them.

---

# P1.9 — Event Timeline

Unify primitives into timestamped, confidence-aware events.

Each event preserves:

- confidence;
- provenance / supporting evidence;
- involved tracks and teams;
- court location;
- frame/time window;
- uncertainty / abstention state.

Example:

```text
12:32  FC Porto possession
12:28  track17 → track8 pass candidate
12:25  shot attempt
12:23  Ovarense rebound candidate
```

---

## P1 promotion rules

1. **Measure before replacing.** Current CourtFlux YOLO + BoT-SORT remains the baseline until a challenger beats it.
2. **One variable at a time.** Detector, tracker, OCR and semantic models should be evaluated independently where possible.
3. **No forced state.** Unknown player, possessor, jersey or event states are valid.
4. **Ball quality is parallel.** Ball evidence improves confidence but does not block P1.
5. **Semantic detection is evidence.** `player-in-possession`, `jump-shot`, etc. do not become event truth by themselves.
6. **Jersey before SAM2.** Jersey-number evidence is the active identity-support priority; SAM2 remains deferred unless track continuity data justifies it.
7. **Keep existing videos.** Use the established CourtFlux evaluation footage for controlled comparisons before introducing additional footage.
