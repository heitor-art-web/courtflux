# CourtFlux P1 Roadmap — Perception + Game State

**Status:** CURRENT PHASE  
**Entry condition:** P0 Vision Foundation Complete  
**Primary objective:** produce stable, confidence-aware `PlayerState`, `BallState`, team and jersey identity evidence from broadcast basketball video.

P1 is the boundary between raw perception and basketball understanding:

```text
video
  ↓
player / ball / court perception
  ↓
tracking + temporal evidence
  ↓
team + jersey identity evidence
  ↓
PlayerState + BallState + CourtState
  ↓
──────────────────────────── P1 COMPLETE
  ↓
P2 basketball semantics / events / spatial reasoning
```

`UNKNOWN` / abstention remains a valid output whenever evidence is insufficient.

---

## Product phase boundary

CourtFlux now uses the following hierarchy:

```text
P0 — Vision Foundation
P1 — Perception + Game State
P2 — Basketball Understanding
P3 — Structured Basketball Analytics
P4 — Tactical Intelligence
```

### P1 answers

> Who/what is visible, where is it, and what identity/state evidence do we trust now?

### P2 answers

> What basketball action or game event is happening?

### P3 answers

> What structured statistics and summaries result from those events?

### P4 answers

> What do those patterns mean tactically?

---

## P1 execution order

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

SAM2: deferred / contingency backlog
```

The new `model_v1.1.0.pt` validation is a separate controlled comparison against `model_v1.0.0.pt`; it does not invalidate conclusions from jersey or ball experiments performed on the frozen baseline.

---

# P1.1 — Player Tracking Quality — COMPLETE

The canonical baseline was frozen and measured before changing detector/tracker behavior.

```text
model            weights/model_v1.0.0.pt
player detector  CourtFlux YOLO
tracker          BoT-SORT
imgsz            960
player conf      0.10
```

The P1.1 evaluation contract includes:

- player recall / players visible vs detected where GT supports it;
- false player detections and referee leakage;
- track fragmentation and ID-switch diagnostics;
- average / median track duration;
- missing-player frames and duplicate tracks;
- players per gameplay frame;
- on-court track persistence;
- `court_xy` continuity;
- team-label stability per track;
- per-video robustness;
- processing FPS / cost.

The target is **stable enough track identity for game-state reasoning**, not perfect long-term person re-identification.

---

# P1.2 — RF-DETR Player Detector Challenger — CLOSED / NOT PROMOTED

RF-DETR was evaluated as a player-detector challenger under the same downstream tracking contract.

```text
CourtFlux YOLO + same downstream tracking
                vs
RF-DETR player detection + same downstream tracking
```

The investigation is complete and RF-DETR is not promoted to the canonical player detector. The production/reference baseline remains CourtFlux YOLO `model_v1.0.0.pt` until a later versioned CourtFlux detector beats it under the same evaluation contract.

No RF-DETR experiment changes the frozen P0 ball baseline.

---

# P1.3 — Jersey Identification — ACTIVE

Jersey identification is an explicit P1 responsibility because it is part of the game-state identity evidence consumed by later basketball reasoning.

Preferred architecture:

```text
player track
    ↓
player / torso crop
    ↓
jersey-number recognizer
    ↓
confidence + uncertainty + abstention
    ↓
temporal aggregation on track
    ↓
team + jersey evidence
    ↓
roster identity candidate when sufficiently reliable
```

The primary metric is **track-level jersey identity precision and coverage**, not only single-frame OCR accuracy.

Required metrics include:

- readable-crop exact / top-k accuracy;
- accepted observation precision;
- observation coverage;
- unreadable false-accept rate;
- track-level jersey accuracy;
- track-level coverage;
- time to identity lock;
- false identity lock rate;
- usable observations per track;
- runtime overhead.

Current investigation includes multiple discarded/diagnostic challengers and an uncertainty-aware ViT-B path. A model or policy is promoted only after train-calibrated thresholds are frozen and evaluated on held-out data.

Roster constraints must never rewrite an unsupported model prediction into an apparently correct answer. `UNKNOWN` remains valid.

---

# P1.4 — Temporal Ball Tracker v1 — IMPLEMENTED / VALIDATION PENDING

The temporal tracker consumes direct basketball detections and provides conservative short-gap temporal continuity.

Core semantics:

- direct observations own state;
- smoothed image-space velocity comes from direct detections only;
- short gaps may be bridged with constant-velocity prediction;
- predicted positions never recursively mutate direct-observation state;
- long-gap reacquisition remains disabled;
- after the prediction budget expires, the result becomes missing/`UNKNOWN`.

This step must be validated against the established ball-quality evaluation contract before promotion.

---

# P1.5 — Ball Trajectory Consistency v1 — IMPLEMENTED / VALIDATION PENDING

Trajectory consistency ranks direct ball candidates using supporting motion evidence:

- detector confidence;
- position residual from temporal prediction;
- velocity consistency;
- acceleration consistency;
- bounding-box size consistency.

Motion quantities are normalized by frame geometry. The trajectory scorer does not own track state.

A high-confidence but trajectory-inconsistent detection may be rejected in favor of a lower-confidence coherent candidate. If no candidate passes, CourtFlux falls back to the conservative short-gap temporal behavior and then to missing/`UNKNOWN`.

---

# P1.6 — CourtFlux Detector `model_v1.1.0.pt` Validation — PLANNED

When the new detector training is complete, `model_v1.1.0.pt` is evaluated as a versioned CourtFlux candidate against the current canonical `model_v1.0.0.pt`.

The comparison must keep the evaluation contract as constant as possible:

```text
same V1–V5 evaluation footage
same gameplay windows
same imgsz=960 baseline contract
same BoT-SORT downstream tracker
same normalized tracking metrics
same failure attribution
```

The validation should answer:

- does player detection improve across videos?
- are false players / referee or staff leakage controlled?
- does track persistence improve?
- do short tracks / fragmentation decrease?
- is ball/rim/backboard behavior preserved or improved where relevant?
- what is the runtime impact?

`model_v1.1.0.pt` becomes canonical only if it provides a clear system-level improvement without unacceptable regressions.

---

# P1.7 — BallState

Create an explicit downstream ball-state contract so analytics never confuse observation, prediction and absence.

Example states:

```text
OBSERVED
PREDICTED
MISSING
UNKNOWN
```

A `BallState` should preserve at least:

- image position / bbox when supported;
- optional apparent/projected court position with provenance;
- velocity / motion evidence;
- direct-vs-predicted source;
- age since last direct observation;
- confidence / uncertainty;
- validity flags.

Low-confidence or expired evidence must produce `MISSING`/`UNKNOWN`, never a fabricated concrete ball state.

---

# P1.8 — PlayerState + CourtState

Build a stable per-frame player representation that joins perception, tracking, team, jersey and court evidence.

Example:

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

Fields may remain `UNKNOWN` independently. A strong track does not imply a known jersey, and a jersey candidate does not imply a fully resolved player identity.

`PlayerState`, `BallState` and `CourtState` form the stable interface between P1 perception and P2 basketball reasoning.

---

# P1.9 — Persistent Team + Jersey Identity Evidence

P1 owns enough identity persistence to support downstream event attribution:

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

P1 does **not** require perfect cross-game or cross-camera person ReID. Difficult identity merges may remain `UNKNOWN` rather than forcing a player.

Advanced appearance/ReID or SAM2-style continuity remains a future challenger if measured failures justify it.

---

# P1.10 — Integration / Exit Validation

P1 is complete when the system exposes confidence-aware, temporally coherent state objects suitable for basketball reasoning:

```text
PlayerState[]
BallState
CourtState
identity evidence
timestamps + provenance + uncertainty
```

The P1 exit contract requires:

- stable player tracking baseline;
- measured jersey identity policy with explicit abstention;
- conservative temporal ball state;
- trajectory-consistency behavior validated;
- no forced player/jersey/ball state when evidence is weak;
- reproducible diagnostics and failure attribution;
- versioned detector baseline, including `model_v1.1.0` decision when available.

P1 does **not** require possession, passes, shots, rebounds or spatial event analytics. Those move to P2.

---

# P2 Preview — Basketball Understanding

P2 consumes P1 state and answers **what is happening in basketball terms**.

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

Planned P2 lanes:

- possession states and transitions;
- pass candidates;
- turnover candidates;
- shot attempts and outcome evidence;
- rebounds;
- basketball-specific semantic evidence such as `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `ball-in-basket` and `rim`;
- spatial analytics using homography and stable state;
- confidence-aware event timeline.

Semantic detections remain **supporting evidence**. A single class must never force an event.

---

# SAM2 — Deferred / Contingency Backlog

SAM2 remains outside the active P1 critical path.

Potential future uses:

- severe player-track fragmentation;
- recovery through hard occlusions;
- mask-based foot-point estimation;
- cleaner player crops for team/jersey analysis;
- cross-track continuity where simpler evidence is insufficient.

Promote SAM2 only if measured track/identity failures justify the additional complexity and runtime.

---

## Promotion rules

1. **Measure before replacing.** A baseline remains canonical until a challenger beats it.
2. **One variable at a time.** Detector, tracker and jersey models/policies should be isolated where possible.
3. **No forced state.** Unknown player, jersey, ball or court state is valid.
4. **Ball quality remains measurable.** Ball evidence is used only when sufficiently reliable.
5. **Jersey is P1 identity evidence.** Roster constraints may validate evidence but must not manufacture it.
6. **Basketball semantics are P2.** Possession/event classes and spatial event reasoning do not belong in the P1 exit criteria.
7. **Keep controlled footage.** Use established evaluation footage for comparisons before expanding datasets.
8. **Version detector promotions.** `model_v1.1.0.pt` must beat `model_v1.0.0.pt` before becoming canonical.
