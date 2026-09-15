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

CourtFlux is in active development. The P0 vision/evaluation foundation and the P1 perception/game-state foundation are accepted. **P2 Basketball Understanding is now the active phase.**

The remaining Jersey/Identity correctness gate has been **formally deferred to the end of P2**. This is a scope decision, not a declaration that current jersey identity is correct. During P2, stable track/entity IDs are the operational identity source; jersey number and player name remain optional, confidence-aware semantic labels and may stay `UNKNOWN`.

---

## Current phase

```text
P0 — VISION FOUNDATION                         ✅ COMPLETE
P1 — PERCEPTION + GAME STATE                  ✅ COMPLETE*
P2 — BASKETBALL UNDERSTANDING                 ← CURRENT PHASE
P3 — STRUCTURED ANALYTICS + ACTIONS
P4 — TACTICAL INTELLIGENCE

* Jersey/Identity correctness gate deferred to end of P2.
```

The confidence contract remains unchanged: CourtFlux must tolerate missing/`UNKNOWN` state and must never force player, jersey or ball identity when evidence is weak.

### P1 exit decision

P1 is accepted with the following explicit exception:

```text
canonical P1 runtime/state contract     ✅
player / referee / CourtState visual QA ✅
ball runtime + continuity               ✅
team identity                           ✅ closed for P1
runtime performance                     ✅
Jersey / player semantic identity       ⏭ deferred to end of P2
```

The deferment does **not** promote any current jersey recognizer and does not create an accuracy claim. Formal accuracy/precision/recall claims still require appropriate ground truth.

During P2:

- `track_id` / stable entity identity is authoritative for possession and event attribution;
- `shirt_num` / `player_name` may enrich output only when supported;
- downstream logic must not assume jersey identity is known;
- `UNKNOWN` remains preferable to a confidently wrong player identity;
- roster information remains validator-only and may never decode or substitute a number;
- Jersey remains available for diagnostics and evidence collection, but it is not a P2 entry blocker.

See:

- `docs/p1_roadmap.md` — P1 execution history and promotion rules;
- `docs/p1_jersey_exit_gate.md` — Jersey/Identity investigation and gate history;
- `docs/p0_foundation_snapshot.md` — frozen P0 exit snapshot and P1 boundary;
- `docs/system_quality_gate.md` — ball-quality evaluation contract.

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

### P1 — Perception + Game State — COMPLETE WITH DEFERRED JERSEY GATE

P1 answers:

> Who/what is visible, where is it, and what state evidence do we trust now?

P1 owns:

- player detection/tracking quality;
- team evidence;
- temporal ball tracking;
- ball trajectory consistency;
- versioned detector validation;
- `BallState`;
- `PlayerState` + `CourtState`;
- confidence, provenance and `UNKNOWN` semantics;
- the Identity v2 temporal/evidence infrastructure used by jersey experiments.

Jersey identification remains part of the long-term identity contract, but its final correctness gate is deferred to the end of P2 so that P2 can proceed using stable track/entity IDs without treating experimental jersey labels as truth.

### P2 — Basketball Understanding — CURRENT

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

P2 decisions should be keyed to stable tracks/entities. A jersey/player-name label may be attached when trusted, but P2 correctness must not depend on it.

### P3 — Structured Analytics + Actions

P3 converts trusted P2 events and P1 state into measurable basketball facts, observable actions and reusable analytics structures, including:

- possession counts and duration;
- FGA / FGM / shooting percentages when supported;
- passes, turnovers, rebounds and touches;
- screens, cuts, drives, handoffs, relocations and dribble sequences;
- shot location, defender distance, spacing, zones and movement measurements;
- shot charts, heatmaps, pass networks and transition maps.

P3 records observable facts. It should not infer coaching intent or full tactical systems.

### P4 — Tactical Intelligence

P4 combines P3 actions, spatial measurements and temporal context to infer higher-level tactical patterns such as:

- pick-and-roll / pick-and-pop;
- Spain PnR, double drag, Horns, Floppy and related structures;
- drop, switch, hedge, blitz, ICE and other defensive coverage;
- help responsibilities and rotations;
- transition principles;
- contextual shot quality;
- advantage creation and tactical tendencies.

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
    ├── optional jersey-number evidence
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

Identity boundary during P2:

```text
stable track/entity ID
        ↓
P2 event attribution

jersey recognizer
        ↓
optional semantic label
        ↓
#number / player name when supported
```

---

# P1 accepted runtime baseline

The accepted integrated P1 reference is the Porto/Ovarense `0–60 s` run `20260914_140057_772692`.

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

Accepted runtime reference:

```text
analysis FPS                     18.60
detector                         44.81 ms/frame
total pipeline                   49.43 ms/frame
ball effective trajectory        89.51%
team classification coverage     99.5%
```

CourtState, PlayerState, referee handling, ball continuity and runtime performance are accepted for P1. Team classification is also closed for P1 despite known visual mistakes. Coverage diagnostics are not accuracy claims.

Smart Ball remains **OFF** by default. Jersey remains **ON** as experimental evidence collection, but final Jersey/Identity correctness is deferred.

---

# Jersey / Identity — deferred exit gate

## What was learned

The P1 Jersey investigation established several useful constraints without producing a production-ready recognizer.

- Legacy digit-head and temporal checkpoints showed severe output-domain collapse and were not suitable for arbitrary jersey numbers.
- A large jersey-finetuned SmolVLM2 feasibility probe demonstrated that the broadcast crops do contain recoverable multi-digit signal, including numbers outside the old `{4,6,8,9}` collapse, but it was far too slow and still produced confident hallucinations. This demonstrated feasibility, not production accuracy.
- `microsoft/trocr-small-printed` was lightweight enough but failed visually and operationally on this domain.
- SoccerNet-finetuned PARSeq was materially better than TrOCR but remained visually unreliable on the current CourtFlux observations.
- A literature-guided PARSeq experiment using a Hockey legibility classifier, multiple localized ROI proposals and an additional internal temporal consensus became too conservative: the integrated identification rate fell to **8.5%**, with the dominant failures becoming `no_valid_number` and insufficient visual consensus. This is a coverage/acceptance result, not an accuracy measurement.

The key architectural lesson is that **localization appears useful, while a hard legibility gate and duplicate temporal consensus are counterproductive in the current CourtFlux pipeline**.

## Next Jersey experiment — end of P2

The next experiment is explicitly frozen as:

> **soft-ranking + localized ROI + Identity-v2-only temporal consensus**

The intended pipeline is:

```text
track crop buffer
    ↓
soft frame ranking
(crop quality + legibility score; no hard legibility rejection)
    ↓
best available frame
    ↓
localized ROI proposals
(torso / number-focused crops)
    ↓
specialized jersey recognizer
    ↓
one raw visual observation or UNKNOWN
    ↓
roster validator-only
    ↓
Identity v2 temporal evidence / consensus / reversible lock
```

Important rules for that experiment:

1. **Legibility is a ranking signal, not a hard gate.** If every observation is weak, keep the best available crop and allow the recognizer/Identity v2 to abstain naturally.
2. **Localization stays.** The experiments suggest that number-focused ROIs are worth retaining and improving.
3. **No duplicate temporal consensus.** The recognizer should emit one visual observation at a time; Identity v2 is the only temporal consensus/locking layer.
4. **Roster remains validator-only.** It may reject a visually produced number but never select or rewrite one.
5. **UNKNOWN is preferred to a false lock.**
6. Do not claim formal jersey accuracy without appropriate GT.

Only after this experiment should CourtFlux decide whether to continue with a SoccerNet/Hockey PARSeq family model, add a lightweight pose-guided shoulders-to-hips ROI, train a dedicated jersey-number region detector, or move to another small recognizer.

This experiment is intentionally deferred until the **end of P2** unless jersey identity becomes a concrete blocker for a P2 feature.

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

Planned P2 capabilities:

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

A single semantic detection is never event truth by itself. P2 combines multiple evidence sources and preserves uncertainty.

---

## Diagnostics and promotion rules

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

Promotion principles:

- measure the current baseline first;
- benchmark one challenger at a time;
- keep only measured improvements;
- prefer `UNKNOWN` over forced state;
- keep roster information out of visual decoding;
- distinguish coverage/acceptance/locks from formal accuracy;
- preserve runtime cost and failure modes in diagnostics.

---

## Running CourtFlux locally

### Setup

```bash
git clone <repository-url>
cd courtflux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Worker + web Lab

Start the analysis worker:

```bash
source .venv/bin/activate
export WORKER_HOST="127.0.0.1"
export WORKER_PORT="8787"
python3 -m services.worker_api
```

Start the web Lab in another terminal:

```bash
cd web
npm install
npm run dev
```

Then open:

```text
http://localhost:3000/lab
```

Current accepted perception defaults:

```text
model                weights/model_v1.1.0.pt
player confidence    0.40
inference confidence 0.03
ball confidence      0.075
imgsz                960
tracker              botsort.yaml
single-forward       ON
Smart Ball           OFF
Jersey               ON
jersey interval      24 frames
```

Generated local run/worker artifacts under `evaluation/runs/` and `evaluation/worker_jobs/` are disposable once a decision has been recorded durably in docs/README.

---

## Status

✅ **P0 Vision Foundation Complete**

✅ **P1 Perception + Game State Complete — Jersey/Identity gate deferred**

🚧 **Current phase: P2 — Basketball Understanding**

Current priority:

> Build confidence-aware possession and basketball event primitives from stable P1 track/state contracts. Keep jersey/player semantic identity optional during P2, then return to the deferred identity gate at the end of P2 with **soft-ranking + localized ROI + Identity-v2-only temporal consensus**.
