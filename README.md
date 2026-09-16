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

CourtFlux is in active development. P0 and P1 are closed. The reconstructed P2 runtime has been integrated into `main`, while P2 validation/calibration continues on `p2-rebuilt`.

---

## Current project status

**Status as of 2026-09-16:**

```text
P0 — VISION FOUNDATION                         ✅ COMPLETE
P1 — PERCEPTION + GAME STATE                  ✅ COMPLETE
P2 — BASKETBALL UNDERSTANDING                 🧪 VALIDATION / CALIBRATION
P3 — STRUCTURED ANALYTICS + ACTIONS           ⏳ FUTURE
P4 — TACTICAL INTELLIGENCE                    ⏳ FUTURE
```

Key milestones:

- PR #43 merged the accepted P1 / Identity-v2 baseline and formally deferred final Jersey/Identity correctness to the end of P2.
- PR #48 merged the reconstructed P2 runtime into `main` while keeping `p2-rebuilt` available for continued P2 work.
- The basketball motor exists and is under validation; P2 is **not yet promoted as complete**.
- Current work is focused on gameplay-boundary robustness, repeatable real-video evaluation and human ground truth rather than adding more basketball event types.

The confidence contract remains unchanged: missing, contradictory or weak evidence may remain `UNKNOWN`. CourtFlux must not force player, jersey, ball or basketball-event identity merely to increase coverage.

---

## Documentation status and source of truth

Use the following order when documents disagree:

1. `README.md` — current project/operational status.
2. `docs/p2_roadmap.md` — current P2 technical contract, validation gates and next sequence.
3. `docs/system_quality_gate.md` — active perception/ball quality evaluation contract where applicable.
4. `docs/p1_roadmap.md` — **HISTORICAL / SUPERSEDED** P1 execution record.
5. `docs/p1_jersey_exit_gate.md` — **HISTORICAL / SUPERSEDED** pre-merge P1 Jersey/Identity gate record.
6. `docs/p0_foundation_snapshot.md` — frozen historical P0 exit snapshot.

The P1 documents are intentionally retained because they contain useful experiment history and design rationale. Their old phase labels and operational instructions must not be interpreted as current instructions.

---

## Current P2 validation contract

P2 converts canonical P1 state into confidence-aware basketball event primitives.

Current control states:

```text
UNKNOWN
POSSESSED
LOOSE
PASS_FLIGHT
SHOT_FLIGHT
```

Current event primitives:

```text
POSSESSION_ACQUIRED
POSSESSION_CHANGE
PASS_CANDIDATE
TURNOVER_CANDIDATE
LOOSE_BALL_STARTED
LOOSE_BALL_RECOVERED
LOOSE_BALL_ENDED
SHOT_ATTEMPT
REBOUND_CANDIDATE
```

P2 currently supports:

- explicit ball-control state;
- possession acquisition/change;
- loose-ball lifecycle, including same-handler recovery and explicit terminal closure;
- pass-flight evidence;
- pass/turnover attribution using stable tracks;
- shot-flight isolation from possession;
- shot attempt/outcome evidence with conservative `unknown` handling;
- robust shot re-arm/de-duplication;
- rebound candidate chaining;
- frame/timestamp discontinuity resets;
- per-event actor/team attribution diagnostics;
- human-GT evaluation tooling.

A semantic detection such as `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `ball-in-basket` or `rim` is supporting evidence only. No single semantic class is event truth by itself.

### Frozen motor rule

Until human ground truth is evaluated, do **not** tune possession/shot state-machine thresholds by intuition from one video. Architecture and observability fixes are allowed when they correct a broken contract, but basketball motor calibration must be driven by the frozen evaluation set and GT.

---

## Gameplay presence vs analysis quality

The canonical runtime now separates two questions that were previously conflated:

1. **Is basketball gameplay present strongly enough to preserve temporal continuity?**
2. **Is this frame high-quality enough for strict spatial/metric analysis?**

The cheap `GameplayFilter` is therefore a high-recall prefilter by default. Its HSV/motion/scene-cut heuristics remain useful diagnostics, but a weak image-only vote no longer automatically prevents detector/tracker/Identity inference on an otherwise valid frame.

Post-detector `CourtState` exposes the effective scope:

```text
NON_GAMEPLAY
GAMEPLAY_LIMITED
GAMEPLAY_ANALYSIS
```

Conceptually:

```text
GAMEPLAY_LIMITED
tracking / Identity / BallState / P2 continuity   ✅
strict wide-view metric analysis                  ❌ / abstain

GAMEPLAY_ANALYSIS
tracking / Identity / BallState / P2 continuity   ✅
strict metric analysis                            ✅
```

`CourtState.gameplay_present` owns the high-recall gameplay/continuity decision. `CourtState.analysis_eligible` remains the conservative gate for strict metric-scoped analysis.

P2 metrics therefore keep separate denominators for:

- `metric_eligible_frames` — strict analysis scope;
- `engine_eligible_frames` — frames where P2 may preserve/control basketball temporal state;
- `limited_gameplay_frames` — gameplay retained for continuity but excluded from strict metric analysis.

This separation is intentionally designed to prevent tight broadcast views, painted court areas or weak HSV evidence from creating artificial tracking/Identity/P2 gaps.

---

## Canonical runtime architecture

```text
Video Session
    ↓
Frame Loader
    ↓
Gameplay Filter
    └── high-recall prefilter evidence; fail-open by default
    ↓
Detector + Tracker + Identity v2
    ↓
CourtState
    ├── gameplay_present        → temporal continuity scope
    └── analysis_eligible       → strict metric scope
    ↓
PlayerState[] + BallState + CourtState            ← P1
    ↓
P2EngineV1
    ↓
control state + basketball event primitives       ← P2
    ↓
Structured statistics / observable actions        ← P3 later
    ↓
Tactical interpretation                           ← P4 later
```

`services/pipeline.py` is the canonical state path. The old alternate analytics runtime and the old detection-only runtime split have been retired.

`services.p2.P2Pipeline` remains only a compatibility wrapper around the canonical P2 stage; it must not run a second event engine.

Smart Ball remains **OFF by default**.

---

## P1 / Identity boundary

P1 was accepted and merged through PR #43. Stable track/entity IDs are the operational identity source for P2.

During P2:

- `track_id` / stable entity identity is authoritative for possession and event attribution;
- `shirt_num` and `player_name` are optional confidence-aware labels;
- downstream basketball logic must work when Jersey/Identity is `UNKNOWN`;
- roster information is validator-only and must never manufacture a visual jersey number;
- `UNKNOWN` is preferred to a false identity lock.

The final Jersey/Identity correctness gate remains deferred to the end of P2. This is a scope decision, not an accuracy claim.

### Deferred end-of-P2 Jersey experiment

The next experiment remains frozen as:

> **soft-ranking + localized ROI + Identity-v2-only temporal consensus**

Rules:

1. Legibility is a ranking signal, not a hard rejection gate.
2. Localized torso/number ROI proposals remain part of the experiment.
3. Identity v2 is the **only** temporal consensus/locking layer.
4. The recognizer emits one raw visual observation or `UNKNOWN`.
5. Roster information may validate/reject evidence but never select or rewrite a number.
6. Formal Jersey accuracy requires reviewed ground truth.

For the current basketball-validation battery, **Jersey stays OFF** so P2 correctness is evaluated independently of unresolved player-number recognition.

---

## Historical P1 baseline

The accepted integrated P1 reference was the Porto/Ovarense `0–60 s` run `20260914_140057_772692`:

```text
model               weights/model_v1.1.0.pt
segment             0–60 s
player conf         0.40
inference conf      0.03
ball conf           0.075
imgsz               960
device              mps
tracker             botsort.yaml
single-forward      ON
Smart Ball          OFF
Jersey              ON
jersey interval     24 frames
court calibration   OFF

analysis FPS                     18.60
detector                         44.81 ms/frame
total pipeline                   49.43 ms/frame
ball effective trajectory        89.51%
team classification coverage     99.5%
```

This is a **historical P1 acceptance baseline**, not the current P2 validation configuration. In particular, the retired detection-only runtime is no longer part of the canonical analyzer/Lab contract, and current P2 basketball validation runs with Jersey OFF.

Coverage/acceptance diagnostics are not formal accuracy measurements.

---

## Current validation sequence

The immediate P2 sequence is:

1. Re-run the same frozen real-video battery with the current gameplay-presence split and Jersey OFF.
2. Compare `metric_eligible_frames`, `engine_eligible_frames` and `limited_gameplay_frames` instead of treating all three concepts as one gate.
3. Verify continuity/reset behavior, possession/control states, loose-ball closure, shot de-duplication and per-event attribution.
4. Keep the basketball motor unchanged unless a contract/regression bug is demonstrated.
5. Human-label the frozen clips and evaluate P2 against GT.
6. Calibrate only from the frozen multi-video GT set.
7. Freeze the validated Jersey-OFF P2 baseline.
8. Run the deferred end-of-P2 Jersey experiment.
9. Promote P2 only after the basketball GT gates and deferred Jersey gate are acceptable.

The canonical Porto/Ovarense probe remains:

```bash
git switch p2-rebuilt
git pull
python evaluation/p2_real_video_probe.py \
  --video "https://www.youtube.com/watch?v=OwiQgAMqyUs" \
  --start 0 \
  --end 60 \
  --model weights/model_v1.1.0.pt \
  --device mps \
  --no-jersey
```

---

## Local development

### Python environment

```bash
git clone <repository-url>
cd courtflux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Worker

```bash
source .venv/bin/activate
export WORKER_HOST="127.0.0.1"
export WORKER_PORT="8787"
python3 -m services.worker_api
```

### Web Lab

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://localhost:3000/lab
```

Generated local evaluation/worker artifacts are disposable once their conclusions are recorded durably in the README or current roadmap.

---

## Summary

✅ **P0 complete**

✅ **P1 complete; PR #43 merged; final Jersey correctness deferred**

✅ **P2 reconstructed runtime integrated into `main` through PR #48**

🧪 **Current work: P2 validation/calibration on `p2-rebuilt`**

Current priority:

> Validate gameplay continuity and basketball event quality across the frozen video battery with Jersey OFF, then move to human GT. Do not use obsolete P1 phase documents as current operational instructions.
