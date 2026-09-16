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

CourtFlux is in active development. P0 and P1 are closed. P2 engineering/reconstruction is also closed: the remaining P2 work is scientific validation against reviewed human ground truth, evidence-driven calibration, baseline freeze and the final Jersey/Identity gate.

---

## Current project status

**Status as of 2026-09-16:**

```text
P0 — VISION FOUNDATION                         ✅ COMPLETE
P1 — PERCEPTION + GAME STATE                  ✅ COMPLETE
P2 — BASKETBALL UNDERSTANDING                 🧪 ENGINEERING COMPLETE / GT VALIDATION PENDING
P3 — STRUCTURED ANALYTICS + ACTIONS           ⏳ FUTURE
P4 — TACTICAL INTELLIGENCE                    ⏳ FUTURE
```

### P2 status in one sentence

> **Architecture/implementation closed; structural stability confirmed; real-video baseline confirmed; basketball correctness against human truth still pending.**

This means P2 does **not** need another reconstruction. The remaining sequence is:

```text
human GT
  ↓
measure prediction vs truth
  ↓
calibrate only from measured error
  ↓
freeze Jersey-OFF P2 baseline
  ↓
final Jersey / Identity gate
  ↓
P2 exit
```

Key milestones:

- PR #43 merged the accepted P1 / Identity-v2 baseline and deferred final Jersey/Identity correctness to the end of P2.
- PR #48 merged the reconstructed canonical P2 runtime into `main`.
- PR #49 closed the current P2 validation/lifecycle work around canonical metrics and event-state stability.
- The gameplay-boundary refactor separates gameplay presence from strict analysis quality.
- The frozen V1–V5 Jersey-OFF battery was rerun with the canonical `p2_metrics_v5` collector.
- A known-team Porto/Ovarense control established a stable real-video P2 baseline for team-aware diagnostics.
- PR #50 restored the selected SoccerNet PARSeq Jersey backend contract after a worker refactor had silently dropped per-job backend propagation.
- A post-PR #50 Jersey-ON run verified `requested_backend=parseq_soccernet`, `backend=parseq_soccernet` and `backend_contract_verified=true` while leaving P2 outputs unchanged.
- Basketball motor thresholds remain frozen until human GT provides evidence for calibration.

The confidence contract remains unchanged: missing, contradictory or weak evidence may remain `UNKNOWN`. CourtFlux must not force player, jersey, ball or basketball-event identity merely to increase coverage.

---

## Documentation status and source of truth

Use the following order when documents disagree:

1. `README.md` — current project/operational status.
2. `docs/p2_roadmap.md` — P2 technical contract, gates and detailed execution history.
3. `docs/system_quality_gate.md` — perception/ball quality evaluation contract where applicable.
4. `docs/p1_roadmap.md` — **HISTORICAL / SUPERSEDED** P1 execution record.
5. `docs/p1_jersey_exit_gate.md` — **HISTORICAL / SUPERSEDED** pre-merge P1 Jersey/Identity gate record.
6. `docs/p0_foundation_snapshot.md` — frozen historical P0 exit snapshot.

Historical P1/P2 notes are retained for experiment history and design rationale. Old branch names, phase labels and runtime instructions must not be interpreted as current instructions when they conflict with this README.

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
- gameplay-gap, frame-discontinuity and timestamp-discontinuity resets;
- per-event actor/team attribution diagnostics;
- canonical `p2_metrics_v5` observability;
- human-GT evaluation tooling.

A semantic detection such as `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `ball-in-basket` or `rim` is supporting evidence only. No single semantic class is event truth by itself.

### Frozen motor rule

Until human ground truth is evaluated, do **not** tune possession/shot/pass/turnover state-machine thresholds by intuition from one video. Architecture and observability fixes are allowed when they repair a broken contract; basketball calibration must be driven by the frozen evaluation set and reviewed GT.

---

## Confirmed P2 v5 multi-video baseline

The frozen V1–V5 battery was rerun after:

- the `gameplay_present` / `analysis_eligible` scope split;
- loose-ball lifecycle closure fixes;
- removal of the duplicate real-video-probe metrics collector;
- adoption of the single canonical `p2_metrics_v5` collector;
- reset diagnostic rename from `analysis_ineligible_gap` to `gameplay_absent_gap`.

These runs use `weights/model_v1.1.0.pt`, BoT-SORT, Jersey OFF and the historical neutral/unknown match config. Because that config has no home/away team evidence, the battery does **not** validate team attribution, pass classification or turnover classification.

| Video | Window | Metric frames | Engine frames | Limited gameplay | Possessed | UNKNOWN | Shots | Loose closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 `hILf-ySowIo` | 10–40 s | 247 | 720 | 473 | 26.94% | 38.75% | 6 | 100% |
| V2 `ULUpcQXsRII` | 0–30 s | 50 | 724 | 674 | 7.18% | 90.19% | 0 | 100% |
| V3 `Vr4105n-cPM` | 0–30 s | 540 | 888 | 348 | 13.85% | 72.97% | 2 | 100% |
| V4 `WMWTbS8J2ec` | 0–30 s | 1782 | 1785 | 3 | 19.83% | 69.58% | 1 | 96.67% |
| V5 `C-JjAlX2Dyc` | 10–40 s | 535 | 644 | 109 | 22.52% | 53.26% | 3 | 100% |

### What this battery confirms

- **Gameplay scope split works.** P2 preserves temporal state through `GAMEPLAY_LIMITED` frames without widening the strict metric denominator.
- **Loose-ball lifecycle is structurally stable.** Four clips close 100% of loose intervals; V4 closes 29/30 (96.67%) with one terminal interval at the clip boundary.
- **Shot de-duplication is stable.** `rapid_unknown_retrigger_count = 0` across all five runs.
- **P1 immutability holds.** `p1_state_not_mutated_rate = 100%` across all five runs.
- **Observability is internally consistent.** Probe and normal runtime use the same canonical v5 collector.
- **The wider engine scope does not create an uncontrolled event explosion.**

### What this battery does not confirm

- **Formal basketball accuracy.** Event counts and coverage are diagnostics, not precision/recall.
- **Team-aware correctness.** Neutral V1–V5 runs cannot prove pass, turnover or team attribution accuracy.
- **Shooter attribution correctness.** Coverage exists, but truth still requires reviewed GT.
- **V2 correctness.** V2 remains a difficult upstream case with high `UNKNOWN`; do not tune the motor specifically to make it emit more events without GT.

The architecture/reconstruction question is closed. The remaining P2 question is:

> **How accurate are these states and events against reviewed human truth?**

---

## Known-team Porto/Ovarense control

The canonical 0–60 s Porto/Ovarense control uses:

```text
model             weights/model_v1.1.0.pt
player conf       0.40
inference conf    0.03
ball conf         0.075
imgsz             960
device            mps
tracker           botsort.yaml
single-forward    ON
Smart Ball        OFF
court calibration OFF
```

The current known-team baseline is structurally stable across Jersey OFF and Jersey ON. With the same video and P2 configuration, the Jersey-ON PARSeq run retained the same P2 shape:

```text
metric eligible frames       953
engine eligible frames      1441
events                        84
shots                           8
passes                          3
turnovers                       5
rebounds                        4
loose-ball closure          100%
p1_state_not_mutated_rate   100%
```

This is evidence of runtime stability, **not basketball correctness**. Human GT is still required to decide whether each possession transition, pass, turnover, shot, rebound, actor and team attribution is correct.

---

## Gameplay presence vs analysis quality

The canonical runtime separates two questions:

1. **Is basketball gameplay present strongly enough to preserve temporal continuity?**
2. **Is this frame high-quality enough for strict spatial/metric analysis?**

Post-detector `CourtState` exposes:

```text
NON_GAMEPLAY
GAMEPLAY_LIMITED
GAMEPLAY_ANALYSIS
```

Conceptually:

```text
GAMEPLAY_LIMITED
tracking / Identity / BallState / P2 continuity   ✅
strict metric analysis                            ❌ / abstain

GAMEPLAY_ANALYSIS
tracking / Identity / BallState / P2 continuity   ✅
strict metric analysis                            ✅
```

`CourtState.gameplay_present` owns the high-recall gameplay/continuity decision. `CourtState.analysis_eligible` remains the conservative gate for strict metric-scoped analysis.

P2 metrics keep separate denominators for:

- `metric_eligible_frames` — strict analysis scope;
- `engine_eligible_frames` — frames where P2 may preserve/control basketball temporal state;
- `limited_gameplay_frames` — gameplay retained for continuity but excluded from strict metric analysis.

---

## Canonical runtime architecture

```text
Video Session
    ↓
Frame Loader
    ↓
Gameplay Filter
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

`services/pipeline.py` is the canonical state path. The old alternate analytics runtime and detection-only runtime split are retired.

`services.p2.P2Pipeline` remains only a compatibility wrapper around the canonical P2 stage; it must not run a second event engine.

`services/p2/metrics.py` owns the canonical `P2MetricsCollector`. Real-video probes and normal analyzer runs must reuse this collector.

Smart Ball remains **OFF by default**.

---

## P1 / Identity / Jersey boundary

Stable track/entity IDs are the operational identity source for P2.

During P2:

- `track_id` / stable entity identity is authoritative for possession and event attribution;
- `shirt_num` and `player_name` are optional confidence-aware labels;
- downstream basketball logic must work when Jersey/Identity is `UNKNOWN`;
- roster information is validator-only and must never manufacture a visual jersey number;
- `UNKNOWN` is preferred to a false identity lock;
- Identity v2 is the downstream temporal evidence/lock layer.

### Jersey backend contract restored in PR #50

A P2 worker refactor had accidentally removed the per-job Jersey backend field and subprocess environment propagation. That allowed the Lab to request a backend while the analyzer silently fell back to `best_temporal`.

PR #50 restored the contract:

```text
Lab request
  ↓
worker JobRequest.jersey_backend
  ↓
resolve_jersey_backend()
  ↓
BASKETBALL_JERSEY_BACKEND
  ↓
Jersey factory
  ↓
requested backend == actual backend, or fail closed
```

The canonical selected backend is now:

```text
parseq_soccernet
```

The runtime records `requested_backend` and `backend_contract_verified`, and the factory fails closed on an explicit requested/actual mismatch instead of silently running a different recognizer.

### Verified Jersey-ON PARSeq baseline

Run `20260916_135307_364002` verified the restored contract on the Porto/Ovarense 0–60 s clip:

```text
backend                       parseq_soccernet
checkpoint                    weights/parseq_soccernet.ckpt
requested backend             parseq_soccernet
backend contract verified     true
jersey attempts               438
accepted raw observations     153
identified tracks              23
jersey identification rate   13.2%
```

Compared with the accidental `best_temporal` run, the old `8/9/4/6` output collapse disappeared and Jersey coverage improved materially. This does **not** establish Jersey accuracy.

The current PARSeq baseline still shows substantial high-confidence `raw_not_in_roster` abstention and a concentration of `#10` observations. Because `#10` is valid for both teams, roster validation cannot determine whether those reads are true or false. This must be resolved with visual/human GT rather than threshold tuning.

### Final Jersey gate

The final Jersey/Identity gate remains part of P2 exit. Current priorities are:

1. Review accepted/locked Jersey observations against visual truth.
2. Measure false-lock and abstention behaviour by number/team/track.
3. Use localized number ROI and/or soft frame ranking only if GT shows that input localization/selection is the error source.
4. Keep Identity v2 as the single downstream temporal consensus/locking layer.
5. Keep roster information validator-only.
6. Freeze the Identity/Jersey configuration only after reviewed GT.

Jersey may be ON for targeted gate experiments, but the basketball motor must continue to be validated independently with Jersey-OFF baselines.

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

This is a historical P1 acceptance baseline, not a formal Jersey accuracy result.

---

## Current validation sequence

Completed:

1. ✅ Reconstruct and integrate one canonical P2 motor.
2. ✅ Separate gameplay presence from strict analysis quality.
3. ✅ Rerun the frozen V1–V5 battery with Jersey OFF.
4. ✅ Validate loose lifecycle closure, shot de-duplication, reset semantics and P1 immutability cross-video.
5. ✅ Collapse probe/runtime observability onto the single canonical `p2_metrics_v5` collector.
6. ✅ Run the Porto/Ovarense known-team control and establish a reproducible P2 baseline.
7. ✅ Restore and verify the selected `parseq_soccernet` Jersey runtime contract.
8. ✅ Confirm that targeted Jersey ON does not mutate the established P2 output shape.

Next:

9. **Human-label the frozen clips** for possession/control transitions, loose-ball lifecycle, passes, turnovers, shots, outcomes and actor/team attribution.
10. Compare predictions against reviewed GT and calculate state/event correctness metrics.
11. Calibrate only where the GT error analysis supports a motor change.
12. Rerun the frozen GT set and freeze the validated Jersey-OFF P2 baseline.
13. Execute the final Jersey/Identity GT gate using the restored PARSeq baseline.
14. Apply ROI/frame-selection changes only when supported by Jersey GT error analysis.
15. Freeze Identity/Jersey and promote P2 only when both basketball and Jersey exit gates are acceptable.

Canonical Porto/Ovarense probe:

```bash
git switch main
git pull --ff-only
python evaluation/p2_real_video_probe.py \
  --video "https://www.youtube.com/watch?v=OwiQgAMqyUs" \
  --start 0 \
  --end 60 \
  --model weights/model_v1.1.0.pt \
  --device mps \
  --no-jersey
```

For a targeted Jersey gate run, use the Lab/worker contract and verify that the resulting summary reports:

```text
backend=parseq_soccernet
requested_backend=parseq_soccernet
backend_contract_verified=true
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

When changing worker/backend code, restart the worker process before validating a new run; a long-lived Python worker keeps the previously imported module code in memory.

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

✅ **P1 complete; PR #43 merged**

✅ **P2 canonical runtime integrated into `main`**

✅ **P2 architecture/implementation closed**

✅ **P2 structural stability + multi-video/known-team baseline confirmed**

✅ **Selected SoccerNet PARSeq Jersey runtime contract restored and verified through PR #50**

🧪 **Current work: human GT → measured calibration → Jersey-OFF freeze → final Jersey/Identity gate**

Current priority:

> Measure basketball correctness against reviewed human truth. Do not tune the basketball motor from visual intuition alone. Keep Jersey evaluation evidence-driven and fail closed on backend/runtime mismatches.