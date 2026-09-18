# CourtFlux — Basketball Intelligence from Video

**CourtFlux** turns ordinary basketball game video into structured evidence, reviewable moments and, over time, actionable development intelligence.

> **Product loop:** Video → Understanding → Insight → Evidence → Improvement
>
> Every important insight should remain traceable to the source video. Weak, missing or contradictory evidence may remain `UNKNOWN`; CourtFlux should not manufacture certainty to increase coverage.

---

## Current project status

**Last updated: 2026-09-18**

| Phase | Scope | Status |
| --- | --- | --- |
| **P0** | Vision Foundation | ✅ Complete |
| **P1** | Perception + Tracking | ✅ Complete |
| **P2** | Basketball Understanding + Game-Level Player Identity | 🧪 Engineering core complete; human GT + game-entity validation pending |
| **P3** | Structured Analytics + Review | 🛠 Foundation may proceed concurrently; product exit depends on validated P2 |
| **P4** | Tactical Intelligence | ⏳ Future |
| **P5** | Persistent Player Identity / Jersey Evidence | ⏳ Future |
| **P6** | Mobile App | ⏳ Future |
| **P7** | Continuous Development / AI Coach | ⏳ Future |

### Current priority

P2 does **not** need another architectural reconstruction. Its active work is validation:

```text
reviewed human ground truth
        ↓
measure basketball-event correctness
        ↓
validate game-level player continuity / attribution
        ↓
calibrate only from measured error
        ↓
freeze the validated P2 baseline
```

P3 foundation work may proceed concurrently against the stable P2→P3 contract:

```text
P2 validation                          P3 foundation
     │                                      │
human GT                              Game Record
calibration                           persistence / provenance
basketball correctness               physical clips
player attribution                   review / corrections
     │                               fixture-driven analytics
     │                                      │
     └────────────── integration ───────────┘
                         ↓
                validated P2 baseline
                         ↓
                   P3 product freeze
```

The concurrency rule is strict: **P3 may advance structurally, while product decisions and derived outputs that depend on final P2 accuracy remain configurable, versioned and reprocessable until P2 passes its exit gate.**

P3 must consume the P2 contract rather than import P2 engine internals, recreate possessions independently or tune P2 thresholds from P3 product behaviour.

The P2 basketball motor should not be tuned from one surprising clip or visual intuition alone. Calibration must be driven by frozen evaluation footage and reviewed ground truth.

---

## What P2 currently understands

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

The implemented P2 runtime includes ball-control state, possession acquisition/change, loose-ball lifecycle, pass/turnover candidates, shot attempt/outcome evidence, rebound candidates, temporal reset handling, actor/team attribution diagnostics and human-GT evaluation tooling.

These are **structured basketball primitives**, not a claim of perfect basketball truth. Formal accuracy remains pending reviewed human ground-truth validation.

---

## Player identity boundary

CourtFlux separates three different concepts explicitly:

```text
Track ID
  = technical visual tracking identity

Game Player Entity
  = the same player consistently represented inside one game

Persistent Player Identity
  = the real roster/person identity across games
```

### P1

P1 owns perception and tracking. A `track_id` is a technical identity, not a real-world player identity.

### P2

P2 owns basketball understanding and must attribute events to a sufficiently stable **game-level player entity** so P3 can aggregate possessions, clips and analytics by player inside one game.

P2 must work without knowing that a player is, for example, `João Silva #7`.

### Jersey recognition

Jersey recognition remains useful R&D and supporting evidence, but it is **not a P1 or P2 exit gate**. `shirt_num` and `player_name` may remain `UNKNOWN` without blocking basketball understanding, P3 review or P4 tactical intelligence.

### P5

Persistent real-world identity is planned for **P5 — Persistent Player Identity**. Jersey recognition is an important evidence source in that phase, but is not the identity system by itself.

P5 combines roster context, jersey evidence, appearance/ReID, previous games and human confirmation to resolve:

```text
game_1.game_player_03
        ↓
     João Silva
        ↑
game_2.game_player_07
```

The detailed P1/P2 identity boundary is recorded in [`docs/p1_p2_identity_boundary.md`](docs/p1_p2_identity_boundary.md). Historical references to `P4.5 — Persistent Player Identity` predate the current roadmap numbering.

---

## Product roadmap

The current product roadmap is:

```text
P0 — Vision Foundation
        ↓
P1 — Perception + Tracking
        ↓
P2 — Basketball Understanding + Game-Level Player Identity
        ↓
P3 — Structured Analytics + Review
        ↓
P4 — Tactical Intelligence
        ↓
P5 — Persistent Player Identity / Jersey Evidence
        ↓
P6 — Mobile App
        ↓
P7 — Continuous Development / AI Coach
```

### P3 — Structured Analytics + Review

Turn P2 events into a durable, reopenable game review product: possessions, game/player summaries, physical evidence clips, Things to Review, timelines, filters, conservative insights, human corrections and simple review actions.

P3 foundation work may be developed concurrently with final P2 validation, provided that P2-dependent calculations remain versioned/reprocessable and the integration boundary remains the stable P2→P3 event contract.

**Exit idea:** a coach or player should no longer need to scrub manually through 60–90 minutes of video to find the moments that deserve attention.

Detailed execution lives in [`docs/p3_roadmap.md`](docs/p3_roadmap.md).

### P4 — Tactical Intelligence

Move from isolated events to temporal relationships between players, space and possession structure: validated spatial state, tactical primitives, transition, screening/P&R and other offensive actions, multi-label tactical segmentation, game/team tendencies, defensive intelligence and evidence-linked radar review.

P4 is offense-first. Defensive intelligence follows the validated spatial/offensive foundation. Tactical conclusions must preserve `UNKNOWN`/abstention when geometry or context is insufficient.

Detailed execution lives in [`docs/p4_roadmap.md`](docs/p4_roadmap.md).

### P5 — Persistent Player Identity

Resolve game-level player entities to persistent real-world identities across games. Jersey OCR is one probabilistic evidence source alongside roster context, appearance/ReID, previous games and human confirmation.

A short confirmation flow is acceptable; false automatic identity is not. Jersey number is mutable membership metadata, not permanent person identity. Existing Identity v2, Jersey GT tooling and historical R&D conclusions are foundations to reuse rather than restart.

Detailed execution lives in [`docs/p5_roadmap.md`](docs/p5_roadmap.md).

### P6 — Mobile App

Make CourtFlux part of the daily coach/player workflow: game feed, clip review, notifications, coach-to-player review, sharing, saved moments and eventually mobile upload/capture flows.

### P7 — Continuous Development / AI Coach

Use evidence from multiple real games to build player/team models, detect recurring patterns and meaningful change, suggest review/development priorities and measure progress over time.

The AI Coach should reason over structured CourtFlux evidence rather than act as a generic chatbot over raw video.

The intended long-term loop is:

```text
GAME
  ↓
ANALYSIS
  ↓
PATTERN
  ↓
FOCUS
  ↓
REVIEW / TRAIN
  ↓
NEXT GAME
  ↓
MEASURE AGAIN
```

Product-level phase definitions live in [`docs/current_product_roadmap.md`](docs/current_product_roadmap.md). Detailed execution contracts live in [`docs/p3_roadmap.md`](docs/p3_roadmap.md), [`docs/p4_roadmap.md`](docs/p4_roadmap.md) and [`docs/p5_roadmap.md`](docs/p5_roadmap.md).

---

## Canonical runtime architecture

```text
Video Session
    ↓
Frame Loader
    ↓
Gameplay Filter
    ↓
Detector + Tracker + team / optional identity evidence
    ↓
PlayerState[] + BallState + CourtState             ← P1
    ↓
P2EngineV1
    ↓
control state + basketball event primitives        ← P2
    ↓
Structured analytics + evidence-linked review      ← P3
    ↓
Tactical interpretation                            ← P4
    ↓
Persistent real-world player identity               ← P5
```

`services/pipeline.py` is the canonical state path. The old alternate analytics runtime and detection-only runtime split are retired. Smart Ball remains **OFF by default**.

---

## Confidence and evidence rules

CourtFlux follows a conservative evidence contract:

- `UNKNOWN` is a valid output;
- weak evidence must not be promoted into confident truth;
- semantic detections are supporting evidence, not event truth by themselves;
- roster information must not manufacture a jersey number or player identity;
- a stable track does not imply a known real-world player;
- event counts and coverage are diagnostics until validated against human truth;
- P3 outputs derived before P2 freeze must remain reproducible/reprocessable against the final validated P2 baseline;
- tactical output must expose when geometry/context is not sufficient for analysis;
- public/product claims must distinguish structural capability from measured accuracy.

---

## Documentation and historical reference

The project keeps historical implementation and experiment records intentionally. They remain useful engineering references even when a later roadmap decision changes a phase boundary or phase number.

Use this order when documents disagree about **current product scope or phase ownership**:

1. [`README.md`](README.md) — current project status, phase order and active execution policy.
2. [`docs/current_product_roadmap.md`](docs/current_product_roadmap.md) — current P0→P7 product roadmap and phase boundaries.
3. [`docs/p3_roadmap.md`](docs/p3_roadmap.md) — detailed P3 product/engineering contract.
4. [`docs/p4_roadmap.md`](docs/p4_roadmap.md) — detailed P4 tactical contract.
5. [`docs/p5_roadmap.md`](docs/p5_roadmap.md) — detailed Persistent Player Identity contract and Jersey/ReID strategy.
6. [`docs/p1_p2_identity_boundary.md`](docs/p1_p2_identity_boundary.md) — current P1/P2 identity boundary.
7. [`docs/p2_roadmap.md`](docs/p2_roadmap.md) — detailed P2 technical contract and execution history.
8. [`docs/system_quality_gate.md`](docs/system_quality_gate.md) — perception/ball quality evaluation contract where applicable.
9. [`docs/p1_roadmap.md`](docs/p1_roadmap.md) — historical P1 execution record.
10. [`docs/p1_jersey_exit_gate.md`](docs/p1_jersey_exit_gate.md) — historical pre-merge Jersey/Identity gate record.
11. [`docs/jersey_recon_rnd.md`](docs/jersey_recon_rnd.md) — frozen Jersey v1–v5 R&D conclusions.
12. [`docs/p0_foundation_snapshot.md`](docs/p0_foundation_snapshot.md) — frozen historical P0 exit snapshot.

Older documents are **not deleted or rewritten merely because the roadmap evolved**. Historical experiment results stay intact. Current roadmap/boundary documents may be updated when phase ownership changes.

---

## Development

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

When worker/backend code changes, restart the worker before validating a new run; a long-lived Python worker keeps previously imported module code in memory.

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

---

## Status summary

- ✅ P0 complete
- ✅ P1 complete
- 🧪 P2 engineering core integrated; reviewed human GT + game-level entity validation pending
- 🛠 P3 structural foundation may proceed concurrently on an isolated branch; P2-dependent semantics remain versioned/reprocessable until P2 exit
- ⏳ P4 Tactical Intelligence is future
- ⏳ P5 Persistent Player Identity / Jersey Evidence is future
- ⏳ P6 Mobile App is future
- ⏳ P7 Continuous Development / AI Coach is future

**Current engineering focus:** validate P2 basketball correctness and game-level player attribution against reviewed truth while allowing isolated P3 foundation work that consumes the stable P2 contract without treating unvalidated P2 output as final product truth.