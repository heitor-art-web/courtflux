# CourtFlux — Basketball Intelligence from Video

**CourtFlux** turns ordinary basketball game video into structured evidence, reviewable moments and, over time, actionable development intelligence.

> **Product loop:** Video → Understanding → Insight → Evidence → Improvement
>
> Every important insight should remain traceable to the source video. Weak, missing or contradictory evidence may remain `UNKNOWN`; CourtFlux should not manufacture certainty to increase coverage.

---

## Current project status

**Last updated: 2026-09-17**

| Phase | Scope | Status |
| --- | --- | --- |
| **P0** | Vision Foundation | ✅ Complete |
| **P1** | Perception + Tracking | ✅ Complete |
| **P2** | Basketball Understanding + Game-Level Player Identity | 🧪 Engineering core complete; human GT + game-entity validation pending |
| **P3** | Structured Analytics + Actions | ⏳ Future |
| **P4** | Tactical Intelligence | ⏳ Future |
| **P4.5** | Persistent Player Identity | ⏳ Future |
| **P5** | Mobile App | ⏳ Future |
| **P6** | Continuous Development / AI Coach | ⏳ Future |

### Current priority

P2 does **not** need another architectural reconstruction. The current work is validation:

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
        ↓
P3 — Structured Analytics + Actions
```

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

CourtFlux now separates three different concepts explicitly:

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

Jersey recognition remains useful R&D and supporting evidence, but it is **not a P1 or P2 exit gate**. `shirt_num` and `player_name` may remain `UNKNOWN` without blocking basketball understanding.

### P4.5

Persistent real-world identity is planned for **P4.5 — Persistent Player Identity**. That phase may combine roster context, jersey evidence, appearance/ReID, previous games and human confirmation to resolve:

```text
game_1.game_player_03
        ↓
     João Silva
        ↑
game_2.game_player_07
```

The detailed identity boundary is recorded in [`docs/p1_p2_identity_boundary.md`](docs/p1_p2_identity_boundary.md).

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
P3 — Structured Analytics + Actions
        ↓
P4 — Tactical Intelligence
        ↓
P4.5 — Persistent Player Identity
        ↓
P5 — Mobile App
        ↓
P6 — Continuous Development / AI Coach
```

### P3 — Structured Analytics + Actions

Turn P2 events into a usable review product: game/player summaries, timelines, automatically organized clips, filters, evidence-linked insights and review/action workflows.

**Exit idea:** a coach or player should no longer need to scrub manually through 60–90 minutes of video to find the possessions that deserve attention.

### P4 — Tactical Intelligence

Move from isolated events to team and possession context: spacing, transition, screening/P&R actions, defensive behaviour, possession patterns and recurring team tendencies.

### P4.5 — Persistent Player Identity

Resolve game-level player entities to persistent roster identities across games. Human confirmation is an acceptable part of the system; perfect automatic Jersey OCR is not required.

### P5 — Mobile App

Make CourtFlux part of the daily coach/player workflow: game feed, clip review, notifications, coach-to-player review, sharing, saved moments and eventually mobile upload/capture flows.

### P6 — Continuous Development / AI Coach

Use evidence from multiple real games to build player/team models, detect recurring patterns and meaningful change, suggest review/development priorities and measure progress over time.

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

Full phase detail lives in [`docs/current_product_roadmap.md`](docs/current_product_roadmap.md).

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
Structured analytics + review actions              ← P3 future
    ↓
Tactical interpretation                            ← P4 future
```

`services/pipeline.py` is the canonical state path. The old alternate analytics runtime and detection-only runtime split are retired. Smart Ball remains **OFF by default**.

---

## Confidence and evidence rules

CourtFlux follows a conservative evidence contract:

- `UNKNOWN` is a valid output;
- weak evidence must not be promoted into confident truth;
- semantic detections are supporting evidence, not event truth by themselves;
- roster information must not manufacture a jersey number;
- a stable track does not imply a known real-world player;
- event counts and coverage are diagnostics until validated against human truth;
- public/product claims must distinguish structural capability from measured accuracy.

---

## Documentation and historical reference

The project keeps historical implementation and experiment records intentionally. They remain useful engineering references even when a later roadmap decision changes a phase boundary.

Use this order when documents disagree about **current product scope or phase ownership**:

1. [`README.md`](README.md) — current project status and viewer-facing summary.
2. [`docs/current_product_roadmap.md`](docs/current_product_roadmap.md) — current P0→P6 product roadmap.
3. [`docs/p1_p2_identity_boundary.md`](docs/p1_p2_identity_boundary.md) — current identity/phase boundary.
4. [`docs/p2_roadmap.md`](docs/p2_roadmap.md) — detailed P2 technical contract and execution history.
5. [`docs/system_quality_gate.md`](docs/system_quality_gate.md) — perception/ball quality evaluation contract where applicable.
6. [`docs/p1_roadmap.md`](docs/p1_roadmap.md) — historical P1 execution record.
7. [`docs/p1_jersey_exit_gate.md`](docs/p1_jersey_exit_gate.md) — historical pre-merge Jersey/Identity gate record.
8. [`docs/p0_foundation_snapshot.md`](docs/p0_foundation_snapshot.md) — frozen historical P0 exit snapshot.

Older documents are **not deleted or rewritten merely because the roadmap evolved**. When old wording conflicts with current phase ownership, the current roadmap/boundary documents above take precedence while the old material remains available as historical reference.

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
- ⏳ P3 next product phase after P2 validation/freeze
- ⏳ P4, P4.5, P5 and P6 are future roadmap phases

**Current engineering focus:** validate basketball correctness and game-level player attribution against reviewed truth, preserve conservative uncertainty, and avoid making Jersey/name recognition a blocker for P3.