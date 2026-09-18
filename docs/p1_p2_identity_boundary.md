# P1 / P2 Identity Boundary

This document records the current identity boundary between perception, basketball understanding and persistent real-world player identity.

It supersedes older P1/P2 wording where Jersey recognition or real-world player naming was treated as a P1/P2 exit gate.

For the consolidated product roadmap from P0 through P7, see `docs/current_product_roadmap.md`. Historical implementation documents remain preserved as engineering reference and are not rewritten merely because phase ownership changed.

---

## P1 — Perception + Tracking

P1 answers:

> What players, teams, ball and court state can CourtFlux perceive and track through the video?

P1 owns:

- player / ball / court perception;
- player tracking and temporal continuity;
- team evidence;
- `PlayerState`, `BallState` and `CourtState`;
- confidence, provenance and conservative `UNKNOWN` behaviour.

P1 does **not** require:

- a real-world player name;
- a roster match;
- a reliable jersey number;
- cross-game player identity.

Raw `track_id` is a technical tracking identity only. It must not be presented as equivalent to a real player identity.

---

## P2 — Basketball Understanding + Game-Level Player Identity

P2 answers:

> What happened in basketball terms, and which game-level player entity was involved?

P2 owns:

- ball-control state;
- possession acquisition/change;
- loose-ball lifecycle;
- pass / turnover candidates;
- shot attempts and outcome evidence;
- rebound candidates;
- event attribution to a stable game-level player entity;
- reviewed human-GT validation of basketball correctness and game-level attribution.

The required identity contract is **game-local**, not real-world identity.

Conceptually:

```text
raw tracks
   ↓
game_player_01
   ↓
P2 basketball events
```

Multiple raw tracks may belong to the same game-level player entity when continuity evidence supports the merge.

P2 must be able to produce useful output such as:

```text
game_player_03
  → POSSESSION_ACQUIRED
  → PASS_CANDIDATE
  → later SHOT_ATTEMPT
```

without knowing that `game_player_03` is João Silva or that the player wears #7.

---

## Jersey and real-world identity

Jersey recognition remains useful identity evidence and important historical R&D, but it is **not a P1 or P2 exit gate**.

`shirt_num` and `player_name` are optional labels. They may remain `UNKNOWN` without blocking P2 basketball understanding, P3 review or P4 tactical intelligence.

The identity distinction is:

```text
Track ID
  = technical visual tracking identity

Game Player Entity
  = stable identity for the same player inside one game

Persistent Player Identity
  = canonical real-world player identity across games
```

Persistent real-world identity belongs to **P5 — Persistent Player Identity**.

P5 resolves game-level entities to persistent players using multiple evidence sources rather than treating Jersey OCR as the identity system:

```text
roster / membership context
jersey evidence
appearance / ReID
team context
previous games
human confirmation
        ↓
Persistent Player Identity
```

Detailed P5 execution is defined in `docs/p5_roadmap.md`.

---

## Identity safety principles

The following principles apply across the roadmap:

- `UNKNOWN` is preferable to a false identity;
- roster information may validate evidence but must not manufacture a visual jersey prediction;
- Jersey number is mutable team/membership metadata, not permanent person identity;
- human confirmation is legitimate identity evidence and must remain auditable;
- automatic identity must be confidence-aware, evidence-linked and reversible;
- a previously resolved identity may return to `UNKNOWN` or conflict when stronger contradictory evidence appears;
- downstream analytics must retain provenance so an identity correction can be propagated/reprocessed safely.

Historical Identity v2/Jersey work remains reusable infrastructure for P5, especially temporal evidence, reversible locks, abstention and GT tooling.

---

## Exit criteria

### P1 exit

P1 is complete when perception and tracking provide stable, confidence-aware state suitable for basketball reasoning. Real-world player identity is not required.

### P2 exit

P2 is complete when basketball events are validated against reviewed human ground truth **and** event attribution is stable enough at the game-player level to support P3 analytics.

P2 does not require automated knowledge of the player's name or persistent cross-game identity.

### P5 exit

P5 owns the later correctness gate for persistent real-world identity across games, including false-merge/false-split behaviour, abstention quality, human correction and evidence provenance.

---

## Roadmap consequence

```text
P0 — Vision Foundation
P1 — Perception + Tracking
P2 — Basketball Understanding + Game-Level Player Identity
P3 — Structured Analytics + Review
P4 — Tactical Intelligence
P5 — Persistent Player Identity / Jersey Evidence
P6 — Mobile App
P7 — Continuous Development / AI Coach
```

The immediate P2 priority is therefore not perfect Jersey OCR. It is reliable basketball correctness and game-level player continuity so P3 can aggregate events and clips per player inside one game.

P3 structural foundation may proceed concurrently with P2 validation as long as P3 consumes a stable versioned P2 contract and keeps P2-accuracy-dependent outputs configurable/reprocessable until P2 passes its exit gate.

---

## Historical-reference rule

Older P1/P2 documents remain valid as records of experiments, baselines and implementation decisions. Their historical Jersey/Identity exit-gate wording or older phase numbering must not be read as the current product phase boundary.

Current phase ownership is defined by:

1. `README.md`;
2. `docs/current_product_roadmap.md`;
3. this identity-boundary document;
4. the dedicated current phase roadmaps (`p3_roadmap.md`, `p4_roadmap.md`, `p5_roadmap.md`).
