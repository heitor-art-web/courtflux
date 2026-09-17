# P1 / P2 Identity Boundary

This document records the current roadmap decision for identity handling. It supersedes older P1/P2 wording where Jersey recognition or real-world player naming was treated as a P1/P2 exit gate.

For the consolidated product roadmap from P0 through P6, see `docs/current_product_roadmap.md`. Historical implementation documents remain preserved as reference and are not rewritten by this decision.

## P1 — Perception + Tracking

P1 answers:

> What players, teams, ball and court state can CourtFlux perceive and track through the video?

P1 owns:

- player / ball / court perception;
- player tracking and temporal continuity;
- team evidence;
- `PlayerState`, `BallState` and `CourtState`;
- conservative uncertainty / `UNKNOWN` behavior.

P1 does **not** require:

- a real-world player name;
- a roster match;
- a reliable jersey number;
- cross-game player identity.

Raw `track_id` is a technical tracking identity only. It must not be presented as equivalent to a real player identity.

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
- event attribution to a stable game-level player entity.

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

without knowing that `game_player_03` is João Silva or that he wears #7.

## Jersey and real-world identity

Jersey recognition remains useful supporting R&D, but it is **not a P1 or P2 exit gate**.

`shirt_num` and `player_name` are optional labels. They may remain `UNKNOWN` without blocking P2 basketball understanding.

The roadmap distinction is:

```text
Track ID                   = technical visual tracking identity
Game Player Entity         = stable identity inside one game
Persistent Player Identity = real person / roster identity across games
```

Persistent real-world identity belongs to a later roadmap phase, currently P4.5, where CourtFlux can resolve game-level entities to roster players using jersey evidence, appearance, roster context, previous games and human confirmation.

## Exit criteria

### P1 exit

P1 is complete when perception and tracking provide stable, confidence-aware state suitable for basketball reasoning. Real-world player identity is not required.

### P2 exit

P2 is complete when basketball events are validated against reviewed human ground truth **and** event attribution is stable enough at the game-player level to support P3 analytics.

P2 does not require automated knowledge of the player's name or persistent cross-game identity.

## Roadmap consequence

```text
P0 — Vision Foundation
P1 — Perception + Tracking
P2 — Basketball Understanding + Game-Level Player Identity
P3 — Structured Analytics + Actions
P4 — Tactical Intelligence
P4.5 — Persistent Player Identity
P5 — Mobile App
P6 — Continuous Development / AI Coach
```

The immediate technical priority is therefore not perfect Jersey OCR. It is reliable game-level player continuity so P3 can aggregate events and clips per player within a game.
