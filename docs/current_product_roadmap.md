# CourtFlux — Current Product Roadmap

> [!IMPORTANT]
> This document is the current product-level roadmap and phase boundary reference.
>
> Historical implementation documents such as `docs/p1_roadmap.md`, `docs/p1_jersey_exit_gate.md` and detailed P2 experiment notes remain preserved as technical history. Where older wording treats Jersey recognition or real-world player naming as a P1/P2 exit gate, the current boundary in `docs/p1_p2_identity_boundary.md` supersedes that phase interpretation.

## Product direction

CourtFlux is being built around a progression from understanding a real basketball game to helping players and teams improve over time:

```text
video
  ↓
understanding
  ↓
structured analysis
  ↓
tactical intelligence
  ↓
persistent player identity
  ↓
daily/mobile workflow
  ↓
continuous development
```

The product loop remains:

```text
Insight → Evidence → Improvement
```

Every important insight should remain connected to the source video evidence that supports it.

---

# Roadmap overview

```text
P0 — Vision Foundation                                      ✅ COMPLETE
P1 — Perception + Tracking                                  ✅ COMPLETE
P2 — Basketball Understanding + Game-Level Player Identity  🧪 ENGINEERING COMPLETE / GT VALIDATION PENDING
P3 — Structured Analytics + Actions                         ⏳ NEXT
P4 — Tactical Intelligence                                  ⏳ FUTURE
P4.5 — Persistent Player Identity                           ⏳ FUTURE
P5 — Mobile App                                             ⏳ FUTURE
P6 — Continuous Development / AI Coach                      ⏳ FUTURE
```

The identity distinction is fundamental:

```text
Track ID
  = technical visual tracking identity

Game Player Entity
  = stable identity for the same player inside one game

Persistent Player Identity
  = real roster/person identity across games
```

P1/P2 do not require CourtFlux to know that a game entity is “João Silva #7”. They do require enough game-level continuity for events and later analytics to belong to the correct player entity inside that game.

---

# P0 — Vision Foundation

## Purpose

Establish the core video-processing and computer-vision foundation required by the rest of CourtFlux.

## Status

Complete.

Historical implementation details remain documented elsewhere and are not redefined here.

---

# P1 — Perception + Tracking

## Question

> What players, teams, ball and court state can CourtFlux perceive and track through the video?

## Owns

- player, ball and court perception;
- raw player tracking and temporal continuity;
- team evidence;
- `PlayerState`, `BallState` and `CourtState`;
- confidence, provenance and `UNKNOWN` / abstention behaviour.

## Does not require

- real-world player names;
- roster matching;
- reliable Jersey OCR;
- cross-game person identity.

Raw `track_id` is a technical tracking identifier only.

## Exit definition

P1 is complete when perception and tracking expose stable, confidence-aware state suitable for basketball reasoning.

See `docs/p1_p2_identity_boundary.md` for the current identity boundary. Historical P1 execution remains in `docs/p1_roadmap.md`.

---

# P2 — Basketball Understanding + Game-Level Player Identity

## Question

> What happened in basketball terms, and which game-level player entity was involved?

## Owns

- ball-control state;
- possession acquisition and change;
- loose-ball lifecycle;
- pass and turnover candidates;
- shot attempts and supported outcome evidence;
- rebound candidates;
- event attribution to stable game-level player entities;
- confidence-aware basketball semantics;
- reviewed human-ground-truth validation.

## Identity requirement

P2 needs to distinguish the same player consistently enough inside one game to support player-level event aggregation.

Conceptually:

```text
raw track(s)
    ↓
game_player_03
    ↓
POSSESSION / PASS / SHOT / TURNOVER / REBOUND events
```

P2 does not need to know that `game_player_03` is João Silva or that the player wears #7.

## Jersey position

Jersey recognition remains useful R&D and may provide optional evidence, but is not a P1/P2 exit gate.

## Status / exit definition

The P2 motor is engineering-complete. Exit requires basketball correctness against reviewed human GT plus game-player attribution stable enough to support P3 analytics.

---

# P3 — Structured Analytics + Actions

## Question

> What does this game tell us, and what should the user review?

P3 is the point where CourtFlux turns basketball understanding into a directly usable analysis product.

## P3.1 — Game Analytics Core

Aggregate P2 events into structured game outputs:

- possessions;
- shots;
- turnovers;
- passes;
- rebounds;
- loose balls;
- event counts and rates;
- possession timeline;
- team splits;
- per-game-player splits.

Real-world player names are not required. A valid output can be:

```text
Player 3
12 possessions
4 shot attempts
2 turnover candidates
6 review clips
```

## P3.2 — Evidence-linked Review

Every important event, metric or insight must link back to the source footage.

Example:

```text
Turnover candidate
02:14
[Review clip]
```

This is a core product principle, not an optional UI enhancement.

## P3.3 — Smart Clip Organization

Automatically create useful review collections such as:

- all shots;
- turnovers;
- possession changes;
- loose-ball sequences;
- possessions ending in a shot;
- player-specific possessions;
- team offensive/defensive possessions.

## P3.4 — Insight Layer

Produce conservative, explainable observations from structured game data.

Examples:

- `Player 4 had 5 possessions ending in a turnover candidate.`
- `Team A generated 8 shot attempts after possession changes.`
- `6 of Player 2's 9 reviewable possessions occurred in the second half.`

P3 should not make unsupported coaching judgments or pretend candidate events are perfect truth.

## P3.5 — Action Workflow

Turn review into action:

```text
Insight
  ↓
Evidence
  ↓
Save / Tag
  ↓
Review
  ↓
Action
```

Examples:

- review these possessions;
- save a clip;
- add to a player review;
- create a coach review collection.

## P3 exit definition

P3 is complete when a user can upload a game and rapidly find the moments, metrics and game-level players that deserve attention without manually searching through 60–90 minutes of video.

This is the first major CourtFlux product “magic moment”.

---

# P4 — Tactical Intelligence

## Question

> How is this team playing, and what basketball patterns are repeating?

P4 moves from isolated events toward relationships between players, space and possession structure.

## P4.1 — Spatial Team Structure

Model:

- player positions;
- ball position;
- spacing;
- width/depth;
- paint/perimeter occupancy;
- inter-player distances;
- offensive and defensive shape.

## P4.2 — Transition Intelligence

Identify and characterize:

- offensive transition;
- defensive transition;
- fast-break situations;
- numerical advantage/disadvantage;
- early offense;
- possession stabilization.

## P4.3 — Screening / Pick-and-Roll Intelligence

Build confidence-aware recognition of patterns such as:

- screen candidate;
- ball handler;
- screener;
- roll / pop;
- reject;
- switch;
- drop;
- hedge.

## P4.4 — Defensive Behaviour

Explore supported evidence for:

- help;
- rotations;
- closeouts;
- switches;
- matchup changes;
- rim protection;
- defensive collapse and recovery.

Weak or ambiguous evidence should remain `UNKNOWN` rather than force a tactical label.

## P4 tactical milestone — Possession Pattern Classification

Classify possessions when evidence supports patterns such as:

```text
transition
pick-and-roll
isolation
post-up
off-ball screen
handoff
spot-up sequence
broken play
unknown
```

This is an internal P4 milestone and should not be confused with roadmap phase **P4.5 — Persistent Player Identity** below.

## P4.6 — Team Tendencies

Aggregate tactical patterns into game/team tendencies while keeping evidence and sample size visible.

Examples:

- P&R usage frequency;
- transition shot generation;
- recurring defensive coverage behaviour;
- offensive/defensive structural tendencies.

## P4 exit definition

P4 is complete when CourtFlux can reliably characterize what kind of basketball is happening in a possession and identify meaningful tactical patterns that repeat through a game.

Real-world player names are still not required for this phase.

---

# P4.5 — Persistent Player Identity

## Question

> Who is this game-level player in the real roster, and is it the same person across games?

This phase is deliberately separate from P2 game-level identity.

## P4.5.1 — Roster Model

Create persistent player records with concepts such as:

```text
player_id
name
team_id
jersey_number
position
active roster status
identity history
```

Jersey number must be treated as mutable metadata, not permanent identity.

## P4.5.2 — Human Confirmation Workflow

Provide a robust product fallback before attempting perfect automation:

```text
We found 9 game players
    ↓
Match / confirm roster identities
```

Example:

```text
Game Player 3 → João Silva
Game Player 4 → Miguel Costa
```

A short confirmation flow is acceptable; false automatic identity is not.

## P4.5.3 — Jersey Evidence

Use Jersey OCR as one probabilistic source of identity evidence rather than the identity system itself.

## P4.5.4 — Appearance / ReID Evidence

Add evidence such as:

- body appearance embeddings;
- kit details;
- shoes/accessories when useful;
- relative height/build cues;
- visual similarity across clips.

## P4.5.5 — Face Evidence When Available

Face evidence may help in suitable footage but should remain optional/secondary because standard game video often lacks sufficient quality.

## P4.5.6 — Cross-game Resolution

Resolve:

```text
game_1.game_player_03
game_2.game_player_07
game_3.game_player_02
          ↓
     player_joao_silva
```

with confidence and provenance.

## P4.5.7 — Identity Memory

Confirmed identities from prior games become evidence for future games, allowing the system to improve progressively while preserving user correction.

## P4.5 exit definition

P4.5 is complete when CourtFlux can associate game-level player entities with persistent roster identities across games using automated evidence plus human confirmation where necessary.

100% automatic identity is not required.

---

# P5 — Mobile App

## Question

> How does CourtFlux become part of the daily workflow of players and coaches?

The mobile app should not be a simple responsive copy of the website. It should focus on frequent consumption, review, collaboration and action.

## P5.1 — Mobile Review Experience

Core app experience:

- game feed;
- game summaries;
- player summaries;
- clip playback;
- event timeline;
- filters;
- saved clips / favourites.

## P5.2 — Notifications

Examples:

- analysis ready;
- possessions ready for review;
- coach added clips;
- new game attached to a player profile;
- new coach feedback.

Notifications should support a useful product loop, not become noise.

## P5.3 — Coach → Player Workflow

Allow a coach to:

```text
select possessions
  ↓
add note / context
  ↓
assign to player
  ↓
player receives mobile review
```

## P5.4 — Sharing

Support controlled sharing of:

- clips;
- review collections;
- private team/player links;
- coach-player feedback;
- later, export formats suitable for external sharing.

## P5.5 — Mobile Upload / Capture

Later in P5, support:

- selecting video from device;
- direct upload;
- background upload;
- analysis status;
- game metadata;
- potentially guided capture.

Capture/upload should not block the first useful mobile release.

## P5.6 — Player Home

With persistent identity available, give players a useful home surface for:

- latest game;
- review moments;
- recent-game trends;
- coach feedback;
- saved priorities.

## P5 exit definition

P5 is complete when players and coaches can consume, review, share and act on CourtFlux analysis without needing the desktop workflow for ordinary use.

---

# P6 — Continuous Development / AI Coach

## Question

> How is this player or team changing over time, and what should they focus on next?

P6 moves CourtFlux from game analysis into continuous basketball development.

The core loop becomes:

```text
GAME
  ↓
ANALYSIS
  ↓
PATTERN
  ↓
FOCUS
  ↓
TRAIN / REVIEW
  ↓
NEXT GAME
  ↓
MEASURE AGAIN
```

## P6.1 — Player Model

Build a persistent, multidimensional model rather than a single player score.

Possible dimensions include:

- possession involvement;
- shooting;
- passing;
- turnovers;
- transition;
- off-ball behaviour;
- defence;
- decision patterns;
- tactical role;
- usage/context.

## P6.2 — Historical Trends

Measure trends across real games while preserving sample size and uncertainty.

Examples:

- shot volume;
- turnover frequency;
- possession involvement;
- transition involvement;
- passing tendencies;
- tactical role changes.

## P6.3 — Development Signals

Detect meaningful changes or recurring patterns without overstating causality.

Examples:

- a repeated turnover pattern under pressure;
- changing transition involvement;
- changes in shot locations or possession outcomes.

## P6.4 — Improvement Priorities

Suggest evidence-backed review priorities.

Example:

```text
Suggested focus
Decision-making after first advantage

Why
8 reviewable possessions across the last 3 games

Evidence
[Clip] [Clip] [Clip]
```

No development recommendation should become detached from the evidence that produced it.

## P6.5 — Player Development Loop

Allow a user or coach to define a focus, review evidence, observe future games and measure whether the relevant pattern changes.

## P6.6 — Coach Intelligence

Help coaches understand where attention may be useful across a roster:

- recurring player patterns;
- recent changes;
- assigned development priorities;
- evidence collections;
- team-level development themes.

## P6.7 — AI Coach

Only at this maturity level should CourtFlux seriously present an AI Coach concept.

The AI Coach should reason over structured CourtFlux evidence and player/team models rather than operate as a generic chatbot over raw video.

It should answer questions such as:

> What should this player review this week?

with:

- an explainable reason;
- the relevant games/possessions;
- direct source-video evidence;
- appropriate uncertainty.

## P6.8 — Goals and Progress

Support measurable development goals connected to game evidence.

Example:

```text
Focus
Reduce turnovers under pressure

Baseline
6 / game

Current window
3 / game

Evidence
Last 5 games
```

Metrics must use appropriate sample-size and uncertainty safeguards.

## P6.9 — Team Model

Extend the longitudinal model to teams:

- offensive identity;
- defensive identity;
- transition profile;
- P&R usage;
- spacing tendencies;
- shot generation;
- turnover patterns;
- rotation behaviour.

## P6 exit definition

P6 is complete when CourtFlux can use evidence from multiple real games to identify recurring patterns, track meaningful change and suggest review/development priorities that always link back to basketball evidence.

---

# Reference-history policy

Historical P0/P1/P2 execution documents are intentionally retained. They record experiments, baselines, rejected challengers and earlier phase assumptions that remain useful for engineering reference.

The rule is:

- do **not** rewrite historical experiment results to match later roadmap decisions;
- use `docs/current_product_roadmap.md` for current product phase definitions;
- use `docs/p1_p2_identity_boundary.md` for the current identity boundary;
- use historical P1/P2 documents for implementation history and experiment evidence.

---

# Product progression

The roadmap can be summarized as:

```text
P0–P2
CourtFlux understands the game.

P3
CourtFlux makes the game fast and useful to review.

P4
CourtFlux understands how the game is being played tactically.

P4.5
CourtFlux knows who the players are across games.

P5
CourtFlux becomes part of the player's and coach's daily workflow.

P6
CourtFlux turns successive games into a continuous development system.
```

The long-term differentiation is therefore not “AI-generated stats”. It is the closed loop:

```text
real game
→ structured understanding
→ evidence
→ insight
→ action
→ next game
→ measured development
```
