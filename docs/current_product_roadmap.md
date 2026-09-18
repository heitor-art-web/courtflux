# CourtFlux — Current Product Roadmap

> [!IMPORTANT]
> This document is the current product-level roadmap and phase-boundary reference.
>
> Historical implementation and experiment documents remain preserved as technical history. When historical wording conflicts with the phase ownership or numbering below, this roadmap and the dedicated current-phase roadmaps take precedence.

---

# Product direction

CourtFlux progresses from understanding a real basketball game to helping players and teams improve over time:

```text
video
  ↓
perception + tracking
  ↓
basketball understanding
  ↓
structured review
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

Every important conclusion should remain connected to the source evidence that supports it. `UNKNOWN` is preferable to unsupported certainty.

---

# Roadmap overview

```text
P0 — Vision Foundation                                      ✅ COMPLETE
P1 — Perception + Tracking                                  ✅ COMPLETE
P2 — Basketball Understanding + Game-Level Player Identity  🧪 ENGINEERING COMPLETE / GT VALIDATION PENDING
P3 — Structured Analytics + Review                          🛠 FOUNDATION MAY PROCEED CONCURRENTLY
P4 — Tactical Intelligence                                  ⏳ FUTURE
P5 — Persistent Player Identity / Jersey Evidence           ⏳ FUTURE
P6 — Mobile App                                             ⏳ FUTURE
P7 — Continuous Development / AI Coach                      ⏳ FUTURE
```

The identity distinction is fundamental:

```text
Track ID
  = technical visual tracking identity

Game Player Entity
  = stable identity for the same player inside one game

Persistent Player Identity
  = canonical real-world player identity across games
```

P1/P2 do not require CourtFlux to know that a game entity is `João Silva #7`. P2 requires enough game-level continuity for events and later analytics to belong to the correct game player. Persistent real-world identity is a P5 responsibility.

---

# Active execution policy — P2 + P3 foundation

P2 architecture is sufficiently mature for P3 structural work to begin without waiting for final P2 accuracy calibration.

The two tracks may run concurrently:

```text
P2 VALIDATION                         P3 FOUNDATION
      │                                     │
reviewed human GT                    P2→P3 adapter / contract
basketball correctness               persistent Game Record
player attribution                   provenance / versioning
measured calibration                 physical clip pipeline
      │                              review / corrections
      │                              fixtures / deterministic analytics
      │                                     │
      └──────────── integration ────────────┘
                         ↓
                validated P2 baseline
                         ↓
                   P3 product freeze
```

Concurrency rules:

- P3 consumes a stable, versioned P2 event contract;
- P3 must not import P2 engine internals as product dependencies;
- P3 must not create a second possession/event engine to compensate for P2;
- P3 must not tune P2 thresholds from product intuition;
- calculations whose correctness depends on final P2 calibration remain configurable and reprocessable;
- historical `analysis_run` provenance must identify which P1/P2/P3 versions produced each output;
- when P2 passes its exit gate, P3 fixtures/integration are rerun against the frozen baseline before P3 semantics are frozen.

This allows useful engineering parallelism without turning unvalidated P2 output into irreversible product truth.

---

# P0 — Vision Foundation

## Question

> Can CourtFlux process basketball video through a measurable, repeatable computer-vision foundation?

## Owns

- video-processing foundation;
- detector/evaluation infrastructure;
- court geometry foundations;
- explicit confidence and diagnostics;
- repeatable experimentation and baseline comparison.

## Status

Complete.

Historical implementation details remain in the P0/P1 technical documents and are not redefined here.

---

# P1 — Perception + Tracking

## Question

> What players, teams, ball and court state can CourtFlux perceive and track through the video?

## Owns

- player, referee, ball and court perception;
- raw player tracking and temporal continuity;
- team evidence;
- `PlayerState`, `BallState` and `CourtState`;
- confidence, provenance and `UNKNOWN` / abstention behaviour.

## Does not require

- real-world player names;
- roster matching;
- reliable Jersey OCR;
- cross-game person identity.

A `track_id` is a technical tracking identifier only.

## Exit definition

P1 is complete when perception and tracking expose stable, confidence-aware state suitable for basketball reasoning.

See `docs/p1_p2_identity_boundary.md` for the current identity boundary. Historical P1 execution remains documented separately.

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

```text
raw track(s)
    ↓
game_player_03
    ↓
POSSESSION / PASS / SHOT / TURNOVER / REBOUND events
```

P2 does not need to know that `game_player_03` is João Silva or that the player wears #7.

## Jersey position

Jersey recognition remains useful supporting evidence/R&D but is not a P1/P2 exit gate. `shirt_num` and `player_name` may remain `UNKNOWN`.

## Status / exit definition

The P2 engineering core is implemented. Exit requires basketball correctness against reviewed human GT plus game-player attribution stable enough to support P3 analytics.

Detailed execution and validation history lives in `docs/p2_roadmap.md`.

---

# P3 — Structured Analytics + Review

## Product question

> What deserves attention in this game, and can I immediately review the evidence?

P3 turns P2 basketball understanding into a durable, reopenable review product.

## Core outcomes

- persistent Game Record;
- possession/event aggregation;
- per-team and per-game-player summaries;
- physical evidence clips linked to source timestamps;
- `Things to Review` as the primary review surface;
- timeline and filters;
- conservative explainable insights;
- human correction without destroying original predictions;
- save / tag / note / collection actions;
- versioning, provenance and reprocessing semantics.

## Concurrent foundation rule

P3 structural development may begin while P2 validation continues.

Safe concurrent work includes:

- persistence/data contracts;
- P2→P3 adapter;
- analysis/model/schema versioning;
- Game Record;
- physical clip generation;
- review UI/data model;
- human corrections;
- evidence lineage;
- fixture-driven deterministic aggregation;
- save/tag/note/collection infrastructure.

P2-accuracy-dependent metrics, rankings and review prioritization remain configurable/reprocessable until P2 passes its exit gate.

## Exit definition

A user can upload a game and rapidly find the moments, metrics and game-level players that deserve attention without manually searching through 60–90 minutes of video, while uncertainty and evidence remain visible.

Detailed execution lives in [`docs/p3_roadmap.md`](p3_roadmap.md).

---

# P4 — Tactical Intelligence

## Product question

> How did this team create or defend advantages, and can I immediately review every example?

P4 moves from isolated basketball events into temporal relationships between players, space and possession structure.

## Core progression

```text
P4.0 — Tactical State Contract
P4.1 — Spatial Foundation
P4.2 — Tactical Primitives
P4.3 — Offensive Transition
P4.4 — Offensive Actions
P4.5 — Tactical Segmentation
P4.6 — Game + Team Tendencies
P4.7 — Defensive Intelligence
P4.8 — Tactical Review + Radar
P4.9 — Human Correction + Tactical GT
P4.10 — Validation + Integrity Exit Gate
```

P4 is offense-first. Defensive intelligence follows the validated spatial/offensive foundation.

Tactical truth is temporal and may be multi-label. When geometry, tracking or contextual evidence is insufficient, output must remain `UNKNOWN` / `NOT_TACTICALLY_ANALYZABLE` rather than forcing a plausible tactical label.

## Integrity exit principle

Every important P2→P4 analytical conclusion must be:

```text
VERSIONED
TRACEABLE
CORRECTABLE
MEASURABLE
REPRODUCIBLE
```

Detailed execution lives in [`docs/p4_roadmap.md`](p4_roadmap.md).

---

# P5 — Persistent Player Identity / Jersey Evidence

## Product question

> Who is this game-level player in the real roster, and is it the same person across games?

P5 resolves game-local player entities into durable real-world player identities.

P5 is deliberately **not** a second attempt to make Jersey OCR solve identity by itself.

```text
game 1.game_player_03
            \
             \
game 2.game_player_07 ---> persistent_player_id ---> João Silva
             /
            /
game 3.game_player_02
```

## Core progression

```text
P5.0 — Persistent Identity Contract
P5.1 — Canonical Player + Membership Model
P5.2 — Human Identity Confirmation MVP
P5.3 — Identity Evidence Store
P5.4 — CourtFlux Jersey vNext
P5.5 — Appearance / ReID
P5.6 — Multi-Evidence Identity Resolver
P5.7 — Cross-Game Identity Memory
P5.8 — Player History + Product Integration
P5.9 — Identity Validation / Exit Gate
```

## Jersey position

The historical Jersey investigation is an asset, not a phase to restart.

P5 reuses:

- Jersey GT tooling;
- Identity v2 temporal/reversible evidence handling;
- `UNKNOWN` / abstention policy;
- roster-as-validator rule;
- historical model/ROI/legibility findings.

New Jersey model work should reopen only through CourtFlux-specific labelled data and measured adaptation/fine-tuning, not another ungrounded cycle of ROI/threshold heuristics.

## Identity principle

```text
UNKNOWN > false identity
```

False merges are especially dangerous because they silently corrupt historical player analytics and future development models.

Detailed execution lives in [`docs/p5_roadmap.md`](p5_roadmap.md).

---

# P6 — Mobile App

## Product question

> How does CourtFlux become part of the daily workflow of players and coaches?

P6 brings existing evidence/review workflows into frequent mobile use rather than creating a separate analytics truth layer.

Core scope includes:

- game feed and game summaries;
- clip review;
- player summaries/history;
- saved moments;
- notifications;
- coach → player review workflow;
- controlled sharing;
- later mobile upload/capture flows.

## Exit definition

Players and coaches can consume, review, share and act on CourtFlux analysis without requiring the desktop workflow for ordinary use.

---

# P7 — Continuous Development / AI Coach

## Product question

> How is this player or team changing over time, and what should they focus on next?

P7 moves CourtFlux from game analysis to longitudinal development intelligence.

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
REVIEW / TRAIN
  ↓
NEXT GAME
  ↓
MEASURE AGAIN
```

Core scope includes:

- persistent player/team models;
- historical trends;
- recurring pattern detection;
- meaningful-change detection;
- evidence-backed review/development priorities;
- goals and progress;
- AI Coach reasoning over structured CourtFlux evidence.

The AI Coach must reason from versioned P2/P3/P4/P5 evidence rather than act as a generic chatbot over raw video.

## Exit definition

CourtFlux can use evidence from multiple real games to identify recurring patterns, track meaningful change and suggest review/development priorities that remain linked to the underlying basketball evidence.

---

# Phase dependency summary

```text
P0
vision foundation
  ↓
P1
reliable perception / tracking state
  ↓
P2
basketball understanding + game-player attribution
  ↓
P3
persistent review / analytics / corrections
  ↓
P4
tactical understanding
  ↓
P5
persistent real-world player identity
  ↓
P6
mobile daily workflow
  ↓
P7
longitudinal development / AI Coach
```

P3 foundation may overlap temporally with final P2 validation, but semantic freeze still follows validated P2 integration.

---

# Reference-history policy

Historical execution and experiment documents are intentionally retained. They record baselines, rejected challengers, research stops and earlier phase assumptions that remain useful for engineering reference.

Rules:

- do **not** rewrite historical experiment results to make them match later roadmap decisions;
- do update current roadmap/boundary documents when phase ownership changes;
- use `README.md` and this document for current phase order;
- use `docs/p1_p2_identity_boundary.md` for the current identity boundary;
- use `docs/p3_roadmap.md`, `docs/p4_roadmap.md` and `docs/p5_roadmap.md` for detailed current execution contracts;
- treat `docs/p1_jersey_exit_gate.md` and `docs/jersey_recon_rnd.md` as historical engineering evidence, not current phase definitions.

---

# Current engineering focus

```text
P2
reviewed GT + measured basketball correctness
+ game-player attribution validation

        in parallel with

P3 foundation
stable adapter + persistence + provenance
+ clips + review + corrections + fixtures
```

Do not let P3 concurrency weaken the P2 correctness gate. Do not let unfinished P2 calibration block structural P3 work that can be safely versioned and reprocessed.
