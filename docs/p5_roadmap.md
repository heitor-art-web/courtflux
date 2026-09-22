# P5 — Persistent Player Identity Roadmap

P5 turns CourtFlux game-local player entities into durable real-world player identities across games.

It is deliberately **not** a second attempt to make Jersey OCR solve identity by itself.

The core problem is:

```text
game 1.game_player_03
            \
             \
game 2.game_player_07 ---> persistent_player_id ---> João Silva
             /
            /
game 3.game_player_02
```

P5 combines roster context, Jersey evidence, appearance/ReID, previous games, team context and human confirmation to resolve that mapping conservatively and reproducibly.

The central safety rule is:

```text
UNKNOWN > false identity
```

A missing identity is inconvenient. A false merge silently corrupts every downstream player statistic, tactical tendency and future development model.

---

# Current status

P5 is a future phase after P4 Tactical Intelligence.

CourtFlux already has substantial identity-related engineering history that P5 must reuse rather than repeat:

- tracked player crops and game-local player continuity;
- team evidence and team-aware identity context;
- Jersey ground-truth extraction / annotation / export tooling;
- uncertainty-aware Jersey benchmarks;
- Identity v2 temporal evidence infrastructure;
- reversible identity locks and return to `UNKNOWN` on sustained conflict;
- roster-as-validator policy;
- current `team_players` roster management in the web product;
- game-player IDs consumed by P3/P4 analytics.

The purpose of P5 is to assemble these foundations into a **persistent identity system**, then add the missing cross-game evidence and product workflow.

---

# Product question

> Who is this game-level player in the real roster, and is it the same person across games?

The product outcome is not merely:

```text
#7 detected
```

It is:

```text
this game_player is João Silva
this is the same João Silva from previous games
all P3/P4 history can now be attached to one persistent player
```

The P5 magic moment is:

```text
Game analyzed
      ↓
10 game players found
      ↓
9 automatically resolved with strong evidence
      ↓
1 ambiguous player shown for confirmation
      ↓
coach confirms / changes / leaves UNKNOWN
      ↓
full game immediately appears in each player's history
```

A successful P5 product should progressively require **less human confirmation** as the system accumulates trustworthy identity evidence across games.

---

# Historical Jersey / Identity lessons that P5 must preserve

P5 begins with a large amount of prior R&D. Those results are architectural input, not discarded history.

## 1. Jersey signal exists, but Jersey-only identity is insufficient

Earlier selective Jersey evaluation demonstrated that high precision is possible when CourtFlux abstains aggressively, but held-out support and coverage were limited.

A historical uncertainty-aware track policy reached approximately:

```text
all-split decision precision   93.3%
numeric-track coverage         38.5%
unreadable false accept         0.0%
```

Those numbers were useful evidence of recoverable signal, not a production identity claim.

The important lesson is:

```text
selective high-confidence Jersey evidence can help identity
                !=
Jersey OCR can own identity
```

## 2. Several Jersey recognizer directions were already explored

The investigation included, among other approaches:

- historical digit / temporal checkpoints;
- uncertainty-aware ViT recognition;
- SmolVLM2 Jersey recognition;
- TrOCR;
- SoccerNet PARSeq;
- temporal consensus;
- geometric localized ROIs;
- OCR-blind region proposals;
- SoccerNet legibility filtering;
- hosted / multimodal OCR experiments.

The outcome was not a production-ready lightweight recognizer.

P5 must not restart the same sequence with slightly different thresholds.

## 3. Jersey reconstruction v1-v5 produced a deliberate hard stop

The controlled Porto/Ovarense reconstruction sequence produced the following diagnostic identification yields:

| Version | Strategy | Diagnostic identification |
| --- | --- | ---: |
| v1 | single torso crop → PARSeq | 13.2% |
| v2 | multi-frame evidence → consensus | 4.31% |
| v3 | multi-frame + localized geometric ROIs | 4.57% |
| v4 | contrast/component region proposals | 5.47% |
| v5 | legibility gate → region proposals → PARSeq | 1.50% |

These were not verified accuracy measurements because that phase did not use reviewed GT for the final integrated comparison.

The durable conclusion was:

```text
Do not invest further in the current SoccerNet/PARSeq + ROI/threshold stack
without CourtFlux-specific labelled data.
```

If Jersey research is reopened in P5, the next credible direction is:

```text
CourtFlux-specific data
      ↓
frozen benchmark
      ↓
adaptation / fine-tuning
      ↓
measured comparison
```

not another loop of ad-hoc ROI and threshold tuning.

## 4. Hard legibility gating is dangerous

The historical legibility-gated v5 path rejected too much usable evidence and collapsed coverage without enough consistency gain.

P5 should prefer:

```text
soft crop / frame ranking
```

over:

```text
hard legible / not-legible gating
```

unless future CourtFlux-specific GT demonstrates otherwise.

## 5. Temporal consensus belongs in one place

Identity v2 already owns temporal evidence aggregation and reversible locking.

Do not stack independent temporal voting inside each recognizer and then perform another temporal consensus layer in the identity engine.

Preferred contract:

```text
recognizer
    ↓
one raw observation or UNKNOWN
    ↓
identity evidence store
    ↓
Identity resolver / temporal history
```

## 6. Roster is validator/context, never a decoder

Forbidden:

```text
weak visual signal says 17
roster contains 7 but not 17
        ↓
pretend visual signal meant 7
```

Correct:

```text
visual evidence says 17
roster conflicts
        ↓
CONFLICT / UNKNOWN
```

Roster context may restrict impossible candidates or strengthen compatible evidence, but it must never manufacture visual truth.

## 7. Identity must remain reversible

Historical Identity v2 introduced an important rule:

```text
identity lock != permanent latch
```

New contradictory evidence may:

```text
RESOLVED
   ↓
CONFLICTED
   ↓
UNKNOWN / REVIEW
   ↓
RESOLVED again
```

P5 extends that same principle from intra-game evidence to cross-game identity.

---

# P5 architecture

P5 consumes stable game-level identities from P2/P3/P4 and adds a persistent person layer above them.

```text
P1
tracks / visual evidence
        ↓
P2
game_player_id + basketball attribution
        ↓
P3/P4
game analytics + tactical history
        ↓
P5
persistent identity resolution
        ↓
persistent_player_id
        ↓
player history across games
```

Identity evidence flows independently into the resolver:

```text
                    Jersey evidence
                         ↓
Appearance/ReID → Identity Resolver ← roster/team context
                         ↑
                    previous games
                         ↑
                 human confirmation
```

No single automated evidence source owns identity.

---

# P5.0 — Persistent Identity Contract

## Goal

Define a stable identity model before introducing new recognition models.

## Identity hierarchy

CourtFlux must preserve the distinction:

```text
track_id
  = technical visual tracking identity

game_player_id
  = stable basketball identity inside one game

persistent_player_id
  = durable real-world person identity across games
```

These identifiers must never be silently substituted for each other.

## Proposed resolution states

```text
UNKNOWN
CANDIDATE
AUTO_RESOLVED
HUMAN_CONFIRMED
CONFLICTED
SUPERSEDED
```

Possible semantics:

- `UNKNOWN` — insufficient evidence;
- `CANDIDATE` — one or more possible persistent players exist, but no safe resolution;
- `AUTO_RESOLVED` — automatic resolver crossed the frozen acceptance gate;
- `HUMAN_CONFIRMED` — user explicitly confirmed the mapping;
- `CONFLICTED` — meaningful contradictory evidence exists;
- `SUPERSEDED` — a previous resolution has been replaced while preserving history.

## Required provenance

Every resolution must record at minimum:

```text
resolution_id
analysis_run_id
game_id
game_player_id
persistent_player_id | null
status
confidence
resolver_version
evidence_ids[]
created_at
supersedes_resolution_id | null
human_review_id | null
```

A current identity is the active resolution, not a destructive overwrite of historical decisions.

## P5.0 acceptance

P5.0 is complete when:

- identifier semantics are explicit;
- resolution states are versioned;
- previous decisions are preserved;
- `UNKNOWN` is supported structurally;
- a resolution can be traced to evidence and resolver version;
- P3/P4 can continue to function using game-player IDs even when P5 is unresolved.

---

# P5.1 — Canonical Player + Membership Model

## Goal

Introduce a real persistent-player entity without corrupting the existing roster model.

The existing `team_players` table is useful roster context, but it was intentionally designed as current team roster data rather than a canonical cross-team player profile.

P5 should introduce a distinct persistent layer.

## Conceptual model

```text
players
-------
player_id
workspace_id
canonical_name
metadata
created_at
updated_at
```

and:

```text
player_team_memberships
-----------------------
membership_id
player_id
team_id
season
jersey_number
position
valid_from
valid_to
is_active
source
created_at
updated_at
```

## Jersey number is membership metadata

Never model:

```text
player.jersey_number = permanent identity
```

Prefer:

```text
João Silva
  2026 Porto      #7
  2028 Braga      #12
```

The `persistent_player_id` remains stable while membership metadata changes.

## Bridge to existing roster

Existing `team_players` rows should be bridgeable to persistent players through an explicit mapping rather than silently reinterpreting their IDs.

Conceptually:

```text
team_players.id
      ↓
roster_player_link
      ↓
persistent_player_id
```

Migration must preserve existing product data and historical analysis snapshots.

## Team identity prerequisite

Cross-game player resolution requires stable team mapping.

P5 must not compare every player against every known person. Candidate generation should first use reliable contextual constraints such as:

- workspace;
- team;
- season/date;
- active roster window;
- game metadata.

## P5.1 acceptance

- canonical player records exist independently of roster rows;
- membership history supports Jersey-number changes;
- existing roster data is preserved;
- active/inactive roster state is temporal;
- historical games do not change when a current roster is edited.

---

# P5.2 — Human Identity Confirmation MVP

## Goal

Make persistent identity useful before automatic identity resolution is perfect.

P5 should first solve identity as a product workflow.

After game analysis:

```text
IDENTITY REVIEW

Player 1
Likely João Silva #7
[Confirm] [Choose another] [Unknown]

Player 2
Miguel Costa #12
[Confirm] [Choose another] [Unknown]
```

## Supported actions

For every game player, a coach/admin can:

```text
Confirm suggested player
Choose another roster player
Create new persistent player
Leave UNKNOWN
Mark conflict / not enough evidence
```

## Human confirmation semantics

Human confirmation is strong identity evidence, but still versioned and reversible.

If a later user corrects the mapping:

```text
old HUMAN_CONFIRMED resolution
      ↓
SUPERSEDED
      ↓
new HUMAN_CONFIRMED resolution
```

Do not destructively rewrite the original decision.

## Product objective

Initial P5 success does **not** require 100% automatic identity.

A small confirmation workflow is acceptable if it reliably converts:

```text
game_player_id → persistent_player_id
```

with minimal user effort.

Automation should progressively reduce the number of confirmations required per game.

## P5.2 acceptance

A coach can finish an identity review for a game and immediately see P3/P4 outputs attached to persistent player profiles.

---

# P5.3 — Identity Evidence Store

## Goal

Make identity evidence first-class, queryable and auditable.

The resolver must not receive only a final probability. It must preserve why a candidate was considered.

## Evidence types

Initial types:

```text
TEAM
JERSEY
APPEARANCE
ROSTER
PREVIOUS_GAME
HUMAN
```

Optional later type:

```text
FACE
```

Face evidence should remain optional because normal game footage often lacks sufficient frontal resolution.

## Conceptual evidence record

```text
IdentityEvidence

evidence_id
game_id
game_player_id
candidate_player_id | null
kind
value
confidence
quality
source_frame_ids[]
source_clip_ids[]
model_name
model_version
schema_version
created_at
```

## Evidence independence

Where possible, store raw evidence separately from resolver interpretation.

Example:

```text
JERSEY evidence:
raw number candidate = 7
confidence = 0.84
frame ids = [...]
```

rather than only:

```text
João Silva confidence = 0.84
```

This separation allows future resolver improvements without rerunning expensive visual models.

## Evidence immutability / versioning

A new model run creates new evidence.

It does not silently mutate an older model's output.

## P5.3 acceptance

Every automated identity decision can answer:

```text
Why this player?
Which source frames?
Which models?
Which versions?
Which prior games?
Was there human confirmation?
```

---

# P5.4 — CourtFlux Jersey vNext

## Goal

Reopen Jersey recognition only through a data-driven CourtFlux-specific program.

P5 must respect the v1-v5 research stop:

```text
no new Jersey architecture work
without either
CourtFlux-specific labelled data
or materially new external evidence
```

## P5.4.1 — Expand CourtFlux Jersey GT

Reuse the existing extraction / annotation / validation / export tooling.

Expand beyond one broadcast/game to include diverse CourtFlux conditions:

- multiple teams;
- multiple kit designs;
- home / away kits;
- different arenas;
- different camera distances;
- lighting variation;
- motion blur;
- occlusion;
- small players in frame;
- front/back/side views;
- readable / partial / unreadable numbers;
- referees / staff / false-player detections.

## Identity grouping remains mandatory

Samples from the same physical player sequence must not leak across train/validation/test.

Use or extend `identity_group` so that fragmented tracks known to be the same player remain grouped for splitting.

## P5.4.2 — Frozen evaluation contract

Before model adaptation, freeze:

```text
training split
validation split
test split
subject taxonomy
readability taxonomy
metric definitions
promotion rules
```

Do not tune against the final test set.

## Metrics

At minimum:

```text
readable crop exact accuracy
readable crop top-k accuracy
accepted-observation precision
observation coverage
unreadable false-accept rate
track/game-player decision precision
track/game-player coverage
correct recall
calibration
runtime
```

## P5.4.3 — Soft frame quality ranking

Use crop quality as a ranking feature, not automatically as a hard rejection gate.

Potential ranking signals:

- crop resolution;
- torso resolution;
- sharpness;
- motion blur;
- contrast;
- exposure;
- view orientation when available;
- number-region likelihood;
- occlusion estimate.

Select the strongest temporally diverse observations for recognition.

## P5.4.4 — Model adaptation / fine-tuning

Only after the dataset and benchmark are frozen should CourtFlux compare:

```text
historical baseline recognizers
        vs
CourtFlux-adapted / fine-tuned candidates
```

No architecture is preselected as the winner in this roadmap.

## Output contract

The Jersey recognizer emits evidence only:

```text
raw_number_candidate | UNKNOWN
confidence
quality
source evidence
model version
```

It does not resolve a persistent player.

## P5.4 acceptance

A Jersey model is promoted only if it demonstrates a measured improvement on frozen held-out CourtFlux data while preserving conservative abstention.

---

# P5.5 — Appearance / ReID Evidence

## Goal

Add an identity signal that does not depend on the number being readable.

This is the largest new technical capability relative to the historical Jersey work.

## Identity crop bank

For human-confirmed or sufficiently trusted players, preserve a bounded set of high-quality identity crops across views and games.

Example categories:

```text
front
back
left / right side
full-body
upper-body
```

The crop bank must retain:

- game/date;
- team/kit context;
- image quality;
- source frame;
- model/provenance;
- confirmation status.

## Embeddings

Generate appearance embeddings using a versioned ReID model.

Do not assume one embedding is permanent identity truth.

Kit changes, lighting, camera quality and season changes can create domain shift.

## Candidate retrieval

Given a new game player:

```text
new crop set
      ↓
appearance embedding(s)
      ↓
nearest candidate persistent players
      ↓
resolver evidence
```

Candidate retrieval should normally be constrained by team/season context before wider search.

## Evaluation

Measure independently from the final resolver:

```text
Recall@1
Recall@K
candidate rank
same-player similarity distribution
different-player similarity distribution
cross-game robustness
cross-kit robustness when available
```

The first ReID milestone is candidate generation quality, not autonomous identity.

## P5.5 acceptance

Appearance/ReID materially improves the probability that the correct persistent player appears in a small candidate set without unacceptable false similarity behavior.

---

# P5.6 — Multi-Evidence Identity Resolver

## Goal

Combine independent evidence into an explainable persistent identity decision.

The resolver is the actual P5 identity engine.

## Candidate generation first

For each `game_player_id`, produce a bounded candidate set from:

- team membership;
- active roster window;
- previous confirmed games;
- appearance retrieval;
- Jersey compatibility;
- optionally existing manual links.

Then score candidates.

## Resolver inputs

Conceptually:

```text
candidate player

team compatibility
roster compatibility
jersey evidence
appearance similarity
previous-game evidence
human evidence
```

## Initial resolver strategy

Begin with an explainable, calibratable resolver rather than an opaque end-to-end model.

Possible early implementation:

- feature-based evidence model;
- calibrated probabilistic scoring;
- explicit hard conflicts;
- explicit abstention band.

Do not permanently freeze hand-written weights without GT.

## Hard conflicts

Examples that should reduce or block resolution:

```text
candidate not on team at game date
strong Jersey evidence conflicts with candidate membership
human confirmation points elsewhere
multiple simultaneous game players map to same player without substitution/context support
```

## Confidence calibration

Resolver confidence must eventually correspond to measured correctness.

Do not use arbitrary `0.95` labels merely because several signals agree.

## Auto-resolution policy

Automatic identity should be selective.

Conceptually:

```text
very strong evidence → AUTO_RESOLVED
ambiguous evidence   → CANDIDATE / human review
conflicting evidence → CONFLICTED
insufficient evidence→ UNKNOWN
```

## One-player / one-game integrity

Within a normal game context, two simultaneous active game-player entities must not resolve to the same persistent player without explicit evidence that the game-player segmentation itself is duplicated/fragmented.

Game-player merge logic and persistent identity logic should remain distinct but able to surface conflicts to each other.

## P5.6 acceptance

The resolver produces explainable, versioned decisions and demonstrates measured advantage over any single evidence source.

---

# P5.7 — Cross-Game Identity Memory

## Goal

Make confirmed history improve future games without turning earlier mistakes into permanent truth.

## Confirmed identity pack

After a trustworthy resolution, CourtFlux may retain a bounded identity pack containing:

```text
persistent_player_id
confirmed memberships
historical jersey metadata
appearance embeddings
high-quality crops
previous game_player links
human confirmation history
```

## Future-game reuse

On a future game:

```text
game_player
    ↓
team + roster candidate filter
    ↓
historical appearance / Jersey evidence
    ↓
candidate persistent players
    ↓
resolver
```

## Memory is evidence, not authority

Historical resolution strengthens a candidate, but must not automatically override contradictory current evidence.

A previous wrong identity must be correctable.

## Revision propagation

When a past identity is corrected, derived persistent-player history must be rebuilt or invalidated cleanly.

Example:

```text
Game 4 Player 3
was attached to João
      ↓
corrected to Miguel
      ↓
João historical aggregates stale
Miguel historical aggregates stale
      ↓
recompute derived player history
```

This must integrate with the analysis/versioning integrity principles established in P3/P4.

## P5.7 acceptance

Identity memory reduces future review work while maintaining reversible, traceable decisions.

---

# P5.8 — Player History + Product Integration

## Goal

Turn persistent identity into visible user value.

Before P5:

```text
Game 1 — Player 3
Game 2 — Player 7
Game 3 — Player 2
```

After P5:

```text
João Silva

Game 1
Game 2
Game 3
...
```

## Persistent player profile

Initial profile can expose:

```text
current team / membership
jersey history
games analyzed
review collections
P3 game analytics
P4 tactical analytics
identity confidence / review status where appropriate
```

## P3/P4 aggregation

P5 enables historical aggregation of metrics that were previously game-local.

Examples:

```text
shots across games
turnovers across games
passes / rebounds across games
transition involvement
P&R ball-handler involvement
screen involvement
drives
cuts
tactical tendencies
```

The underlying game-local data remains unchanged. Persistent-player history is a derived view over resolved mappings.

## Correction behavior

If identity changes, the player's historical aggregates must update automatically through reprocessing/invalidation rather than remain silently wrong.

## Product surfaces

Potential surfaces:

- player profile;
- game identity-review panel;
- roster page persistent-player link;
- cross-game player analytics;
- P3/P4 review filters by persistent player.

## P5.8 acceptance

A user can open one persistent player and navigate all correctly resolved games, clips, P3 analytics and P4 tactical evidence attached to that person.

---

# P5.9 — Identity Validation / Exit Gate

P5 does not exit because Jersey OCR looks good on a few clips.

It exits when the full persistent identity system is measurable and safe.

## Ground truth

Create a reviewed cross-game identity set containing multiple games and players with known mapping:

```text
game_id
game_player_id
persistent_player_gt
team_gt
membership_gt
jersey_gt where observable
```

The dataset must include:

- same player across multiple games;
- multiple players on the same roster;
- number collisions across different teams;
- changed Jersey numbers when available;
- visually similar teammates;
- unreadable Jersey cases;
- fragmented tracks/game entities;
- ambiguous cases that should remain `UNKNOWN`.

## Required evaluation areas

### Jersey evidence

- accepted precision;
- coverage;
- unreadable false-accept rate;
- calibration;
- runtime.

### Appearance / ReID

- Recall@1;
- Recall@K;
- cross-game retrieval quality;
- false similarity behavior.

### Resolver

- auto-resolution precision;
- auto-resolution coverage;
- candidate recall;
- confidence calibration;
- abstention quality.

### Identity integrity

- false merge rate;
- false split rate;
- wrong-team resolution rate;
- duplicate-player mapping conflicts;
- correction / revision behavior.

### Human workflow

- confirmations required per game;
- corrections required per game;
- time to complete identity review;
- percent of automatic resolutions accepted without correction.

### Traceability

Every final resolution must be able to return:

```text
which game player?
which persistent player?
which evidence?
which source frames/clips?
which model versions?
which resolver version?
which human review?
which prior decision was superseded?
```

## Primary safety metric — false merges

A false merge is more damaging than an unresolved identity.

```text
João → UNKNOWN
```

reduces convenience.

```text
Miguel's games → João
```

corrupts historical analytics and future AI reasoning.

Therefore the P5 release policy must optimize for conservative correctness before maximum automatic coverage.

## Exit integrity gate

Every persistent identity must be:

```text
VERSIONED
Which model/resolver produced it?

TRACEABLE
Which evidence and source footage support it?

CORRECTABLE
Can a user revise it without destroying history?

MEASURABLE
Do we know resolution quality, coverage and abstention behavior?

REPRODUCIBLE
Can the decision be reconstructed from the recorded evidence/version?
```

## P5 exit definition

P5 is complete when CourtFlux can reliably resolve a useful share of game-level player entities to persistent real-world players across games, safely abstain when evidence is insufficient, provide a low-friction confirmation workflow for ambiguity, and maintain correct cross-game player history with full evidence/provenance.

100% automatic identity is **not** required.

A high-confidence wrong player identity is unacceptable merely to increase coverage.

---

# Recommended execution order

Do not start P5 by training another Jersey model.

Recommended sequence:

```text
1. P5.0 identity contract
        ↓
2. P5.1 canonical player / membership schema
        ↓
3. P5.2 human confirmation MVP
        ↓
4. P5.3 identity evidence store
        ↓
5. cross-game identity GT foundation
        ↓
6. P5.5 appearance/ReID baseline
        ↓
7. P5.4 CourtFlux Jersey dataset + vNext benchmark
        ↓
8. P5.6 multi-evidence resolver
        ↓
9. P5.7 identity memory
        ↓
10. P5.8 player history
        ↓
11. P5.9 validation / exit
```

The ordering intentionally puts **human-confirmable persistent identity before perfect automation**.

This creates immediate product value and also generates high-quality identity labels that improve later ReID/Jersey/resolver development.

---

# What P5 should not do

P5 should not:

- treat Jersey number as permanent identity;
- treat roster membership as visual truth;
- force a player name when evidence is weak;
- reopen SoccerNet/PARSeq threshold tuning without new data;
- optimize only on Porto/Ovarense;
- use one broadcast clip as proof of general identity accuracy;
- report coverage/acceptance as accuracy;
- stack multiple hidden temporal-consensus layers;
- permanently lock a player from one weak observation;
- silently overwrite corrected identity history;
- make face recognition mandatory;
- block P3/P4 analytics when real-world identity is unknown.

---

# What P5 deliberately reuses

P5 should reuse and evolve:

- Identity v2 evidence-window concepts;
- reversible lock/unlock semantics;
- roster validator-only policy;
- `UNKNOWN` preference;
- existing Jersey GT tooling;
- identity-group-aware dataset splitting;
- existing team / roster product surfaces;
- game-player identity from P2;
- P3/P4 provenance and reprocessing semantics;
- evidence-linked review UI patterns from P3;
- cross-game team identity required by P4 tendencies.

---

# Relationship to later phases

## P6 — Mobile App

P5 gives the mobile product stable player profiles, persistent review history and coach-to-player routing.

## P7 — Continuous Development / AI Coach

P7 depends strongly on P5 correctness.

Without persistent identity:

```text
Game 1 Player 3
Game 2 Player 7
Game 3 Player 2
```

cannot safely become:

```text
João Silva — 3-game trend
```

P5 therefore acts as the identity bridge between game-level intelligence and longitudinal development intelligence.

---

# Final P5 milestone

The P5 milestone is reached when this flow is real:

```text
GAME
  ↓
P2 game-player entities
  ↓
P3/P4 analytics
  ↓
P5 identity evidence
  ↓
automatic candidate resolution
  ↓
human confirms only ambiguity
  ↓
persistent_player_id
  ↓
CROSS-GAME PLAYER HISTORY
```

At that point CourtFlux no longer merely understands anonymous players inside individual games.

It begins to understand the continuing basketball history of the same real player over time.
