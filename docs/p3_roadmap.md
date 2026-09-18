# P3 — Structured Analytics + Review Roadmap

P3 turns trusted, confidence-aware P2 basketball primitives into a durable review product for real games.

```text
P2 basketball understanding
        ↓
confidence-aware events + game-player attribution
        ↓
P3 persistent game record
        ↓
structured analytics + physical evidence clips
        ↓
Things to Review
        ↓
human confirmation / correction
        ↓
save / tag / note / collection
```

The product loop remains:

```text
Insight → Evidence → Improvement
```

P3 is the first phase where CourtFlux must feel like a directly useful basketball product rather than an engineering analysis pipeline.

---

## Current status

**P3 is the next product phase after P2 validation/freeze.**

P2 remains responsible for basketball correctness, confidence-aware event primitives and stable enough game-level player attribution. P3 must not compensate for weak P2 evidence by silently converting candidates into truth.

The existing customer product already owns upload, durable `games`, `analysis_runs` and `analysis_jobs`, account history and worker execution. P3 extends that foundation into a persistent, reopenable basketball game record and customer-facing review experience; it must not create a parallel game identity or a second analysis runtime.

---

# Product question

> What deserves attention in this game, and can I immediately review the evidence?

The primary target user is a **coach**, but the review experience should remain understandable and useful to players, assistants and other basketball users without requiring analyst-level terminology.

The P3 magic moment is:

```text
Game analyzed
      ↓
Things to Review
      ↓
open a relevant basketball event
      ↓
play the physical source-evidence clip immediately
      ↓
confirm / correct it
      ↓
save, tag, note or collect it
```

Metrics and the full event timeline are important exploration tools, but they are secondary to the review workflow.

---

# Product principles

## 1. Evidence is structural

Every important P3 event, metric and insight must be traceable to source-video evidence.

Evidence linkage is not a presentation-only feature. It belongs in the data model.

```text
review item / metric / insight
          ↓
source event(s)
          ↓
source timestamps
          ↓
physical review clip
```

If evidence cannot be resolved, the product must expose that failure rather than present an unsupported basketball conclusion.

## 2. Confidence survives aggregation

P3 must preserve P2 uncertainty.

A candidate must not become an apparently exact statistic simply because it was aggregated.

Prefer:

```text
4 supported turnovers
2 additional turnover candidates
```

or:

```text
Turnover candidates: 6
5 have reviewable evidence
```

over silently presenting:

```text
Turnovers: 6
```

when the underlying events are still candidates.

## 3. Show uncertainty transparently

P3 may surface low-confidence and incomplete information when it is useful for review.

The product should distinguish concepts such as:

```text
supported
candidate
low confidence
incomplete possession
unknown actor/team
```

Low confidence is a review state, not a reason to invent certainty.

## 4. Incomplete possessions are valid records

Possessions may be fragmented or incomplete when tracking, ball evidence or basketball understanding is interrupted.

P3 should retain the useful fragment with explicit completeness/confidence metadata instead of dropping the entire possession or fabricating missing state.

## 5. Human correction is durable data

Coach corrections must be preserved as structured feedback without destroying the original prediction.

```text
original prediction
        +
original confidence / provenance
        ↓
human review
        ↓
confirmed / corrected / none
        ↓
reviewed feedback dataset
```

P3 corrections may later improve evaluation, calibration, heuristics or model training, but they must **not** trigger uncontrolled online learning or silently change the production model.

## 6. Game-level identity is sufficient

P3 operates on stable game-level player entities such as `game_player_03`.

Real-world names, roster matching, reliable Jersey OCR and cross-game persistent identity remain outside the P3 exit gate.

A valid product output is:

```text
Player 3
12 possessions
4 shot attempts
2 turnover candidates
6 review clips
```

---

# P3.0 — Analytics Contract

Before product aggregation expands, freeze the canonical P2 → P3 contract.

P3 consumes P2 events; it does not run a second basketball event engine.

## Required event fields

Conceptually, every P3-consumable source event should provide or resolve:

```text
event_id
game_id
analysis_run_id

event_type
decision / support state
confidence

start_frame
end_frame
start_ms
end_ms

actor_game_player_id
secondary_game_player_id
team_id

attributes
evidence
provenance
runtime/model version
```

Fields may be unknown when evidence is insufficient. Unknown values must remain explicit.

## Canonical time contract

P3 source-of-truth evidence uses source-video time:

```text
start_ms
end_ms
```

Display clock / game clock may be added when available, but must not replace the source-video timestamps required to recreate evidence.

## Confidence contract

P3 must preserve at least:

```text
source confidence
source decision/support state
aggregation coverage
human review state
```

These concepts must not be collapsed into one generic score.

## Provenance contract

A stored P3 output must be attributable to the analysis/runtime version that generated its source data.

Reprocessing a game in the future must not make historical output impossible to interpret.

---

# P3.1 — Persistent Game Record

A completed analysis becomes a durable, navigable and reopenable **Game Record**.

Reopening a completed game must not require rerunning the analyzer.

Conceptually:

```text
Game
├── source video
├── teams
├── analysis run / provenance
├── game players
├── possessions
├── events
├── analytics
├── clips
├── insights
├── corrections
├── tags / notes
└── review collections
```

The existing durable `games` domain remains canonical; P3 extends it rather than introducing a separate product-level game object.

## Possession record

A P3 possession should be representable approximately as:

```text
possession_id
game_id
team_id

start_ms
end_ms

start_reason
end_reason

completeness
confidence

event_ids[]
game_player_ids[]

evidence_ids[]
```

`completeness` must support fragmented/incomplete possessions.

## Game-player summary

P3 should expose per-game-player aggregation without requiring roster identity:

```text
game_player_id

possessions_involved
shots
passes
turnover_candidates
rebounds
loose_ball_events
dunks
alley_oop_candidates

review_item_ids[]
```

The precise storage schema may use normalized tables/read models, but product semantics must remain stable.

---

# P3.2 — Structured Events + Analytics

P3 aggregates source evidence into reviewable basketball outputs.

The required initial review taxonomy is:

- shots;
- turnovers;
- passes;
- rebounds;
- loose balls;
- dunks;
- alley-oop candidates.

It also exposes:

- possessions;
- event counts and rates;
- team splits;
- per-game-player splits;
- possession/event timeline;
- confidence/support breakdowns;
- evidence coverage.

## Shot semantics

A dunk is a **shot classification**, not an unrelated base event primitive.

Conceptually:

```text
SHOT_ATTEMPT
    └── shot_type = DUNK
```

Dunk classification must remain confidence-aware and evidence-linked.

## Alley-oop semantics

An alley-oop is initially a **composite candidate** derived from a supported relationship between basketball events, typically pass/reception context plus a finishing shot/dunk.

Conceptually:

```text
PASS evidence
    +
receiver / finisher relationship
    +
SHOT / DUNK evidence
        ↓
ALLEY_OOP_CANDIDATE
```

P3 must not force an alley-oop label when the relationship is ambiguous.

## No false precision

Analytics must preserve the distinction between supported events and candidates.

If an aggregate mixes confidence/support classes, the UI and API must make that visible.

---

# P3.3 — Physical Clip Pipeline

P3 generates physical evidence clips for review items.

A clip is a durable product artifact linked back to the original source interval.

Conceptually:

```text
Clip
├── clip_id
├── game_id
├── source_video_id
├── source_start_ms
├── source_end_ms
├── storage_path
├── event_ids[]
├── possession_id
├── game_player_ids[]
├── generation_version
└── generation status / error
```

The original source timestamps are mandatory even when a physical clip exists so clips can be regenerated with different context, encoding or overlays later.

## Default context windows

Initial clip windows should be configuration, not hardcoded product logic.

Recommended starting defaults:

| Review type | Pre-roll | Post-roll |
| --- | ---: | ---: |
| Pass | 4 s | 4 s |
| Shot | 6 s | 4 s |
| Turnover | 7 s | 5 s |
| Rebound | 5 s | 4 s |
| Loose ball | 6 s | 5 s |
| Dunk | 7 s | 5 s |
| Alley-oop candidate | 8 s | 5 s |

These values are starting defaults and may be calibrated from real review behaviour without changing event truth.

## Clip failure behaviour

If physical clip generation fails:

- preserve the source interval;
- expose the clip-generation failure;
- allow regeneration;
- do not silently present the review item as fully evidence-linked.

P3 validation should measure clip-generation success separately from basketball-event correctness.

---

# P3.4 — Things to Review

**Things to Review is the primary P3 product surface.**

A completed game should open into an immediately useful review queue rather than a metrics-only dashboard.

Example:

```text
Team A vs Team B

THINGS TO REVIEW

12 turnover candidates
7 low-confidence shots
4 loose-ball sequences
3 dunks
1 alley-oop candidate

[Open review]
```

The product must still provide access to all supported/candidate events through filters and timeline. The default review surface may prioritize attention, but it must not hide lower-confidence evidence from users who want to inspect it.

## Review item

Conceptually:

```text
review_item_id
game_id

review_type
source_event_ids[]
source_possession_id

game_player_ids[]
team_id

confidence
support state
completeness

clip_id
reason_shown
human_review_state
```

## Review detail

A typical review interaction should expose:

```text
Turnover candidate
Player 4
confidence 74%

Why shown
- possession change detected
- control moved Team A → Team B

[Play clip]

[Confirm]
[Correct action]
[Correct player]
[Save]
[Tag]
[Note]
```

The exact UI may evolve, but the evidence/confidence semantics are part of the product contract.

## Secondary exploration surfaces

P3 should also support:

- metrics summary;
- full event timeline;
- filters by event type;
- filters by team;
- filters by game player;
- supported vs candidate filtering;
- confidence filtering;
- reviewed/unreviewed filtering.

---

# P3.5 — Human Review + Correction

Human correction is a first-class P3 capability.

## Event correction

The reviewer can confirm the proposed action or correct it to the basketball action that actually occurred.

Initial correction labels:

```text
PASS
TURNOVER
SHOT
REBOUND
LOOSE_BALL
DUNK
ALLEY_OOP
NONE
```

`NONE` means the reviewer believes no relevant event from this correction taxonomy occurred at that review item.

Skipping an item or leaving it unreviewed must remain distinct from `NONE`.

## Player correction

The reviewer can correct game-level attribution, for example:

```text
Player 3 → Player 5
```

This does not require CourtFlux to know the real-world player name.

## Immutable prediction history

Corrections must not overwrite the original model output.

Conceptually:

```text
original_event_type
original_game_player_id
original_confidence
original_provenance

review_state
corrected_event_type
corrected_game_player_id
reviewer_id
reviewed_at
```

Useful review states include:

```text
UNREVIEWED
CONFIRMED
CORRECTED
REJECTED
```

where `REJECTED` corresponds to an explicit `NONE` correction.

## Feedback-to-model boundary

P3 stores corrections so they can later support:

- evaluation datasets;
- error analysis;
- threshold/calibration work;
- heuristic improvements;
- controlled future training/fine-tuning.

P3 does **not** implement automatic per-user online training or production-model mutation after a correction.

Reviewed human labels must remain distinguishable from model predictions and must never be generated by copying predictions into ground truth.

---

# P3.6 — Explainable Insight Layer

P3 produces conservative observations from structured game data.

Initial insights should be deterministic and explainable rather than free-form coaching judgments.

Examples:

```text
Player 4 had 5 possessions ending in turnover candidates.
```

```text
Team A generated 8 shot attempts after possession changes.
```

```text
6 of Player 2's 9 reviewable possessions occurred in the second half.
```

Every insight must resolve to:

```text
source metric/data
coverage/confidence context
supporting event/possession ids
physical evidence clips
```

P3 must not claim unsupported tactical causality, player quality, decision quality or coaching conclusions. Those require later tactical/development phases and stronger evidence.

An LLM is not required for P3 insight truth. If language generation is added later, it may phrase supported structured findings but must not invent unsupported conclusions.

---

# P3.7 — Review Actions

Keep P3 actions deliberately small.

The initial action set is:

```text
Save
Tag
Add note
Add to collection
```

Example:

```text
Collection
"Player 3 — turnovers to review"

items:
- possession_18
- event_42
- clip_91
```

Collections should reference durable P3 records rather than duplicate event truth.

## Out of scope for P3 actions

Do not expand P3 into collaboration infrastructure such as:

- coach → player assignments;
- messaging/chat;
- notifications;
- complex sharing permissions;
- mobile collaboration workflows.

Those belong to later product phases.

---

# P3.8 — Product Validation + Exit Gate

P3 does not exit because the aggregation code works. It exits when the product demonstrably reduces the effort required to review a real basketball game while preserving trustworthy evidence and uncertainty.

The current product-level exit definition remains:

> A user can upload a game and rapidly find the moments, metrics and game-level players that deserve attention without manually searching through 60–90 minutes of video.

P3 validation must make that statement measurable.

## Required validation dimensions

### 1. Evidence coverage

Measure:

```text
review items with valid physical clip
-------------------------------------
all surfaced review items
```

Also report source-interval availability and clip-generation failures separately.

### 2. Player attribution coverage

Measure the share of player-level review items/analytics that resolve to a stable game-player entity, with unknown/ambiguous attribution reported explicitly.

### 3. Possession coverage and completeness

Measure:

- represented game time / possession sequences;
- complete vs fragmented possessions;
- unknown possession ownership;
- source discontinuities.

Fragmented possessions are not automatically failures; hidden fragmentation is.

### 4. Review usefulness / correctness

Use human review states to measure, by event type:

- confirmation rate;
- correction rate;
- `NONE` / rejection rate;
- player-attribution correction rate;
- confidence calibration vs human review.

### 5. Time to evidence

Measure how long it takes a user to move from the game review surface to the correct source evidence for a review item.

### 6. Manual-search reduction

Benchmark the existing manual workflow on representative 60–90 minute games, then measure how much source-video searching is avoided with P3.

The final target should be frozen after the first representative baseline rather than invented before observing real review behaviour.

### 7. Reopenability

A completed Game Record must be reopenable later with its:

- analytics;
- events;
- clips;
- corrections;
- notes/tags/collections;
- provenance.

Reopening must not require re-analysis.

## Non-negotiable exit conditions

P3 cannot be declared complete if any of the following remain true:

- customer-facing metrics silently erase candidate/confidence semantics;
- review items cannot resolve to source-video evidence;
- completed games cannot be reopened without rerunning analysis;
- corrections overwrite model predictions instead of preserving history;
- game-player attribution errors cannot be corrected;
- physical clip failures are hidden;
- the product has not been exercised on representative real games with human review;
- review-time reduction has not been measured against a manual baseline.

---

# Execution order

P3 should be implemented as vertical product slices rather than building every backend component before exposing a review workflow.

Recommended order:

```text
P3.0  freeze P2 → P3 analytics/evidence contract
  ↓
P3.1  persist canonical Game Record
  ↓
P3.2  aggregate initial events + possessions
  ↓
P3.3  generate durable physical evidence clips
  ↓
P3.4  ship Things to Review + event detail playback
  ↓
P3.5  add confirm/correct/none + player correction
  ↓
P3.6  add conservative evidence-linked insights
  ↓
P3.7  add save/tag/note/collections
  ↓
P3.8  validate on representative real games and freeze exit metrics
```

## First end-to-end vertical slice

Before broadening all analytics, prove this flow on a real analyzed game:

```text
analysis completes
      ↓
open persistent Game Record
      ↓
Things to Review
      ↓
open turnover candidate
      ↓
play generated physical clip
      ↓
inspect confidence + reason shown
      ↓
confirm / correct / none
      ↓
correct game player if needed
      ↓
reopen the game later and see the same reviewed state
```

Once this path is reliable, expand the same contract to shots, passes, rebounds, loose balls, dunks and alley-oop candidates.

---

# P3 boundaries

## P3 owns

- P2 → P3 analytics contract;
- durable Game Record/read models;
- possession/event aggregation;
- confidence-preserving metrics;
- physical review clips;
- Things to Review;
- timeline/filter exploration;
- human confirmation/correction;
- game-player correction;
- reviewed feedback persistence;
- conservative evidence-linked insights;
- save/tag/note/collections;
- product usefulness/review-time validation.

## P3 does not own

- fixing P2 basketball event truth with UI heuristics;
- persistent real-world player identity across games;
- automatic roster resolution;
- tactical pattern classification such as P&R/transition/defensive coverage;
- longitudinal player development models;
- coach-to-player assignment/messaging workflow;
- mobile-first workflow;
- AI coaching judgments;
- uncontrolled online learning from user corrections.

---

# Relationship to later phases

P3 should deliberately create foundations that later phases can reuse:

```text
P3 Game Record + evidence + corrections
                 ↓
P4 tactical patterns
                 ↓
P4.5 persistent player identity
                 ↓
P5 mobile collaboration/review
                 ↓
P6 longitudinal development / AI Coach
```

The most important architectural consequence is that later insights should not need to invent a new evidence system. P3 establishes the durable chain from structured basketball claims back to the source footage.

---

# P3 completion statement

P3 is complete when CourtFlux can take a real uploaded basketball game and turn it into a durable, reopenable review record where a coach can quickly see what deserves attention, play the physical evidence for each important moment, understand uncertainty, correct the system when it is wrong, and organize useful review items without manually searching the full source video.
