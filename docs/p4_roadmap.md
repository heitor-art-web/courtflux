# P4 — Tactical Intelligence Roadmap

P4 turns CourtFlux from an event-review product into a tactical analysis system that can explain how teams create and defend advantages while preserving evidence, uncertainty and reproducibility.

```text
P1 perception + court geometry
        ↓
P2 basketball understanding
        ↓
P3 persistent game record + possessions + evidence
        ↓
P4 tactical state
        ↓
spatial relationships + tactical primitives
        ↓
tactical segments + actions
        ↓
game / team tendencies
        ↓
coach review with evidence
```

P4 is not an AI coach. It observes, structures and explains tactical patterns. Recommendations about what a team or player *should* do belong to a later phase.

---

## Current status

**P4 is a future phase after P3 is sufficiently stable as a persistent, evidence-linked game review product.**

P4 must consume the canonical P1/P2/P3 state. It must not introduce a second detector, tracker, possession engine or game identity model merely to make tactical classification easier.

The implementation order is deliberately conservative:

```text
geometry
  ↓
normalized tactical state
  ↓
observable movement primitives
  ↓
temporal player relationships
  ↓
offensive tactical actions
  ↓
tactical segmentation
  ↓
tendencies
  ↓
defensive intelligence
```

The first tactical release is **offense-first**. Defensive intelligence remains part of P4, but it must not block useful offensive tactical analysis.

---

# Product question

> How did this team create or defend advantages, and can I immediately review every example?

The primary user is a **coach**.

The experience should remain understandable to players and other basketball users, but product decisions should optimize for a coach reviewing a real game rather than for a generic analytics dashboard.

The P4 magic moment is:

```text
Game analyzed
      ↓
Tactical Review
      ↓
"How did we create advantage?"
      ↓
Transition / P&R / drive / cut / screen / etc.
      ↓
open all examples
      ↓
video + tactical court evidence
      ↓
review without manually searching the game
```

Example product output:

```text
TEAM A — TACTICAL REVIEW

Transition
9 possessions
6 shots
[Review all]

Pick-and-roll
12 detected actions
8 roll
3 pop
1 unknown
[Review all]

Paint pressure
18 drive / paint-entry sequences
[Review]
```

P4 should help the coach understand *how the basketball happened*, not merely count isolated P2 events.

---

# P4 product principles

## 1. Observable before interpretive

CourtFlux should infer tactical meaning from observable spatial and temporal evidence rather than jump directly from player coordinates to basketball labels.

Prefer:

```text
players approach
screen proximity
handler changes path
screener moves toward rim
        ↓
PICK_AND_ROLL_CANDIDATE
```

instead of a black-box tactical label with no explainable intermediate evidence.

## 2. Attack first, defense second

Offensive tactical structure is the first P4 product target.

Initial offensive scope includes:

- transition;
- screen;
- pick-and-roll;
- roll;
- pop;
- reject;
- handoff;
- isolation;
- post-up;
- cut;
- drive;
- spot-up sequence;
- off-ball screen.

Defensive concepts such as help, rotation, closeout, switch, drop and hedge come later after spatial/matchup primitives are validated.

## 3. Tactical truth is temporal and multi-label

A possession must not be forced into one exclusive tactical class.

A real possession may contain:

```text
TRANSITION
    ↓
EARLY_OFFENSE
    ↓
PICK_AND_ROLL
    ↓
DRIVE
    ↓
PASS
    ↓
SPOT_UP_SHOT
```

P4 therefore stores tactical **segments and relationships over time**, with optional derived primary labels only for convenience.

## 4. `UNKNOWN` is a valid tactical result

When geometry, tracking, team attribution, possession context or tactical evidence is insufficient, P4 must abstain.

Supported review states include concepts such as:

```text
SUPPORTED
CANDIDATE
LOW_CONFIDENCE
UNKNOWN
NOT_TACTICALLY_ANALYZABLE
```

A plausible-but-wrong tactical label is worse than `UNKNOWN`.

## 5. Tactical analyzability is explicit

Not every frame or possession is suitable for tactical reasoning.

P4 must distinguish:

```text
basketball gameplay exists
        ≠
geometry is reliable enough for tactical analysis
        ≠
all required players/context are visible
```

Segments with inadequate spatial/context evidence may remain reviewable as P3 basketball events while being excluded from P4 metrics.

## 6. Evidence remains structural

Every tactical label or tendency must resolve back to supporting data and footage.

```text
team tendency
    ↓
tactical action(s)
    ↓
tactical segment(s)
    ↓
P3 possession(s)
    ↓
P2 event(s) / P1 state
    ↓
source-video evidence
```

The coach should be able to inspect the evidence behind the conclusion.

## 7. Human correction is a safety net, not the normal workflow

The intended experience is that CourtFlux is correct enough that the coach normally reviews and moves on.

Correction remains available when needed and becomes structured GT for evaluation and future model improvement.

The product must not require the coach to continuously annotate the game in order to obtain value.

## 8. Thresholds come from GT, not intuition

Distances, time windows, speed thresholds and confidence cutoffs must not be frozen because they seem visually reasonable on one clip.

The roadmap defines what must be measured. Numeric thresholds are calibrated using reviewed ground truth and frozen evaluation data.

---

# Phase boundary and dependencies

P4 depends on four upstream capabilities.

## P1 provides

- tracked player state;
- team evidence;
- court state;
- player court positions when geometry is valid;
- motion / speed evidence;
- confidence and provenance;
- camera / homography diagnostics.

P1 remains perception/state only and must not be polluted with tactical semantics.

## P2 provides

- ball-control state;
- possession acquisition/change;
- pass / turnover candidates;
- loose-ball lifecycle;
- shot attempts/outcomes;
- rebound candidates;
- game-level player attribution;
- confidence-aware basketball semantics.

P4 must not recreate these basketball primitives.

## P3 provides

- persistent Game Record;
- possessions / possession fragments;
- structured events;
- evidence clips;
- review state;
- corrections;
- team/game-player summaries;
- analysis provenance/versioning contract.

P4 tactical output becomes another evidence-linked layer of the same Game Record.

## P5 remains separate

Persistent real-world player identity moves to:

> **P5 — Persistent Player Identity**

P4 requires stable **game-level** player identity and stable team mapping, but it does not require a game player to be resolved to a real roster person such as `João Silva #7`.

---

# P4.0 — Tactical State Contract

Before tactical classifiers expand, define one canonical tactical state contract.

Conceptually:

```text
TacticalFrame

frame_id
timestamp_ms
possession_id

analysis_eligible
tactical_eligible
eligibility_reason

court_geometry
  source
  confidence
  reprojection_error
  homography_age
  projection_uncertainty

possession_context
  offense_team_id
  defense_team_id
  ball_handler_game_player_id
  ball_control_state
  possession_confidence

attack_direction
normalized_court_orientation

players[]
  game_player_id
  track_id
  team_id
  role = offense | defense | unknown
  court_xy
  velocity_xy
  speed_mps
  observation_confidence
  position_confidence

ball_context
  state
  handler_game_player_id
  planar_position_if_reliable
  position_semantics

provenance
```

The exact implementation may use several structures rather than one monolithic object, but these semantics must remain explicit.

## Tactical eligibility

P4 requires a stricter tactical gate than generic gameplay continuity.

A segment may be tactically eligible only when the relevant inputs are sufficiently supported, for example:

- gameplay is present;
- possession/team context is available when required;
- court orientation can be normalized;
- enough player positions are reliable;
- camera geometry is not invalid/stale beyond accepted bounds;
- the tactical action has sufficient visibility/context.

P4 should report why a segment was excluded.

## Attack direction is mandatory

Court coordinates alone are not enough for tactics.

P4 must determine or preserve which basket the offensive team is attacking and normalize possession coordinates so downstream logic can reason consistently.

Preferred normalized representation:

```text
own basket                         attacking basket
x = 0  --------------------------------------→  x = 28
```

The same tactical logic should work regardless of side, half or camera orientation.

Attack direction may be `UNKNOWN` when evidence is insufficient.

## Team mapping

Multi-game tendencies require a stable product-level team identity.

`home`, `away`, detector colour labels or match-local team classes are not sufficient as historical identifiers by themselves.

P4 multi-game aggregation must resolve game-local team evidence to the canonical team record used by the Game Record.

## Ball spatial semantics

A single court homography projects ground-plane points correctly but does not recover the real 3D position of an airborne ball.

Therefore P4 must distinguish:

```text
ball handler position
    = strong tactical ground-plane proxy

ball planar/apparent projection
    = limited spatial evidence

PASS_FLIGHT / SHOT_FLIGHT
    = do not treat projected ball XY as true physical ground position
```

Tactical logic should prefer handler/player geometry where appropriate.

### P4.0 exit gate

P4.0 is complete when tactical modules can consume one documented, versioned state contract with explicit team roles, normalized attack direction, geometry quality, player positions and tactical eligibility.

---

# P4.1 — Spatial Foundation

P4.1 hardens existing court geometry and player trajectories for tactical use.

CourtFlux already has the concept of image-to-court homography and player ground-plane positions. P4 must validate and stabilize this foundation rather than create a parallel spatial stack.

## Required spatial capabilities

- player court coordinates in metres;
- normalized offensive direction;
- smoothed player trajectories;
- velocity direction and speed;
- inter-player distances;
- distance to basket / paint / perimeter zones;
- team width and depth;
- team centroid / shape evidence;
- offensive/defensive spacing;
- paint/perimeter occupancy;
- geometry confidence and coverage.

## Geometry quality contract

P4 must retain enough information to distinguish geometry generated from different evidence quality.

Conceptually:

```text
DIRECT_CALIBRATION
OPTICAL_FLOW_HOLD_FRESH
OPTICAL_FLOW_HOLD_AGING
INVALID
```

Useful diagnostics include:

```text
homography_source
last_direct_calibration_ms
homography_age_frames
anchor_count
inlier_count
reprojection_rmse_px
projection_uncertainty_m
```

The exact confidence model should be calibrated rather than invented upfront.

## Homography drift

Small projection errors can materially change tactical conclusions involving distance.

For example:

- screen proximity;
- closeout distance;
- defensive help distance;
- spacing;
- player matchup relations.

P4 must therefore test spatial stability across camera pan/zoom/scene changes and abstain when geometry is not reliable enough.

## Court zones

Define canonical, versioned court zones for tactical reasoning, for example:

- paint;
- rim area;
- perimeter;
- corner;
- wing;
- top;
- backcourt/frontcourt;
- lane / central drive areas where useful.

Zone definitions must be basketball/court geometry, not raw image regions.

## Spatial snapshots

P4 should support tactical spatial snapshots that can later power radar review:

```text
SpatialSnapshot
  timestamp_ms
  possession_id
  player positions
  player roles
  team shapes
  ball-handler context
  geometry confidence
```

### P4.1 exit gate

P4.1 is complete when real-game player trajectories and court-relative spatial metrics are stable enough to evaluate tactical primitives against reviewed evidence, with explicit coverage and abstention for unreliable geometry.

---

# P4.2 — Tactical Primitives

P4.2 converts geometry/motion into small, observable tactical building blocks.

These primitives should be individually testable before they are composed into labels such as pick-and-roll or defensive coverage.

## Motion primitives

Examples:

```text
APPROACHING
SEPARATING
STATIONARY_WINDOW
ACCELERATING
DECELERATING
LATERAL_MOVEMENT
MOVING_TOWARD_BASKET
MOVING_AWAY_FROM_BASKET
PATH_CROSSING
```

## Offensive spatial primitives

Examples:

```text
PAINT_ENTRY
PAINT_EXIT
PERIMETER_RELOCATION
BALL_HANDLER_DRIVE_CANDIDATE
OFF_BALL_CUT_CANDIDATE
SCREEN_PROXIMITY
SCREEN_CONTACT_WINDOW_CANDIDATE
SPACING_COMPRESSION
SPACING_EXPANSION
```

## Relationship primitives

Examples:

```text
OFFENSE_PLAYER_PAIR_DISTANCE
DEFENDER_PROXIMITY
BALL_HANDLER_DEFENDER_PROXIMITY
SCREENER_DEFENDER_PROXIMITY
MATCHUP_CANDIDATE
NUMERICAL_ADVANTAGE
RECOVERY / STABILIZATION
```

`MATCHUP_CANDIDATE` is intentionally probabilistic. Nearest defender alone must not automatically become a claimed defensive assignment.

## Primitive output contract

Conceptually:

```text
TacticalPrimitive

primitive_id
game_id
possession_id
start_ms
end_ms
primitive_type
participant_game_player_ids[]
confidence
support_state
source_state_ids[]
evidence_ids[]
provenance
```

### P4.2 exit gate

P4.2 is complete when the spatial/motion primitives needed by transition, screen/P&R and common offensive actions can be evaluated independently against reviewed examples.

---

# P4.3 — Offensive Transition Intelligence

Transition is the first major tactical product capability because it builds naturally from possession change, player trajectories and attack direction.

## Detect and characterize

- transition opportunity;
- offensive transition;
- defensive transition context;
- fast-break candidate;
- numerical advantage/disadvantage;
- players ahead/behind the ball;
- early offense;
- time from possession acquisition/change to shot/action;
- possession stabilization into halfcourt.

## Phase semantics

P4 should support macro possession phases such as:

```text
TRANSITION
EARLY_OFFENSE
HALFCOURT
BROKEN_PLAY
UNKNOWN
```

These phases are temporal segments, not necessarily exclusive labels for an entire possession.

## Transition metrics

Examples:

```text
transition opportunities
transition attacks
transition shots
transition turnovers
average time to first action
numerical-advantage frequency
```

All rates require clearly defined eligible denominators.

### P4.3 exit gate

P4.3 is complete when CourtFlux can identify transition/early-offense windows with useful precision and evidence coverage on reviewed real games without forcing transition labels during ambiguous possession boundaries.

---

# P4.4 — Offensive Actions

P4.4 composes tactical primitives into basketball actions.

The initial required action set is:

```text
SCREEN
PICK_AND_ROLL
ROLL
POP
REJECT
HANDOFF
ISOLATION
POST_UP
CUT
DRIVE
SPOT_UP
OFF_BALL_SCREEN
```

## Screen semantics

A screen should be supported by spatial/temporal relationships rather than simple proximity.

Evidence may include:

- offensive player approaches relevant defender/action area;
- screener has a stationary or low-motion window;
- ball handler or off-ball teammate changes path around the screen area;
- relevant defenders react/change relation;
- participants remain consistent through the action window.

## Pick-and-roll as a composition

Conceptually:

```text
SCREEN_CANDIDATE
      +
BALL_HANDLER_PATH
      +
SCREENER_PATH
      +
DEFENDER_RELATION
      ↓
PICK_AND_ROLL_CANDIDATE
```

Then:

```text
screener moves toward basket
        ↓
ROLL

screener separates toward perimeter
        ↓
POP
```

`REJECT` is a handler-path outcome relative to the screen action and must not be guessed from one frame.

## Handoff

Handoff detection should combine:

- control/possession continuity;
- two offensive participants;
- close spatial interaction;
- handler transfer;
- player trajectories.

## Isolation

Isolation should be treated carefully because it is a tactical context, not merely “ball handler has space.”

Potential support includes:

- stable one-on-one matchup candidate;
- relative spacing from teammates;
- limited immediate screening/handoff context;
- sustained handler action window.

## Post-up

Post-up candidates require court location, offensive/defensive proximity and player movement/orientation proxies where available.

P4 must remain conservative where body orientation cannot be robustly inferred.

## Drive

Drive should primarily use normalized basket direction and ball-handler trajectory.

Useful evidence includes:

- controlled ball-handler state;
- meaningful movement toward attacking basket/paint;
- speed/path change;
- paint entry;
- defender relation.

## Cut

Cuts are off-ball movement events and should not be collapsed into generic player speed.

Support may include:

- off-ball status;
- movement relative to basket/paint;
- sharp route change;
- teammate/defender context;
- temporal duration.

## Spot-up sequence

A spot-up sequence may be derived from an offensive player maintaining useful perimeter spacing followed by reception/shot opportunity, with appropriate P2/P3 evidence.

## Off-ball screen

Off-ball screens reuse screen primitives but involve a non-handler target relationship.

### P4.4 exit gate

P4.4 is complete when the required offensive action set can be emitted with participants, time windows, confidence, evidence and `UNKNOWN`/candidate handling suitable for coach review.

---

# P4.5 — Tactical Segmentation

`P4.5` is now an internal P4 milestone.

Persistent Player Identity has moved to **P5**, so `P4.5` no longer represents identity work.

## Why segmentation is required

A possession can contain several tactical phases and actions.

CourtFlux must not reduce it to one forced class.

Conceptually:

```text
Possession 93

00:00–00:05  TRANSITION
00:05–00:09  EARLY_OFFENSE
00:09–00:15  PICK_AND_ROLL
00:15–00:18  DRIVE
00:18–00:21  SPOT_UP
```

## Tactical segment contract

```text
TacticalSegment

segment_id
game_id
possession_id
start_ms
end_ms

phase_labels[]
action_labels[]
participant_game_player_ids[]

confidence
support_state
coverage

evidence_ids[]
source_primitive_ids[]
provenance
```

Multiple compatible labels may coexist.

## Optional primary action

For summaries/search, P4 may derive:

```text
primary_action = PICK_AND_ROLL
```

but the canonical truth remains the temporal/multi-label segment history.

## Broken/unknown sequences

Use:

```text
BROKEN_PLAY
UNKNOWN
NOT_TACTICALLY_ANALYZABLE
```

when evidence supports those states. Do not invent a structured play simply to maximize classification coverage.

### P4.5 exit gate

P4.5 is complete when a possession can be represented as an evidence-linked tactical timeline rather than one forced tactical label.

---

# P4.6 — Game + Team Tendencies

P4.6 aggregates validated tactical actions/segments into tendencies.

Tendencies exist at two levels:

```text
Game tendency
Team tendency across multiple games
```

## Required denominator contract

A tendency must not be presented without its opportunity/sample context.

Prefer:

```text
P&R actions: 12
Eligible halfcourt possessions: 37
P&R usage: 32%
Coverage: 91%
```

instead of:

```text
P&R: 12
```

with no context.

Every tendency should preserve at least:

```text
count
eligible_opportunities
coverage
confidence / support mix
sample_size
game_ids[]
evidence_ids[]
analytics_version
```

## Initial offensive tendencies

Examples:

- transition opportunity/attack rate;
- transition shot generation;
- P&R usage;
- roll/pop/reject mix;
- handoff frequency;
- isolation frequency;
- post-up frequency;
- drive / paint-entry frequency;
- cut frequency;
- spot-up sequence frequency;
- off-ball screen frequency;
- team spacing/shape distributions where validated.

## Multi-game team identity

Cross-game team tendencies require stable canonical team mapping.

P4 must never merge team histories merely because local detector labels or home/away positions happen to match.

## Game-player tactical analytics

P4 may aggregate tactical involvement by game-level player inside a game.

Examples:

```text
Player 3
11 P&R ball-handler actions
7 drives
4 transition possessions
```

and:

```text
Player 5
8 screens
5 rolls
2 pops
```

Historical player trends across games are deferred until **P5 — Persistent Player Identity** can reliably map game-player entities to the same real person.

## Language discipline

P4 may report evidence-backed tendencies but should avoid unsupported coaching judgments.

Prefer:

```text
Team A used P&R in 12 of 37 eligible halfcourt possessions.
```

Avoid unsupported claims such as:

```text
Team A relies too much on P&R.
```

The latter belongs to a later recommendation/AI-coach layer and requires broader context.

### P4.6 exit gate

P4.6 is complete when game and multi-game team tendencies preserve denominators, coverage, sample size and evidence lineage and can be opened back into their contributing possessions/actions.

---

# P4.7 — Defensive Intelligence

Defensive intelligence comes after the offensive/spatial foundation is validated.

P4 should first model observable defensive relationships, then infer higher-level coverages.

## Defensive primitives

Initial concepts include:

- defender proximity;
- ball-handler defender candidate;
- matchup candidate;
- help-position candidate;
- defensive collapse;
- defensive recovery;
- closeout candidate;
- rim-protection presence;
- matchup-change candidate.

## Higher-level defensive actions

Only after the underlying relationships are validated should P4 attempt:

```text
SWITCH
DROP
HEDGE
HELP
ROTATION
```

These labels require temporal defensive context.

## Switch example

A switch cannot be defined merely as two defenders being close.

Conceptually:

```text
Defender A ↔ Offensive A
Defender B ↔ Offensive B
        ↓
action window
        ↓
Defender A ↔ Offensive B
Defender B ↔ Offensive A
        ↓
SWITCH_CANDIDATE
```

The matchup relation itself remains confidence-aware.

## Drop / hedge

Drop and hedge must be inferred relative to a supported screen/P&R context, screener/handler participants, basket direction and defender trajectories.

Without supported P&R context, P4 should not emit a coverage label merely because a defender occupies a familiar location.

## Defensive tendencies

Once individual defensive actions are validated, team tendencies may include:

- switch frequency in eligible screen actions;
- drop/hedge frequency;
- help/collapse frequency;
- closeout patterns;
- recovery timing;
- paint/rim occupancy patterns.

### P4.7 exit gate

P4.7 is complete when supported defensive concepts can be reviewed with participant/time/evidence context and ambiguous coverage remains `UNKNOWN` rather than forced.

---

# P4.8 — Tactical Review + Radar

P4 must expose tactical reasoning visually, not only as text labels.

## Primary tactical review surface

A tactical review item should support:

```text
Action / segment
participants
confidence / support state
why it was surfaced
video clip
court/radar view
source possession
correct / edit action when needed
```

Example:

```text
Pick-and-roll #7

Ball handler: Player 3
Screener: Player 5
00:31–00:42

Classification
P&R → Roll
Confidence: 78%

Why shown
- screen-proximity window
- handler path change
- screener movement toward rim

[Watch clip]
[View court]
[Correct]
```

## Radar / court view

P4 includes an explicit tactical court visualization.

Useful overlays may include:

- player positions;
- offense/defense role;
- player trails/trajectories;
- ball-handler marker;
- screen relation;
- drive/cut path;
- player-to-player distance when relevant;
- normalized attacking direction;
- tactical segment window;
- geometry quality indicator.

The radar is an explainability surface, not independent event truth.

## Video + radar synchronization

The ideal review experience synchronizes video time with tactical court state so the coach can inspect how the interpretation was derived.

## Degraded state

If court geometry is unreliable, the UI should clearly mark the radar/tactical overlay as unavailable or degraded rather than display false precision.

### P4.8 exit gate

P4.8 is complete when coaches can move from a tactical tendency/action directly to synchronized footage and spatial evidence without manually reconstructing the play.

---

# P4.9 — Human Correction + Tactical GT

Human correction extends the P3 review contract into tactical semantics.

## Product requirement

Correction must be easy to access but should not dominate the normal review UI.

The goal is:

```text
CourtFlux predicts correctly
      ↓
coach reviews
      ↓
no correction required
```

When CourtFlux is wrong:

```text
Wrong?
[Correct]
```

## Correctable fields

At minimum support correction of:

- tactical action/phase label;
- `NONE` / no relevant tactical action;
- ball-handler participant;
- screener participant;
- other action participants where applicable;
- offense/defense role where appropriate;
- tactical time window when necessary.

Example:

```text
Prediction
PICK_AND_ROLL_CANDIDATE
Player 3 + Player 5
confidence = 0.71

Human review
HANDOFF
Player 3 + Player 5
```

or:

```text
Human review
NONE
```

## Preserve prediction and correction separately

Never overwrite the original model output.

Persist conceptually:

```text
original_prediction
original_participants
original_confidence
original_provenance

human_label
human_participants
reviewer
timestamp
review_version
```

## Learning boundary

Corrections may feed:

- human GT;
- error analysis;
- calibration;
- threshold selection;
- future heuristic/model improvements;
- future training/fine-tuning.

They must not trigger uncontrolled online learning in production.

## Correction-rate product metric

One important product-quality metric is:

> What percentage of tactical review items can the coach accept without correction?

Correction volume itself is not the success metric.

### P4.9 exit gate

P4.9 is complete when reviewed tactical corrections are durable, auditable, reusable as GT and do not destroy original model provenance.

---

# P4.10 — Validation + Integrity Exit Gate

P4 is not complete merely because tactical labels render in the UI.

The full P2 → P3 → P4 analytical chain must be **versioned, traceable, correctable, measurable and reproducible**.

## Integrity requirements

### Versioned

For every derived output, CourtFlux can answer:

```text
Which runtime/model/schema/analytics version produced this?
```

### Traceable

For every tactical conclusion, CourtFlux can answer:

```text
Which tactical segments?
Which primitives?
Which possessions?
Which P2/P1 evidence?
Which timestamps/clips?
```

### Correctable

A human can correct a tactical action/participant when necessary without deleting the original prediction.

### Measurable

CourtFlux exposes the relevant:

```text
confidence
support state
coverage
eligibility denominator
sample size
validation metrics
abstention / unknown rate
```

### Reproducible

Stored analytical output retains enough input/version/provenance information to explain and, where supported, regenerate the result without silently changing historical meaning.

## End-to-end lineage example

If the product says:

```text
Team A
P&R usage: 38%
```

CourtFlux must be able to answer:

```text
Which games?
Which eligible possessions?
Which P&R actions?
Which tactical segments?
Which players?
Which timestamps?
Which evidence clips?
Which model/analytics version?
What coverage/sample size?
Were any predictions human-corrected?
```

If those questions cannot be answered, the tendency is not product-grade P4 output.

## Validation layers

P4 validation should include at least:

### Geometry validation

Measure:

- tactical-geometry coverage;
- direct vs held calibration coverage;
- reprojection quality;
- trajectory stability;
- projection drift;
- normalized court/orientation correctness;
- tactical abstention when geometry is insufficient.

### Primitive validation

Evaluate observable primitives independently before evaluating composite actions.

Examples:

- paint entry/exit;
- drive candidate;
- cut candidate;
- screen-proximity/contact window;
- numerical advantage;
- matchup/proximity relations.

### Offensive action GT

Evaluate each required action against reviewed human labels:

- transition;
- screen;
- P&R;
- roll/pop/reject;
- handoff;
- isolation;
- post-up;
- cut;
- drive;
- spot-up;
- off-ball screen.

Measure appropriate precision/recall/F1, participant correctness and temporal error/overlap.

### Tactical segmentation GT

Measure:

- phase/action label agreement;
- temporal overlap/error;
- multi-label handling;
- `UNKNOWN` / abstention quality;
- possession coverage.

### Defensive GT

Validate defensive primitives and higher-level coverages separately so errors can be attributed to the correct layer.

### Product validation

Measure whether the tactical review experience actually reduces manual video search/reconstruction.

Useful product measures include:

- time to reach evidence from a tendency/action;
- percentage of outputs with usable source evidence;
- percentage of tactical items accepted without correction;
- tactical review coverage per game;
- proportion marked `NOT_TACTICALLY_ANALYZABLE`;
- coach ability to find all examples of a tactical action without manual scrubbing.

## No intuition-only thresholds

The roadmap intentionally does not freeze numeric thresholds for screen distance, drive speed, tactical confidence or exit precision before reviewed GT exists.

The process is:

```text
frozen real-game evaluation set
        ↓
reviewed tactical GT
        ↓
measure errors / coverage
        ↓
calibrate thresholds
        ↓
re-run frozen evaluation
        ↓
freeze validated baseline
```

## P4 exit definition

P4 is complete when a coach can open a real game and reliably understand how the team created or defended advantages through evidence-linked tactical actions and tendencies, with direct video/radar review and transparent uncertainty.

Additionally, every production tactical conclusion must satisfy the integrity gate:

```text
VERSIONED
TRACEABLE
CORRECTABLE
MEASURABLE
REPRODUCIBLE
```

P4 does **not** require persistent real-world player names. That becomes P5.

---

# Suggested execution order

The intended implementation sequence is:

```text
P4.0 Tactical State Contract
        ↓
P4.1 Spatial Foundation
        ↓
P4.2 Tactical Primitives
        ↓
P4.3 Offensive Transition
        ↓
P4.4 Offensive Actions
        ↓
P4.5 Tactical Segmentation
        ↓
P4.6 Game + Team Tendencies
        ↓
P4.8 Tactical Review + Radar
        ↓
P4.9 Human Correction + Tactical GT
        ↓
P4.7 Defensive Intelligence
        ↓
P4.10 Validation + Integrity Exit Gate
```

The numeric phase names define ownership, not a requirement that implementation work be perfectly serial. Review/GT tooling should be introduced as soon as enough tactical output exists to evaluate it.

---

# Recommended vertical slices

## Slice 1 — Spatial trust

```text
real game
  ↓
normalized court positions
  ↓
player trajectories
  ↓
radar playback
  ↓
geometry coverage / confidence
```

No sophisticated tactical label is required yet.

## Slice 2 — Transition

```text
possession change
  ↓
attack direction + trajectories
  ↓
transition / early offense candidate
  ↓
evidence clip + radar
  ↓
human review
```

## Slice 3 — Screen / P&R

```text
screen primitives
  ↓
P&R composition
  ↓
roll / pop / reject
  ↓
participants + evidence
  ↓
review + correction
```

## Slice 4 — Offensive action set

Add handoff, isolation, post-up, cut, drive, spot-up and off-ball screen through the same evidence/GT contract.

## Slice 5 — Tactical timeline + tendencies

```text
possession
  ↓
multiple tactical segments
  ↓
game summary
  ↓
team tendencies
  ↓
open every contributing example
```

## Slice 6 — Defensive intelligence

Reuse validated spatial/matchup/action context to build defensive primitives and coverages.

---

# Persistence model

Exact database tables may evolve, but product semantics should support durable records roughly equivalent to:

```text
tactical_analysis_runs
spatial_snapshots
tactical_primitives
tactical_segments
tactical_action_participants
tactical_tendencies
tactical_corrections
```

Derived output must belong to the existing Game Record and preserve analysis/version lineage.

P4 should prefer idempotent materialization from a frozen upstream analysis snapshot rather than ad-hoc live aggregation that changes silently when code changes.

---

# Evidence model

Every tactical action/segment should be able to reference:

```text
source possession
source P2 events
relevant P1/tactical state window
source video timestamps
physical P3 review clip(s) where available
spatial/radar snapshots
```

Tendencies must reference the actions/segments that produced their aggregate.

---

# Test strategy

P4 should use layered tests rather than only end-to-end tactical labels.

## Unit / invariant tests

Examples:

- normalized attack direction is consistent after side reversal;
- distances remain symmetric;
- invalid geometry cannot emit metric-distance tactical truth;
- multi-label tactical segments preserve overlapping compatible labels;
- unknown team/player fields remain unknown;
- aggregate denominators exclude ineligible possessions correctly;
- human corrections never mutate original prediction records.

## Contract tests

Validate P1/P2/P3 → P4 schema/version compatibility.

## Golden tactical fixtures

Create frozen reviewed game windows for:

- transition;
- halfcourt stabilization;
- P&R roll/pop/reject;
- handoff;
- isolation;
- post-up;
- drive/cut;
- spot-up;
- off-ball screen;
- representative defensive coverage cases.

## End-to-end product tests

Validate:

```text
game record
→ tactical tendency
→ tactical action
→ clip/radar evidence
→ correction
→ persisted reviewed GT
```

---

# Non-goals for P4

P4 does not require:

- persistent real-world player identity across games;
- perfect Jersey OCR;
- player-name resolution;
- generic coaching recommendations;
- autonomous training plans;
- causal claims about why a team wins/loses;
- a conversational AI coach;
- live tactical advice during a game;
- forcing tactical labels onto every possession.

Persistent Player Identity/Jersey becomes **P5**.

Mobile becomes **P6**.

Continuous Development / AI Coach becomes **P7**.

---

# Final P4 progression

```text
P3
CourtFlux makes a game fast and useful to review.

P4
CourtFlux explains how the basketball is being played tactically.

P5
CourtFlux knows who the real player is across games.

P6
CourtFlux becomes part of the daily mobile workflow.

P7
CourtFlux uses longitudinal evidence to support continuous development.
```

P4 succeeds when tactical intelligence remains basketball-grounded: observable evidence first, interpretation second, and every conclusion reviewable at the source.