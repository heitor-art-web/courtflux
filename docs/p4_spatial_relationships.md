# P4 Spatial Relationship Metrics v1

**Status:** complete for P4 Spatial Foundation task 9  
**Contract:** `p4.spatial_relationships.v1`  
**Dependency boundary:** P1/P4 only; no P2/P3 or tactical semantics.

## Purpose

Task 9 turns canonical per-player P4 position/velocity evidence into deterministic
relationship measurements that later motion/tactical layers can reuse.

It intentionally answers only questions such as:

- how far apart are two tracked players?
- what is their relative velocity?
- are they currently closing or separating in the radial sense?
- which tracked player is nearest under an explicit local-team filter?
- how far is a tracked player from a selected physical basket or court target?

It does **not** decide who is offense/defense, who is guarding whom, whether a screen exists,
or whether a matchup is real.

## Neutral actor state

`SpatialActorState` contains:

```text
track_id
timestamp_ms
position_m
velocity_mps | null
local_team_id | null
provenance
```

`local_team_id` remains P1-local evidence. It is not a persistent team identity and never
implies offense/defense.

Positions must lie inside the canonical FIBA 28 × 15 m court.

`actor_state_from_kinematics()` provides the normal bridge from `p4.kinematics.v1`.

## Pair relationship

`PairwiseSpatialRelationship` stores:

```text
timestamp_ms
track_a_id
track_b_id
distance_m
delta_position_m
local_team_relation
relative_velocity_mps
closing_speed_mps
provenance
contract_version
```

`local_team_relation` is one of:

```text
same_local_team
different_local_team
unknown
```

This is deliberately weaker than `teammate`, `opponent`, `defender`, or `matchup`.

## Closing-speed sign

For relative position `r = B - A` and relative velocity `v = vB - vA`:

```text
radial_rate = dot(r, v) / |r|
closing_speed = -radial_rate
```

Therefore:

```text
closing_speed > 0  -> distance is decreasing
closing_speed < 0  -> distance is increasing
closing_speed = 0  -> no instantaneous radial change
```

If velocity is unavailable, closing speed remains `null`. If both players occupy the same
point, radial closing speed is undefined and remains `null`.

No threshold converts this metric into `APPROACHING` or `SEPARATING`; that belongs to Task 11.

## Deterministic helpers

Task 9 provides:

```text
distance_between()
distance_to_target()
distance_to_physical_basket()
pair_relationship()
pairwise_relationships()
pairwise_distance_matrix()
nearest_actor()
```

Pair enumeration and nearest-player tie breaking are deterministic by `track_id`.

## Physical basket semantics

Distance-to-basket requires an explicit `CourtEnd.LEFT` or `CourtEnd.RIGHT`.

The helper does not infer:

```text
attacking basket
own basket
offense
defense
```

Those concepts remain gated by validated basketball context later in P4.

## No hidden policy

Relationship metrics are geometry/math rather than a classifier. Task 9 therefore adds no
distance, speed, matchup, or tactical threshold.

Calibration of concepts such as screen proximity, defender proximity or numerical advantage
belongs to later GT-backed tactical work.

## Dependency boundary

The module may consume:

```text
p4.kinematics.v1
canonical FIBA court geometry
optional P1-local team labels
```

It must not consume:

```text
P2 events
possessions
ball handler
offense / defense roles
game-player identity
P3 derivations / review / clips
tactical labels
```

## Task 9 exit gate

Task 9 is complete when:

- pairwise distance is symmetric and metric;
- relative vectors preserve canonical XY orientation;
- closing-speed sign is documented and tested;
- missing velocity remains explicit;
- local-team relation remains non-tactical;
- nearest/pairwise outputs are deterministic;
- physical basket distance requires an explicit end;
- serialization round-trips exactly;
- P2/P3 import boundaries remain clean.
