# P4 Spatial Snapshot Materialization

**Status:** complete for P4 Spatial Foundation task 12  
**Contract:** `p4.spatial_snapshot.v1`  
**Source contract:** `p4.spatial_frame.v1`  
**Dependency boundary:** P1/P4 only; no P2/P3/tactical semantics.

## Purpose

`SpatialFrame` is the frozen P1→P4 source envelope. It intentionally contains only source observations and geometry evidence.

`SpatialSnapshot` is different: it is the **materialized, timestamp-aligned P4 spatial state** built from the source frame plus the spatial foundation layers that have already been computed.

```text
SpatialFrame
    ↓
Geometry Quality
    ↓
Spatial Eligibility
    ↓
Trajectory + Kinematics
    ↓
Spatial Relationships
    ↓
Team Shape
    ↓
Generic Motion Primitives
    ↓
SpatialSnapshot
```

This snapshot is the structure that the internal radar/court viewer can consume next and, later, the structure that validated basketball context can enrich into a tactical frame.

## Contract

`p4.spatial_snapshot.v1` stores:

```text
frame_id
timestamp_ms
source_runtime_version
source_spatial_contract_version

geometry
geometry_quality
spatial_eligibility

players[]
relationships[]
team_shapes[]
motion_primitives[]

provenance
contract_version
```

The source spatial contract remains `p4.spatial_frame.v1`. The materialized snapshot has its own version because derived spatial state must not be hidden inside `SpatialFrame.provenance` or silently change the meaning of the frozen P1→P4 source contract.

## Player state

Each snapshot player preserves the available spatial stages:

```text
track_id
local_team_id
visible
image_position_px
court_position_m
smoothed_position_m
trajectory_state
kinematics_state
velocity_mps
speed_mps
observation_confidence
position_confidence
provenance
```

The distinctions are intentional:

- `court_position_m` is raw source P1 metric evidence;
- `smoothed_position_m` is the trajectory position used by spatial relationships;
- velocity/speed come only from the kinematics layer;
- an interpolated trajectory point can remain present even when the source frame has no player observation for that track.

## Derived collections

### Relationships

Pairwise relationships are materialized from `smoothed_position_m`. No relationship is renamed to `defender`, `matchup`, `screener` or another basketball role.

### Team shape

Team shape is calculated only from P1 `local_team_id`, so centroid, width/depth, spacing, hull and physical-end occupancy can be stored without claiming which team is offense or defense.

### Motion primitives

Only generic primitives active at the snapshot timestamp are included. They remain `APPROACHING`, `SEPARATING`, `STATIONARY_WINDOW`, `ACCELERATING`, `DECELERATING`, `LATERAL_MOVEMENT`, `MOVING_TOWARD_TARGET`, `MOVING_AWAY_FROM_TARGET` and `PATH_CROSSING`.

No drive, cut, screen, P&R, closeout or other tactical label is created by materialization.

## Eligibility and abstention

`INVALID` eligibility is a hard boundary.

A snapshot with `spatial_eligibility.state = INVALID` may preserve raw source evidence such as image position and court projection for diagnosis, but it may not expose derived metric truth:

```text
smoothed_position_m
velocity / speed
relationships
team_shapes
motion_primitives
```

must be absent/empty.

This is enforced in the Python contract and again at the persistence boundary for derived collections.

`DEGRADED` remains policy-controlled upstream. The snapshot records the result; it does not invent a second eligibility policy.

## Persistence

Task 2 already created:

```text
p4_spatial_runs
p4_spatial_snapshots
p4_spatial_artifacts
```

Task 12 makes `p4_spatial_snapshots` store `p4.spatial_snapshot.v1` while the parent run remains rooted in `p4.spatial_frame.v1`.

The database checks that:

- snapshot payload contract matches the indexed snapshot contract;
- `source_spatial_contract_version` matches the parent spatial run;
- frame/timestamp indexes match the payload;
- geometry quality and eligibility refer to the same frame/time;
- player count matches the payload;
- geometry/court visibility indexes match the payload;
- derived collections are arrays;
- `INVALID` snapshots do not persist relationship/team-shape/motion-primitive truth.

Dense `analysis_artifacts` may contain either raw `p4.spatial_frame.v1` records or materialized `p4.spatial_snapshot.v1` records, with the content contract recorded explicitly.

## No basketball semantics

The snapshot contains no possession, game-player identity, ball handler, ball-control state, offense/defense role, attack-direction inference, tactical action or P3 review/clip state.

Later tactical P4 may combine validated P2/P3 basketball context with `p4.spatial_snapshot.v1` without rebuilding the spatial foundation.

## Production status

The P4 persistence migration remains source-controlled on `p4_spatial_foundation` and is **not applied to production by this task**.

Because the P4 migration has not been rolled out to production, task 12 updates that branch migration in place so the first eventual deployment creates the final foundation schema directly.
