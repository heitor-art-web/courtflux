# P4 Spatial Contract v1

**Status:** complete for P4 Spatial Foundation task 1  
**Contract:** `p4.spatial_frame.v1`  
**Dependency boundary:** P1 only; no P2 or P3 input is permitted.

## Purpose

The P4 Spatial Foundation needs one stable envelope for court geometry and tracked-player observations before trajectories, kinematics, tactical eligibility or tactical semantics are added.

The contract deliberately answers only:

> What spatial evidence did P1 produce for this source-video frame?

It does **not** answer who has possession, who is attacking, which player is a game-level entity, or what tactical action occurred.

## Source boundary

Allowed source evidence is canonical P1 state:

```text
FrameState
  ├── timestamp / frame id
  ├── homography matrix
  ├── CourtState
  └── PlayerState[]
          ↓
    SpatialFrame
```

The adapter ignores P2/P3/product fields even if they exist on `FrameState`.

Forbidden contract semantics include:

- possession id;
- game-player id;
- ball-handler identity;
- ball-control state;
- offense/defense team roles;
- P2 event rows;
- P3 derivations/review items/clips;
- tactical action labels.

`local_team_id` is permitted only as optional **P1-local team evidence**. It is not a canonical historical team identity and does not imply offense or defense.

## Contract objects

### `SpatialFrame`

```text
frame_id
timestamp_ms
geometry
players[]
source_runtime_version
provenance
contract_version
```

Time is canonical source-video milliseconds. `frame_id` and `timestamp_ms` are non-negative integers.

### `SpatialGeometryEvidence`

```text
geometry_valid
court_visible
homography_source
reprojection_rmse_px
homography_matrix
court_polygon_px
court_coordinate_system
projection_uncertainty_m
homography_age_frames
provenance
```

Task 1 preserves geometry evidence but does **not** invent geometry-quality classes or thresholds. `projection_uncertainty_m` and `homography_age_frames` are explicitly nullable until later Spatial Foundation tasks provide them.

The current upstream court coordinate label is:

```text
p1.runtime_metric.v1
```

P1 already projects image points into FIBA court metres. The canonical FIBA 28 × 15 coordinate convention and transforms are frozen separately in the Court Coordinate System task rather than being silently inferred here.

### `SpatialPlayerObservation`

```text
track_id
local_team_id
local_team_confidence
image_center_px
court_position_m
observation_confidence
observation_uncertainty
position_confidence
visible
source
provenance
```

Identity is intentionally limited to `track_id`. `position_confidence` is nullable because P1 observation confidence must not be silently reinterpreted as metric projection confidence.

Canonical P4 velocity, acceleration, smoothing and trajectories are not stored in contract v1; those are derived by later Spatial Foundation layers.

## Validation rules

The v1 contract is strict:

- serialized objects must contain the exact frozen field set;
- unknown values use explicit `null`;
- booleans are not coerced from strings or integers;
- ids and timestamps are non-negative integers;
- probabilities must be finite and within `[0, 1]`;
- coordinates and homography values must be finite;
- reprojection/projection errors and homography age cannot be negative;
- track ids must be unique inside one frame;
- provenance must be JSON serializable;
- the contract version must match exactly.

Invalid values are rejected instead of clamped or repaired.

## Machine-readable manifest

`p4_spatial_contract_manifest()` exposes:

- frozen field sets;
- time and coordinate units;
- identity/team scope;
- forbidden semantics;
- explicit `p2_dependency = false`;
- explicit `p3_dependency = false`;
- explicit `tactical_labels = false`.

This is intended for contract tests and future persistence/materialization compatibility checks.

## P1 adapter

`services.p4.spatial_adapter.spatial_frame_from_p1()` creates the v1 contract from P1 state.

It consumes only:

- frame id/time;
- homography/court polygon;
- `CourtState`;
- `PlayerState[]`;
- optional local P1 team evidence supplied by track id.

Changing `p2_events`, `ball_handler` or `offensive_team` on the input frame does not change the spatial output. This invariant is covered by tests.

## Deliberately deferred

The following are **not** part of task 1:

- database persistence — task 2;
- geometry-quality state machine — task 3;
- canonical FIBA transforms/orientation — task 4;
- court zones — task 5;
- spatial eligibility thresholds/reasons — task 6;
- smoothing/trajectory/kinematics — later Spatial Foundation tasks;
- any P2/P3/tactical integration.

This keeps the first P4 contract structurally stable while leaving downstream semantics independently testable and calibratable.
