# P4 Trajectory Engine v1

**Status:** complete for P4 Spatial Foundation task 7  
**Contract:** `p4.trajectory.v1`  
**Policy:** `p4.trajectory.policy.v1`  
**Dependency boundary:** P1/P4 spatial state only; no P2 or P3 input.

## Purpose

Task 7 converts per-frame canonical FIBA player observations into deterministic, auditable
track trajectories. It is a spatial reconstruction layer, not basketball understanding.

The engine consumes ordered:

```text
SpatialFrame
+
SpatialEligibilityObservation
        ↓
PlayerTrajectory[]
```

It never consumes possessions, ball handler, offense/defense roles, game-player identity,
P2 events, P3 derivations/review state, clips or tactical labels.

## Policy

All behaviour that can materially change a trajectory is explicit and versioned:

```text
smoothing_window_points
max_interpolation_gap_frames
max_step_speed_mps
accept_degraded_input
```

There are no hidden defaults in the contract object.

`max_step_speed_mps` is a trajectory outlier guard, not a basketball claim and not a
calibrated player-performance threshold. It must be supplied by the caller and remains
reprocessable.

## Point states

```text
OBSERVED
INTERPOLATED_GAP
INTERPOLATED_OUTLIER
GAP
OUTLIER_REJECTED
INVALID_INPUT
```

Every point also retains a reason:

```text
OBSERVED
TRACK_NOT_VISIBLE
METRIC_POSITION_MISSING
OUT_OF_BOUNDS_METRIC_POSITION
SPATIAL_INPUT_INVALID
DEGRADED_INPUT_REJECTED
TELEPORT_SPEED_EXCEEDED
INTERPOLATED_GAP
INTERPOLATED_OUTLIER
```

Raw position evidence is preserved separately from resolved and smoothed positions.

## Invalid and degraded spatial input

`SpatialEligibility.INVALID` is never interpolated across and never becomes metric truth.

`DEGRADED` input is accepted only when `accept_degraded_input=true`. Otherwise the point is
recorded as `INVALID_INPUT` with reason `DEGRADED_INPUT_REJECTED`.

This keeps trajectory behaviour downstream of the explicit Task 6 gate.

## Gap interpolation

Only short, bounded gaps may be filled.

A gap is interpolated when:

- it lies between two resolved metric positions;
- the number of consecutive unresolved frames is within `max_interpolation_gap_frames`;
- no frame in the gap is `INVALID_INPUT`;
- source timestamps are strictly increasing.

Interpolation is linear in source-video time, not frame number.

Unbounded gaps, long gaps and invalid-spatial spans remain explicit gaps.

## Teleport rejection

For observed points, the engine compares the candidate to the last accepted observed
position using source-video elapsed time.

If:

```text
distance / dt > max_step_speed_mps
```

the candidate raw position remains preserved but is rejected from the metric trajectory.

A short rejected outlier may later be replaced by bounded interpolation. Its raw position
is still available for diagnosis.

## Smoothing

Smoothing is a causal arithmetic mean over the most recent resolved points:

```text
raw / interpolated points
        ↓
causal window
        ↓
smoothed_position_m
```

It uses no future points. An unresolved point clears smoothing history so a later segment
does not silently bridge a real gap.

## Ordering and identity

Trajectory identity remains `track_id` only.

Input frames must have:

- strictly increasing `frame_id`;
- strictly increasing `timestamp_ms`;
- one matching eligibility observation per frame.

A trajectory begins at the first visible observation of a track and ends at its last
visible observation in the supplied sequence.

## Serialization

`PlayerTrajectory` is exact-schema serializable and contains:

```text
track_id
points[]
policy
source_spatial_contract_version
source_eligibility_contract_version
provenance
contract_version
```

Each `TrajectoryPoint` stores:

```text
track_id
frame_id
timestamp_ms
state
reason
raw_position_m
resolved_position_m
smoothed_position_m
observation_confidence
spatial_eligibility_state
provenance
```

## Deliberately deferred

Task 7 does not calculate:

- velocity;
- speed;
- acceleration;
- heading;
- stationary/accelerating classifications;
- player-player relationships;
- tactical actions.

Velocity and acceleration belong to Task 8. Semantic movement primitives remain Task 11.

## Task 7 exit gate

Task 7 is complete when:

- trajectories use canonical FIBA metric positions only;
- invalid spatial frames cannot become interpolated truth;
- degraded input is explicit policy;
- short gaps are deterministic and bounded;
- long/unbounded gaps remain unknown;
- teleport candidates preserve raw evidence;
- smoothing is causal and deterministic;
- sequence ordering is strict;
- contracts round-trip exactly;
- no P2/P3 imports or semantics exist.
