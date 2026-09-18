# P4 Kinematics Engine v1

**Status:** complete for P4 Spatial Foundation task 8  
**Contract:** `p4.kinematics.v1`  
**Policy:** `p4.kinematics.policy.v1`  
**Dependency boundary:** P4 trajectories only; no P2 or P3 input.

## Purpose

Task 8 derives metric motion quantities from Task 7 trajectories:

```text
PlayerTrajectory
      ↓
smoothed_position_m
      ↓
velocity_xy
speed_mps
acceleration_xy
acceleration_magnitude_mps2
heading_rad
```

The engine is intentionally mathematical. It does not label a player as driving,
cutting, screening, accelerating, stationary, offensive or defensive.

## Input

Kinematics consumes `PlayerTrajectory` and uses only:

```text
TrajectoryPoint.smoothed_position_m
frame_id
timestamp_ms
trajectory point state
```

Raw image positions are not used.

## Policy

The explicit versioned policy is:

```text
max_derivative_gap_ms
use_interpolated_points
```

`max_derivative_gap_ms` prevents a derivative from being formed across a temporal gap that
the caller considers too large.

`use_interpolated_points` controls whether Task 7 interpolated trajectory positions may
participate in derivatives. There is no hidden choice.

## Derivatives

Velocity uses causal finite differences:

```text
v(t) = (p(t) - p(previous)) / dt
```

Speed is:

```text
sqrt(vx² + vy²)
```

Acceleration is the difference between consecutive velocity estimates divided by elapsed
source-video time.

Heading is:

```text
atan2(vy, vx)
```

in radians in the canonical FIBA XY axes.

When speed is exactly zero, heading remains `null`; a direction is not invented.

## States

Each kinematics point is:

```text
UNAVAILABLE
VELOCITY_ONLY
FULL
```

`UNAVAILABLE` contains no derivative values.

`VELOCITY_ONLY` has velocity/speed and optional heading, but does not yet have a valid
acceleration estimate.

`FULL` contains both velocity and acceleration evidence.

## Gap behaviour

An unresolved trajectory point resets derivative continuity.

Therefore:

```text
valid position
gap
valid position
```

does not produce a velocity across the gap.

A usable point after the gap becomes the new derivative anchor. The following usable
point may then produce velocity.

The same reset occurs when an interpolated point is disallowed by policy.

## No semantic thresholds

Task 8 deliberately does not define:

```text
STATIONARY
ACCELERATING
DECELERATING
SPRINTING
LATERAL_MOVEMENT
```

Those are movement interpretations and belong to Task 11 — Generic Motion Primitives.

Task 8 only exposes measured kinematic quantities.

## Serialization

`PlayerKinematics` is exact-schema serializable and contains:

```text
track_id
points[]
policy
source_trajectory_contract_version
provenance
contract_version
```

Each point retains its source `trajectory_point_state`, making interpolation/gap lineage
visible to downstream consumers.

## Units

```text
position        metres
velocity        metres / second
speed           metres / second
acceleration    metres / second²
heading         radians
time            source-video milliseconds
```

## Task 8 exit gate

Task 8 is complete when:

- derivatives use smoothed canonical FIBA positions;
- velocity and speed are deterministic;
- acceleration requires consecutive supported velocities;
- heading follows canonical XY orientation;
- stationary points do not invent heading;
- large temporal gaps cannot create derivatives;
- interpolated inputs are explicitly policy-controlled;
- contracts round-trip exactly;
- no semantic movement labels are emitted;
- no P2/P3 imports or semantics exist.
