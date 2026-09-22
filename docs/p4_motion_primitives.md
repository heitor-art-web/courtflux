# P4 Generic Motion Primitives

Task 11 of the P4 Spatial Foundation freezes small, observable motion primitives that can later be composed into tactical semantics without importing P2/P3 truth.

## Contracts

```text
p4.motion_primitives.v1
p4.motion_primitives.policy.v1
```

The primitive layer is intentionally below basketball semantics:

```text
trajectory + kinematics + neutral spatial relationships
                         ↓
generic motion primitives
                         ↓
future tactical composition
```

It does **not** infer possession, ball handler, offense/defense, screens, cuts, drives, matchups or attack direction.

## Supported primitives

```text
APPROACHING
SEPARATING
STATIONARY_WINDOW
ACCELERATING
DECELERATING
LATERAL_MOVEMENT
MOVING_TOWARD_TARGET
MOVING_AWAY_FROM_TARGET
PATH_CROSSING
```

### Pair radial motion

`APPROACHING` and `SEPARATING` consume the neutral pair relationship from `p4.spatial_relationships.v1`.

A positive `closing_speed_mps` means the distance between the two tracks is shrinking. A negative value means it is increasing.

The classifier does not interpret the pair as offense/defense or infer a matchup.

### Stationary window

`STATIONARY_WINDOW` requires a caller-supplied minimum duration and maximum speed. Both values are part of the serialized policy.

The primitive is descriptive only. It does not claim a screen, post-up, spot-up, dead-ball state or tactical role.

### Acceleration / deceleration

The classifier uses tangential acceleration:

```text
dot(velocity, acceleration) / speed
```

This distinguishes increasing from decreasing speed without relying on absolute court direction.

### Lateral movement

`LATERAL_MOVEMENT` means the velocity is dominated by the canonical FIBA `y` axis relative to the longitudinal `x` axis.

This is **not** attack-relative lateral movement and does not infer which basket a team is attacking.

### Generic target motion

`MOVING_TOWARD_TARGET` and `MOVING_AWAY_FROM_TARGET` require an explicit caller-supplied metric court point.

The target can later be a basket, landmark or tactical point once the caller has the required context. The primitive layer itself does not choose the target.

### Path crossing

`PATH_CROSSING` compares two track path segments over the same time interval in canonical FIBA court space.

It is a geometric relationship only. It is not an off-ball screen, cut, exchange or defensive switch.

## Explicit policy

No motion threshold is hidden in the module. `MotionPrimitivePolicy` requires:

```text
pair_radial_speed_min_mps
stationary_speed_max_mps
stationary_window_min_ms
tangential_acceleration_min_mps2
movement_speed_min_mps
lateral_dominance_ratio
target_radial_speed_min_mps
path_crossing_epsilon_m
```

These values are structural runtime parameters, not validated basketball thresholds. Final numeric calibration remains a later validation concern.

## Evidence contract

Each primitive preserves:

```text
primitive_type
start_ms
end_ms
track_ids
target_m (optional)
support
provenance
contract_version
```

`support` stores the observable quantities behind the primitive, including the policy version used.

## Dependency boundary

The module imports only P4 spatial layers:

```text
p4.kinematics.v1
p4.spatial_relationships.v1
```

It has:

```text
P2 dependency: false
P3 dependency: false
attack-direction inference: false
tactical semantics: false
```

This is deliberate. Future P4 tactical modules may compose these primitives with validated basketball context without changing the underlying motion evidence.

## Validation

Task 11 tests cover:

- approaching vs separating;
- abstention below radial threshold;
- stationary duration;
- accelerating vs decelerating;
- canonical-axis lateral motion;
- caller-supplied target toward/away motion;
- path crossing vs parallel paths;
- strict policy validation;
- serialization round-trip;
- P2/P3 import boundary.

Spatial GT and measured threshold calibration remain outside the Spatial Foundation, in the later P4 Spatial Validation tranche.
