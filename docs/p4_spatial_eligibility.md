# P4 Spatial Eligibility + Abstention v1

**Status:** complete for P4 Spatial Foundation task 6  
**Contract:** `p4.spatial_eligibility.v1`  
**Policy:** `p4.spatial_eligibility.policy.v1`  
**Dependency boundary:** P1/P4 spatial evidence only; no P2 or P3 semantics.

## Purpose

Geometry can exist without being safe enough for every metric computation. Task 6 adds an explicit, reproducible gate between raw spatial evidence and later trajectories/relationships.

The gate answers only:

> Is this frame sufficiently supported for metric spatial processing, and if not, why?

It does not infer possession, offense/defense, ball handler, game-player identity or tactics.

## States

```text
ELIGIBLE
DEGRADED
INVALID
```

- `ELIGIBLE`: policy requirements are fully satisfied.
- `DEGRADED`: the frame remains usable, but evidence has a declared limitation.
- `INVALID`: downstream metric truth must abstain for this frame.

`DEGRADED` is intentionally distinct from `INVALID`: for example, an aging optical-flow hold or one missing player position may still support some later computations.

## Reasons

The frozen v1 reasons are:

```text
eligible
geometry_invalid
court_not_visible
coordinate_system_mismatch
metric_player_count_below_min
metric_player_fraction_below_min
aging_geometry
missing_metric_player_position
out_of_bounds_metric_position
```

Multiple reasons may be preserved on one observation. The evaluator never hides a secondary evidence problem just because another reason already made the frame invalid.

## Explicit policy

`SpatialEligibilityPolicy` requires callers to provide:

```text
min_metric_player_count
min_metric_player_fraction
court_extent_tolerance_m
allow_aging_geometry
require_court_visible
policy_version
```

There are no hidden operational defaults in the contract. The serialized policy accompanies every result so the decision is reproducible.

These are spatial support requirements, not basketball-accuracy thresholds.

## Geometry-quality integration

Task 6 consumes the Task 3 `GeometryQualityObservation` for the exact same frame/timestamp.

```text
GeometryQuality.INVALID
        ↓
SpatialEligibility.INVALID
```

`OPTICAL_FLOW_HOLD_AGING` is policy-controlled:

- when `allow_aging_geometry = false`, it yields `DEGRADED` if no stronger invalid reason exists;
- when true, aging alone does not prevent `ELIGIBLE`.

This avoids silently turning one freshness policy into universal spatial truth.

## Canonical coordinates

Metric player positions are accepted only when the `SpatialFrame` declares:

```text
p4.fiba_28x15_xy.v1
```

Visible player positions are checked against the canonical 28 × 15 m court extent using the policy's explicit tolerance.

A provided out-of-bounds point does not count toward valid metric coverage and is recorded as `out_of_bounds_metric_position`.

## Player coverage

Only visible P1 player observations participate in coverage counts.

The result records:

```text
visible_player_count
metric_player_count
missing_metric_player_count
out_of_bounds_metric_player_count
metric_player_fraction
```

If configured count/fraction minimums are not met, the frame becomes `INVALID` and later metric layers must abstain.

If minimums still pass but some visible players are missing metric positions or have out-of-bounds positions, the frame is `DEGRADED`.

## Fail-closed rules

A frame is invalid when any hard requirement fails, including:

- geometry quality is `INVALID`;
- the court must be visible but is not;
- coordinate-system lineage is not canonical P4 FIBA v1;
- metric player count is below configured minimum;
- metric player fraction is below configured minimum.

Invalid frames are not repaired, clamped or silently promoted.

## No confidence invention

Task 6 does not introduce a synthetic numeric spatial-confidence score and does not threshold P1 observation confidence. Calibration of confidence-aware acceptance belongs to later empirical validation.

The gate operates on explicit structural evidence and explicit caller policy only.

## Dependency boundary

Allowed inputs:

```text
SpatialFrame
GeometryQualityObservation
canonical P4 FIBA coordinate helpers
SpatialEligibilityPolicy
```

Forbidden inputs:

```text
P2 events
possessions
ball handler
Game Player Entity
P3 derivations/reviews/clips
attack direction
offense/defense
tactical labels
```

## Task 6 exit gate

Task 6 is complete when:

- states and reasons are versioned;
- policy is strict, explicit and serialized;
- geometry-quality/frame lineage must match exactly;
- invalid geometry fails closed;
- canonical coordinate-system lineage is enforced;
- metric player coverage is measurable;
- partial evidence can be `DEGRADED` rather than fabricated as complete;
- aging geometry is explicitly policy-controlled;
- no P2/P3 imports or semantics exist;
- round-trip serialization and failure cases are covered by tests.
