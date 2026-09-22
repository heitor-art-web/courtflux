# P4 Canonical FIBA Court Coordinate System v1

**Status:** complete for P4 Spatial Foundation task 4  
**Contract:** `p4.fiba_court.v1`  
**Coordinate system:** `p4.fiba_28x15_xy.v1`  
**Dependency boundary:** P1/P4 only; no P2 or P3 context is required.

## Purpose

P4 needs one metric court convention before trajectories, zones, distances and radar visualization can be compared across frames. Task 4 freezes that convention rather than letting downstream modules infer orientation from whichever calibration produced a frame.

## FIBA geometry baseline

CourtFlux v1 uses the full-size FIBA court geometry:

```text
length = 28.00 m
width  = 15.00 m
```

The canonical marking constants also preserve:

```text
basket centre from endline = 1.575 m
free-throw line from endline = 5.80 m
restricted-area half width = 2.45 m
3-point arc radius = 6.75 m
3-point straight line from sideline = 0.90 m
no-charge semi-circle radius = 1.30 m
```

These values are versioned in code. CourtFlux does not silently change historical spatial truth if a future rules edition changes a marking.

Reference baseline: FIBA Official Basketball Rules, Article 2 / Diagram 1.

## Axes and origin

```text
upper sideline, y = 15
(0,15) ---------------------------------------- (28,15)
   |                                                |
   |                 (14, 7.5)                     |
   |                                                |
(0,0)  ---------------------------------------- (28,0)
left endline                                  right endline
lower sideline, y = 0
```

The origin is the intersection of the inner edges of the left endline and lower sideline.

```text
+x : left endline → right endline
+y : lower sideline → upper sideline
units : metres
```

This orientation is physical/canonical. It does not imply which team is attacking.

## P1 → P4 bridge

The existing P1 calibration map already emits 28 × 15 metre coordinates with the same numeric orientation. P4 therefore supports exactly:

```text
p1.runtime_metric.v1
        ↓ identity transform + validation
p4.fiba_28x15_xy.v1
```

The transform is explicit even though it is currently numerically identical. If the P1 convention ever changes, P4 will reject an unknown source coordinate-system label instead of silently reinterpreting old values.

`services.p4.spatial_adapter.spatial_frame_from_p1()` now stores the canonical P4 coordinate-system label on `SpatialGeometryEvidence` and preserves both upstream and canonical labels in provenance.

## Core transforms

Task 4 provides pure geometry helpers for:

- extent validation;
- unit normalization / denormalization;
- longitudinal mirror;
- lateral mirror;
- explicit longitudinal-direction normalization;
- basket centres;
- distance to either physical basket;
- physical-endline-relative coordinates.

Direction normalization requires the caller to supply either `LEFT_TO_RIGHT` or `RIGHT_TO_LEFT`. P4 does not infer attack direction in this task.

## Boundary storage policy

`within_court_extent()` treats the metric rectangle `[0,28] × [0,15]` as storable court geometry, including coordinates exactly on the lines.

That is a storage/geometry rule, not an official in/out basketball ruling. Later eligibility/rules logic can decide how to treat boundary-line observations without changing the coordinate system itself.

## Symmetry guarantees

The contract tests require:

```text
mirror_longitudinal(mirror_longitudinal(p)) == p
mirror_lateral(mirror_lateral(p)) == p
from_end_relative(to_end_relative(p, end), end) == p
```

This gives later zones and radar rendering a stable symmetry primitive.

## Dependency boundary

Allowed inputs:

```text
P1 metric court coordinates
explicit physical court end/direction arguments
FIBA v1 constants
```

Forbidden inputs:

```text
possession
ball handler
offense / defense
P2 events
P3 derivations/reviews/clips
tactical actions
```

The manifest therefore records:

```text
p2_dependency = false
p3_dependency = false
direction_inference = false
tactical_semantics = false
```

## Task 4 exit gate

Task 4 is complete when:

- FIBA 28 × 15 m is frozen as the v1 court;
- origin and axis directions are explicit;
- the P1→P4 coordinate bridge is explicit/versioned;
- numeric inputs fail closed;
- extent validation is deterministic;
- mirrors and end-relative transforms are reversible;
- attack direction is never inferred;
- no P2/P3 imports or semantics exist;
- task 5 can define zones entirely in this coordinate system.
