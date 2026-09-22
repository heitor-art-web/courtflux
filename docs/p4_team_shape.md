# P4 Team Shape Metrics v1

**Status:** complete for P4 Spatial Foundation task 10  
**Contract:** `p4.team_shape.v1`  
**Dependency boundary:** P1/P4 only; team grouping uses P1-local team evidence.

## Purpose

Task 10 aggregates neutral `SpatialActorState` observations into descriptive team geometry
without introducing offense/defense, possession, matchup, or tactical labels.

The output is intended to support later spacing and motion primitives, radar diagnostics and
eventually tactical reasoning once validated basketball context is available.

## Team scope

A shape is keyed by:

```text
local_team_id
timestamp_ms
track_ids[]
```

`local_team_id` is explicitly local P1 evidence. It is not a canonical historical team id.

Actors with no local-team label are not silently assigned to a team.

## Shape metrics

`TeamShapeMetrics` contains:

```text
player_count
centroid_m

width_m
depth_m
bounding_box_m

convex_hull_m
convex_hull_area_m2

mean_pairwise_spacing_m
pairwise_spacing_variance_m2
min_pairwise_spacing_m
max_pairwise_spacing_m

occupancy_by_physical_end[]
```

### Width and depth

Before attack direction exists:

```text
width = max(y) - min(y)
depth = max(x) - min(x)
```

These are **canonical physical court axes** only.

They must not be described as offensive width/depth or frontcourt/backcourt depth.

## Convex hull

Task 10 uses a deterministic monotonic-chain hull over canonical metric points.

For fewer than three non-collinear points, area is naturally `0.0`.

This does not imply that the team occupies zero meaningful basketball space; it only reports
the planar polygon area supported by the available points.

## Pairwise spacing

Spacing is calculated from Task 9 pair relationships.

For one-player shapes, pairwise spacing fields remain `null` rather than inventing a value.

No threshold labels the shape as compressed, expanded, good spacing, bad spacing, etc.
Those are later observable/tactical primitives.

## Paint and perimeter occupancy

Because attack direction is deliberately unavailable in the Spatial Foundation, occupancy is
reported for **both physical basket ends**:

```text
left physical basket
right physical basket
```

For each end the contract records:

```text
paint_count
rim_area_count
perimeter_count
other_half_count
paint_fraction
perimeter_fraction
```

This preserves useful geometry without claiming which basket a team is attacking.

After validated P2/P3 context exists, tactical P4 may select the appropriate physical end and
derive attack-relative occupancy without rebuilding these metrics.

## Deterministic helpers

Task 10 provides:

```text
convex_hull()
polygon_area()
compute_team_shape()
compute_all_team_shapes()
```

`compute_all_team_shapes()` groups only actors with a known `local_team_id` and returns teams
in deterministic id order.

## Dependency boundary

Task 10 may consume:

```text
p4.spatial_relationships.v1
canonical FIBA court zones
P1-local team evidence
```

It must not consume:

```text
P2 possession / events
ball handler
offense / defense roles
game-player identity
P3 Game Record / review / clips
attack direction inference
tactical labels
```

## Task 10 exit gate

Task 10 is complete when:

- centroid, width and depth use canonical metres;
- pairwise spacing is deterministic;
- convex hull/area are stable for normal and degenerate point sets;
- team grouping is local-team-only;
- unknown-team actors are not manufactured into a team;
- physical-end occupancy is available symmetrically;
- longitudinal mirroring preserves shape geometry and swaps physical-end occupancy;
- serialization round-trips exactly;
- P2/P3 import boundaries remain clean.
