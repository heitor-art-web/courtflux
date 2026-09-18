# P4 Canonical FIBA Court Zones v1

**Status:** complete for P4 Spatial Foundation task 5  
**Contract:** `p4.fiba_zones.v1`  
**Dependency boundary:** canonical FIBA geometry only; no P2/P3 context is required.

## Purpose

Task 5 defines deterministic, versioned court regions on top of `p4.fiba_28x15_xy.v1`.
The zones are geometry primitives for later trajectories, spacing, diagnostics and radar
rendering. They are not possession or tactical labels.

Every query that depends on one basket requires the caller to choose a physical `CourtEnd`:

```text
LEFT
RIGHT
```

That argument means only "which physical basket/end of the court?". It does not mean the
attacking basket, offense, defense, frontcourt or backcourt.

## Frozen zones

`CourtZone` v1 includes:

```text
OUTSIDE
OTHER_HALF
INTERIOR
PAINT
RIM_AREA
PERIMETER
LOWER_CORNER
UPPER_CORNER
LOWER_WING
UPPER_WING
TOP
```

`LOWER` / `UPPER` are canonical y-axis directions. They deliberately avoid attack-relative
"left" / "right" semantics until P2 supplies validated team/possession context later.

## Paint

The paint is defined from official FIBA marking geometry relative to the selected physical
end:

```text
depth: 0.00 m → 5.80 m
y:     5.05 m → 9.95 m
```

The boundary is included for geometric membership.

## Rim area

CourtFlux v1 uses the official 1.30 m no-charge semi-circle radius as the radius of a simple
basket-centred spatial grouping:

```text
distance(point, basket centre) <= 1.30 m
```

This is intentionally documented as a **CourtFlux spatial grouping**, not FIBA rules
adjudication. It provides a stable near-rim region without introducing an arbitrary hidden
threshold.

## Three-point / perimeter region

The FIBA three-point boundary uses:

```text
arc radius                  = 6.75 m
straight line from sideline = 0.90 m
```

The corner connection depth is derived from those official dimensions and the basket
centre, rather than hard-coded independently.

The three-point line itself belongs to the two-point area, so points exactly on the line are
not classified as `PERIMETER` by this contract.

For one selected basket, the perimeter is limited to that physical half-court. Points in
the opposite physical half are `OTHER_HALF`, avoiding accidental attack-direction claims.

## Perimeter subzones

The perimeter is subdivided using canonical geometry:

```text
LOWER_CORNER
UPPER_CORNER
LOWER_WING
UPPER_WING
TOP
```

Corners require both the sideline strip and the official straight-line connection depth.
Beyond the corner connection, perimeter points are grouped into lower wing, top or upper
wing from their canonical y coordinate.

These names are intentionally physical/canonical rather than offense-relative.

## Overlapping memberships

`zone_memberships()` can return overlapping geometric facts. Example near the rim:

```text
{RIM_AREA, PAINT}
```

A corner three returns:

```text
{PERIMETER, LOWER_CORNER}
```

`primary_zone()` provides a deterministic convenience classification with specificity
precedence:

```text
OUTSIDE
OTHER_HALF
RIM_AREA
PAINT
CORNER / WING / TOP
INTERIOR
```

The underlying memberships remain available so downstream code does not lose information.

## Outside / opposite-half handling

Invalid court extents are not clamped or repaired:

```text
outside [0,28] × [0,15] → OUTSIDE
```

For a selected physical basket:

```text
end-relative depth > 14 m → OTHER_HALF
```

Again, `OTHER_HALF` is a physical geometry label only. It is not `BACKCOURT` or
`FRONTCOURT` because those require basketball context that this foundation intentionally
does not consume.

## Dependency boundary

Allowed inputs:

```text
CourtPoint
CourtEnd
p4.fiba_28x15_xy.v1 geometry
```

Forbidden inputs:

```text
possession
attack direction
ball handler
offense / defense
P2 events
P3 derivations/review/clips
tactical labels
```

The manifest records:

```text
p2_dependency = false
p3_dependency = false
attack_direction_inference = false
tactical_semantics = false
```

## Task 5 exit gate

Task 5 is complete when:

- paint geometry is versioned;
- rim area is explicit/versioned;
- FIBA three-point geometry is reproduced from official dimensions;
- boundary-line semantics are explicit;
- corner/wing/top subdivisions are symmetric for both physical ends;
- outside and opposite-half states are explicit;
- overlapping memberships are preserved;
- no attack direction is inferred;
- no P2/P3 imports or semantics exist.
