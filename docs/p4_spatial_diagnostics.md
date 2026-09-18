# P4 Geometry + Spatial Diagnostics

**Status:** complete for P4 Spatial Foundation task 14  
**Contract:** `p4.spatial_diagnostics.v1`  
**Input:** `p4.spatial_snapshot.v1`  
**Dependency boundary:** P1/P4 spatial state only; no P2/P3 basketball semantics.

## Purpose

The diagnostics layer summarizes the observable quality and coverage of the P4 Spatial Foundation without turning those measurements into product-quality thresholds.

It answers questions such as:

- how much of the timeline is direct calibration versus optical-flow hold or invalid geometry;
- how much of the timeline is spatially eligible, degraded or invalid;
- how often reprojection RMSE and projection uncertainty are available, and their observed mean/max values;
- how much metric/smoothed/velocity coverage exists per `track_id`;
- how many materialized track-presence frames are explicit trajectory gaps or rejected/invalid states;
- how large the raw-to-smoothed displacement is;
- how raw and smoothed step distances compare.

It deliberately does **not** decide whether those numbers are acceptable. Threshold calibration, Golden Fixtures, real-video GT and the formal Spatial Exit Gate remain in the later P4 Spatial Validation tranche.

## Contract

```text
p4.spatial_snapshot.v1[]
        ↓
build_spatial_diagnostics(...)
        ↓
p4.spatial_diagnostics.v1
```

`SpatialDiagnosticsReport` contains:

### Frame-level geometry coverage

- total frames;
- counts and fractions for:
  - `DIRECT_CALIBRATION`;
  - `OPTICAL_FLOW_HOLD_FRESH`;
  - `OPTICAL_FLOW_HOLD_AGING`;
  - `INVALID`;
- `court_visible_frames`;
- `geometry_valid_frames`.

### Eligibility coverage

Counts and fractions for:

- `ELIGIBLE`;
- `DEGRADED`;
- `INVALID`.

The report reuses the already-frozen task-6 states. It does not reinterpret or repair eligibility.

### Geometry error evidence

When present in `GeometryQualityObservation`:

- reprojection RMSE sample count;
- mean reprojection RMSE in pixels;
- max reprojection RMSE in pixels;
- projection uncertainty sample count;
- mean projection uncertainty in metres;
- max projection uncertainty in metres.

Missing diagnostics remain missing; absence is not converted to zero error.

### Per-track spatial diagnostics

For each `track_id` observed in materialized snapshots:

- frame-presence count;
- raw metric position count/fraction;
- smoothed metric position count/fraction;
- velocity count/fraction;
- explicit trajectory gap/rejected/invalid state count;
- mean/max raw-to-smoothed displacement;
- mean raw step distance;
- mean smoothed step distance.

Coverage denominators are the frames in which the track is present in `SpatialSnapshot`, not all video frames. This avoids treating periods outside the materialized track lifetime as missing evidence.

Step-distance summaries are descriptive diagnostics only. They do not classify a trajectory as jittery or stable.

## No hidden thresholds

`p4.spatial_diagnostics.v1` intentionally has no quality policy and no pass/fail state.

```text
diagnostics → measurement
validation tranche → calibrated criteria
exit gate → pass/fail
```

This keeps the structural foundation independent from empirical accuracy decisions that require real footage and reviewed ground truth.

## Radar relationship

Task 13 already provides frame-by-frame inspection in the internal Spatial Radar:

- raw vs smoothed positions;
- geometry state;
- homography source/age;
- anchors/inliers;
- RMSE/uncertainty;
- eligibility/reasons;
- metric-player coverage.

Task 14 complements that visual inspection with deterministic aggregate diagnostics that can later be persisted, exported or shown in the same internal tooling without changing the spatial contracts.

## Boundary

The diagnostics module does not consume or infer:

- possession;
- ball handler;
- offense/defense roles;
- game-player identity;
- defender/matchup;
- screen/cut/drive/P&R;
- tactical labels;
- P3 review state or clips.

Manifest fields make this explicit:

```text
thresholds_applied = false
p2_dependency = false
p3_dependency = false
tactical_semantics = false
```

## Foundation completion

With this task, the structural P4 Spatial Foundation is complete:

1. Spatial Contract v1
2. Spatial Persistence Model
3. Geometry Quality Contract
4. Canonical FIBA Court Coordinate System
5. Canonical Court Zones
6. Spatial Eligibility + Abstention
7. Trajectory Engine
8. Kinematics Engine
9. Spatial Relationship Metrics
10. Team Shape Metrics
11. Generic Motion Primitives
12. Spatial Snapshot Materialization
13. Radar / Court Viewer Foundation
14. Geometry + Spatial Diagnostics

The next spatial-quality tranche is intentionally separate:

```text
P4 Spatial Validation
    ├── Spatial GT / Golden Fixtures
    ├── real-video measurements
    ├── threshold calibration
    └── Spatial Validation + Exit Gate
```

Tactical P4 remains gated until validated P2/P3 basketball context is combined with this completed spatial base.
