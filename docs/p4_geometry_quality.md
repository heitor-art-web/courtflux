# P4 Geometry Quality Contract v1

**Status:** complete for P4 Spatial Foundation task 3  
**Contract:** `p4.geometry_quality.v1`  
**Policy:** `p4.geometry_quality.policy.v1`  
**Dependency boundary:** P1/P4 only; no P2 or P3 input is permitted.

## Purpose

The Spatial Foundation must distinguish how trustworthy the current image-to-court
geometry is before later code uses metric distances, trajectories or tactical relations.

This contract is deliberately narrower than spatial eligibility. It answers:

> What geometry evidence produced this frame, and how old is that evidence?

It does **not** decide whether a frame is safe for a particular downstream metric. That
decision belongs to the later Spatial Eligibility + Abstention task.

## States

The frozen v1 states are:

```text
DIRECT_CALIBRATION
OPTICAL_FLOW_HOLD_FRESH
OPTICAL_FLOW_HOLD_AGING
INVALID
```

Serialized values are:

```text
direct_calibration
optical_flow_hold_fresh
optical_flow_hold_aging
invalid
```

### `DIRECT_CALIBRATION`

Requires:

- P1 geometry marked valid;
- a metric homography matrix;
- `homography_source = direct_keypoints`.

A direct observation resets P4's calibration history:

```text
last_direct_calibration_ms = current timestamp
homography_age_frames = 0
```

### `OPTICAL_FLOW_HOLD_FRESH`

Requires:

- valid metric geometry;
- `homography_source = optical_flow_hold`;
- a previously observed direct calibration in the same ordered P4 sequence;
- age less than or equal to the explicitly configured fresh window.

### `OPTICAL_FLOW_HOLD_AGING`

Has the same lineage requirements as a fresh hold, but its age is greater than the
configured fresh window.

`AGING` does **not** automatically mean unusable. Later eligibility logic decides what a
specific spatial computation may accept.

### `INVALID`

P4 fails closed when geometry cannot be supported, including:

- P1 marks geometry invalid;
- the homography matrix is missing;
- the source is missing or unsupported;
- optical-flow hold appears without direct-calibration history.

## Explicit policy, no invented confidence

P4 does not create a synthetic geometry confidence score in this task.

The only classification parameter is:

```text
fresh_hold_max_age_frames
```

It is mandatory when constructing `GeometryQualityPolicy`; there is no hidden/default
product threshold.

The serialized policy accompanies every observation:

```text
fresh_hold_max_age_frames
policy_version
```

This makes a `FRESH` / `AGING` decision reproducible.

The threshold is a categorical freshness policy, not an accuracy estimate. Calibration
of downstream acceptance thresholds remains future validation work.

## Observation contract

`GeometryQualityObservation` contains:

```text
frame_id
timestamp_ms
state
reason

homography_source
last_direct_calibration_ms
homography_age_frames

anchor_count
inlier_count
reprojection_rmse_px
projection_uncertainty_m

policy
provenance
contract_version
```

The diagnostics match the P4 roadmap without forcing unavailable upstream values to be
fabricated. Unknown diagnostics remain explicit `null`.

## P1 diagnostics bridge

The P4 spatial adapter now copies a small allow-listed subset of existing P1 camera
diagnostics into:

```text
SpatialFrame.geometry.provenance.p1_camera_diagnostics
```

Allowed values include:

```text
valid
source
anchor_count
inlier_count
reprojection_rmse_px
model_id
keypoint_schema
keypoint_backend
error
```

This does not change `CameraTracker`, `CourtState`, detector behaviour or the P1
pipeline. It only preserves P1 evidence inside the existing P4 provenance field so
persisted `SpatialFrame` rows remain explainable.

## Age reconstruction

`GeometryQualityTracker` operates over ordered `SpatialFrame` values.

```text
DIRECT frame 100
       ↓
FLOW frame 101 → age 1
       ↓
FLOW frame 102 → age 2
       ↓
FLOW frame 110 → age 10
```

P4 therefore derives age from its own sequence history instead of adding stateful
behaviour to P1.

Frames must arrive with strictly increasing `frame_id` and non-decreasing source-video
timestamps. Out-of-order sequences are rejected rather than repaired.

## No RMSE threshold duplication

P1 already decides whether its direct homography estimate is valid. P4 preserves
`reprojection_rmse_px` but does not invent a second RMSE acceptance threshold here.

This avoids two independent definitions of "valid homography".

Later validation may use RMSE/projection uncertainty when defining task-specific spatial
eligibility, but it must do so explicitly and version those decisions.

## Dependency boundary

The geometry-quality modules may consume only:

```text
SpatialFrame
SpatialGeometryEvidence
P1 camera diagnostics stored as provenance
ordered P4 sequence history
```

They must not consume:

```text
P2 events
possessions
ball handler
game-player identity
P3 derivations
P3 review state
clips
tactical labels
```

The contract manifest explicitly reports:

```text
p2_dependency = false
p3_dependency = false
numeric_confidence_model = false
spatial_eligibility_decision = false
```

## Task 3 exit gate

Task 3 is complete when:

- the four quality states are frozen and versioned;
- policy configuration is explicit and serialized;
- direct calibration resets age correctly;
- optical-flow holds require direct history;
- fresh/aging state follows explicit policy;
- invalid/unknown geometry fails closed;
- anchor/inlier/RMSE diagnostics are preserved when available;
- ordered-sequence invariants are tested;
- no P2/P3 imports or semantics exist;
- P1 runtime behaviour is unchanged.
