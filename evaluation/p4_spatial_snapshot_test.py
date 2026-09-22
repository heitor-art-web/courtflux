from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.p4_geometry_quality import (
    GeometryQualityObservation,
    GeometryQualityPolicy,
    GeometryQualityState,
)
from core.p4_kinematics import (
    KinematicsPoint,
    KinematicsPointState,
    KinematicsPolicy,
    PlayerKinematics,
)
from core.p4_motion_primitives import (
    MotionPrimitiveObservation,
    MotionPrimitiveType,
)
from core.p4_spatial_contract import (
    SpatialFrame,
    SpatialGeometryEvidence,
    SpatialPlayerObservation,
)
from core.p4_spatial_eligibility import (
    SpatialEligibilityObservation,
    SpatialEligibilityPolicy,
    SpatialEligibilityReason,
    SpatialEligibilityState,
)
from core.p4_spatial_snapshot import (
    P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION,
    P4SpatialSnapshotError,
    SpatialSnapshot,
    materialize_spatial_snapshot,
    materialize_spatial_snapshots,
    p4_spatial_snapshot_manifest,
)
from core.p4_trajectory import (
    PlayerTrajectory,
    TrajectoryPoint,
    TrajectoryPointReason,
    TrajectoryPointState,
    TrajectoryPolicy,
)


TRAJECTORY_POLICY = TrajectoryPolicy(
    smoothing_window_points=1,
    max_interpolation_gap_frames=1,
    max_step_speed_mps=20.0,
    accept_degraded_input=True,
)
KINEMATICS_POLICY = KinematicsPolicy(
    max_derivative_gap_ms=200,
    use_interpolated_points=True,
)
GEOMETRY_POLICY = GeometryQualityPolicy(fresh_hold_max_age_frames=3)
ELIGIBILITY_POLICY = SpatialEligibilityPolicy(
    min_metric_player_count=1,
    min_metric_player_fraction=0.5,
    court_extent_tolerance_m=0.1,
    allow_aging_geometry=True,
    require_court_visible=True,
)


def _frame(
    *,
    frame_id: int = 10,
    timestamp_ms: int = 1000,
    geometry_valid: bool = True,
    players: tuple[SpatialPlayerObservation, ...] | None = None,
) -> SpatialFrame:
    if players is None:
        players = (
            SpatialPlayerObservation(
                track_id=1,
                local_team_id="local_a",
                local_team_confidence=0.9,
                image_center_px=(100.0, 200.0),
                court_position_m=(1.0, 1.0),
                observation_confidence=0.9,
                observation_uncertainty=0.1,
                position_confidence=0.8,
                visible=True,
                source="tracked_player_detection",
                provenance={},
            ),
            SpatialPlayerObservation(
                track_id=2,
                local_team_id="local_a",
                local_team_confidence=0.8,
                image_center_px=(200.0, 200.0),
                court_position_m=(4.0, 1.0),
                observation_confidence=0.8,
                observation_uncertainty=0.2,
                position_confidence=0.7,
                visible=True,
                source="tracked_player_detection",
                provenance={},
            ),
        )
    return SpatialFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        geometry=SpatialGeometryEvidence(
            geometry_valid=geometry_valid,
            court_visible=geometry_valid,
            homography_source="direct_keypoints" if geometry_valid else None,
            reprojection_rmse_px=1.5 if geometry_valid else None,
            homography_matrix=(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                if geometry_valid
                else None
            ),
            court_coordinate_system="p4.fiba_28x15_xy.v1",
            provenance={},
        ),
        players=players,
        source_runtime_version="p1-runtime-v1",
        provenance={},
    )


def _quality(frame: SpatialFrame, *, valid: bool = True) -> GeometryQualityObservation:
    return GeometryQualityObservation(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        state=(
            GeometryQualityState.DIRECT_CALIBRATION
            if valid
            else GeometryQualityState.INVALID
        ),
        reason="direct calibration" if valid else "geometry invalid",
        homography_source="direct_keypoints" if valid else None,
        last_direct_calibration_ms=frame.timestamp_ms if valid else None,
        homography_age_frames=0 if valid else None,
        anchor_count=8 if valid else None,
        inlier_count=8 if valid else None,
        reprojection_rmse_px=1.5 if valid else None,
        projection_uncertainty_m=None,
        policy=GEOMETRY_POLICY,
        provenance={},
    )


def _eligibility(
    frame: SpatialFrame,
    quality: GeometryQualityObservation,
    *,
    valid: bool = True,
) -> SpatialEligibilityObservation:
    return SpatialEligibilityObservation(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        state=(SpatialEligibilityState.ELIGIBLE if valid else SpatialEligibilityState.INVALID),
        reasons=(
            (SpatialEligibilityReason.ELIGIBLE,)
            if valid
            else (SpatialEligibilityReason.GEOMETRY_INVALID,)
        ),
        geometry_quality_state=quality.state,
        visible_player_count=len(frame.players),
        metric_player_count=len(frame.players) if valid else 0,
        missing_metric_player_count=0,
        out_of_bounds_metric_player_count=0,
        metric_player_fraction=1.0 if valid and frame.players else 0.0,
        policy=ELIGIBILITY_POLICY,
        provenance={},
    )


def _trajectory(
    track_id: int,
    frame: SpatialFrame,
    *,
    raw: tuple[float, float] | None,
    resolved: tuple[float, float] | None,
    smoothed: tuple[float, float] | None,
    state: TrajectoryPointState = TrajectoryPointState.OBSERVED,
) -> PlayerTrajectory:
    reason = {
        TrajectoryPointState.OBSERVED: TrajectoryPointReason.OBSERVED,
        TrajectoryPointState.INTERPOLATED_GAP: TrajectoryPointReason.INTERPOLATED_GAP,
        TrajectoryPointState.INTERPOLATED_OUTLIER: TrajectoryPointReason.INTERPOLATED_OUTLIER,
        TrajectoryPointState.INVALID_INPUT: TrajectoryPointReason.SPATIAL_INPUT_INVALID,
    }.get(state, TrajectoryPointReason.METRIC_POSITION_MISSING)
    return PlayerTrajectory(
        track_id=track_id,
        points=(
            TrajectoryPoint(
                track_id=track_id,
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                state=state,
                reason=reason,
                raw_position_m=raw,
                resolved_position_m=resolved,
                smoothed_position_m=smoothed,
                observation_confidence=0.9 if raw is not None else None,
                spatial_eligibility_state=(
                    SpatialEligibilityState.ELIGIBLE
                    if state is not TrajectoryPointState.INVALID_INPUT
                    else SpatialEligibilityState.INVALID
                ),
                provenance={},
            ),
        ),
        policy=TRAJECTORY_POLICY,
        source_spatial_contract_version="p4.spatial_frame.v1",
        source_eligibility_contract_version="p4.spatial_eligibility.v1",
        provenance={},
    )


def _kinematics(
    track_id: int,
    frame: SpatialFrame,
    *,
    position: tuple[float, float],
    velocity: tuple[float, float],
) -> PlayerKinematics:
    return PlayerKinematics(
        track_id=track_id,
        points=(
            KinematicsPoint(
                track_id=track_id,
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                state=KinematicsPointState.VELOCITY_ONLY,
                trajectory_point_state=TrajectoryPointState.OBSERVED,
                position_m=position,
                velocity_mps=velocity,
                speed_mps=(velocity[0] ** 2 + velocity[1] ** 2) ** 0.5,
                acceleration_mps2=None,
                acceleration_magnitude_mps2=None,
                heading_rad=0.0 if velocity[1] == 0.0 and velocity[0] > 0.0 else None,
                provenance={},
            ),
        ),
        policy=KINEMATICS_POLICY,
        source_trajectory_contract_version="p4.trajectory.v1",
        provenance={},
    )


def test_materializes_enriched_snapshot_from_smoothed_positions() -> None:
    frame = _frame()
    quality = _quality(frame)
    eligibility = _eligibility(frame, quality)
    trajectories = (
        _trajectory(1, frame, raw=(1.0, 1.0), resolved=(1.5, 1.0), smoothed=(1.5, 1.0)),
        _trajectory(2, frame, raw=(4.0, 1.0), resolved=(3.5, 1.0), smoothed=(3.5, 1.0)),
    )
    kinematics = (
        _kinematics(1, frame, position=(1.5, 1.0), velocity=(1.0, 0.0)),
        _kinematics(2, frame, position=(3.5, 1.0), velocity=(-1.0, 0.0)),
    )

    snapshot = materialize_spatial_snapshot(
        frame, quality, eligibility, trajectories=trajectories, kinematics=kinematics
    )

    assert snapshot.contract_version == P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION
    assert [player.track_id for player in snapshot.players] == [1, 2]
    assert snapshot.players[0].court_position_m == (1.0, 1.0)
    assert snapshot.players[0].smoothed_position_m == (1.5, 1.0)
    assert snapshot.players[0].velocity_mps == (1.0, 0.0)
    assert len(snapshot.relationships) == 1
    assert snapshot.relationships[0].distance_m == pytest.approx(2.0)
    assert len(snapshot.team_shapes) == 1
    assert snapshot.team_shapes[0].centroid_m == pytest.approx((2.5, 1.0))
    assert SpatialSnapshot.from_dict(snapshot.as_dict()).as_dict() == snapshot.as_dict()


def test_invalid_snapshot_abstains_but_preserves_raw_source_evidence() -> None:
    frame = _frame(geometry_valid=False)
    quality = _quality(frame, valid=False)
    eligibility = _eligibility(frame, quality, valid=False)

    snapshot = materialize_spatial_snapshot(frame, quality, eligibility)

    assert snapshot.spatial_eligibility.state is SpatialEligibilityState.INVALID
    assert snapshot.players[0].court_position_m == (1.0, 1.0)
    assert snapshot.players[0].smoothed_position_m is None
    assert snapshot.relationships == ()
    assert snapshot.team_shapes == ()
    assert snapshot.motion_primitives == ()


def test_invalid_snapshot_rejects_derived_metric_truth() -> None:
    frame = _frame(geometry_valid=False)
    quality = _quality(frame, valid=False)
    eligibility = _eligibility(frame, quality, valid=False)
    bad_trajectory = _trajectory(
        1,
        frame,
        raw=None,
        resolved=(1.0, 1.0),
        smoothed=(1.0, 1.0),
        state=TrajectoryPointState.INVALID_INPUT,
    )

    with pytest.raises(P4SpatialSnapshotError, match="INVALID"):
        materialize_spatial_snapshot(
            frame, quality, eligibility, trajectories=(bad_trajectory,)
        )


def test_interpolated_track_without_frame_observation_is_preserved() -> None:
    frame = _frame(players=(_frame().players[0],))
    quality = _quality(frame)
    eligibility = _eligibility(frame, quality)
    extra = _trajectory(
        9,
        frame,
        raw=None,
        resolved=(5.0, 5.0),
        smoothed=(5.0, 5.0),
        state=TrajectoryPointState.INTERPOLATED_GAP,
    )

    snapshot = materialize_spatial_snapshot(
        frame, quality, eligibility, trajectories=(extra,)
    )
    player = next(item for item in snapshot.players if item.track_id == 9)
    assert player.visible is False
    assert player.local_team_id is None
    assert player.court_position_m is None
    assert player.smoothed_position_m == (5.0, 5.0)


def test_motion_primitives_are_scoped_to_snapshot_timestamp() -> None:
    frame = _frame(players=(_frame().players[0],))
    quality = _quality(frame)
    eligibility = _eligibility(frame, quality)
    trajectory = _trajectory(
        1, frame, raw=(1.0, 1.0), resolved=(1.0, 1.0), smoothed=(1.0, 1.0)
    )
    active = MotionPrimitiveObservation(
        primitive_type=MotionPrimitiveType.STATIONARY_WINDOW,
        start_ms=900,
        end_ms=1100,
        track_ids=(1,),
        target_m=None,
        support={},
        provenance={},
    )
    inactive = MotionPrimitiveObservation(
        primitive_type=MotionPrimitiveType.STATIONARY_WINDOW,
        start_ms=100,
        end_ms=200,
        track_ids=(1,),
        target_m=None,
        support={},
        provenance={},
    )

    snapshot = materialize_spatial_snapshot(
        frame,
        quality,
        eligibility,
        trajectories=(trajectory,),
        motion_primitives=(active, inactive),
    )
    assert snapshot.motion_primitives == (active,)


def test_alignment_and_sequence_lengths_are_fail_closed() -> None:
    frame = _frame()
    quality = _quality(frame)
    eligibility = _eligibility(frame, quality)
    bad_quality = _quality(_frame(frame_id=11, timestamp_ms=1100))

    with pytest.raises(P4SpatialSnapshotError, match="match"):
        materialize_spatial_snapshot(frame, bad_quality, eligibility)

    with pytest.raises(P4SpatialSnapshotError, match="identical length"):
        materialize_spatial_snapshots((frame,), (), (eligibility,))


def test_manifest_and_import_boundary() -> None:
    manifest = p4_spatial_snapshot_manifest()
    assert manifest["contract_version"] == P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION
    assert manifest["relationship_position_source"] == "smoothed_position_m"
    assert manifest["invalid_emits_metric_truth"] is False
    assert manifest["attack_direction_inference"] is False
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_semantics"] is False

    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / "core/p4_spatial_snapshot.py").read_text())
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any(
        name.startswith("services.p2") or name.startswith("core.p3")
        for name in imports
    )
