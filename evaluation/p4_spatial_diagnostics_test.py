from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.p4_geometry_quality import (
    GeometryQualityObservation,
    GeometryQualityPolicy,
    GeometryQualityState,
)
from core.p4_spatial_contract import SpatialGeometryEvidence
from core.p4_spatial_diagnostics import (
    P4_SPATIAL_DIAGNOSTICS_CONTRACT_VERSION,
    P4SpatialDiagnosticsError,
    build_spatial_diagnostics,
    p4_spatial_diagnostics_manifest,
)
from core.p4_spatial_eligibility import (
    SpatialEligibilityObservation,
    SpatialEligibilityPolicy,
    SpatialEligibilityReason,
    SpatialEligibilityState,
)
from core.p4_spatial_snapshot import SpatialSnapshot, SpatialSnapshotPlayer
from core.p4_trajectory import TrajectoryPointState


GEOMETRY_POLICY = GeometryQualityPolicy(fresh_hold_max_age_frames=3)
ELIGIBILITY_POLICY = SpatialEligibilityPolicy(
    min_metric_player_count=1,
    min_metric_player_fraction=0.5,
    court_extent_tolerance_m=0.1,
    allow_aging_geometry=True,
    require_court_visible=True,
)


def _snapshot(
    frame_id: int,
    timestamp_ms: int,
    *,
    geometry_state: GeometryQualityState,
    eligibility_state: SpatialEligibilityState,
    raw: tuple[float, float] | None,
    smoothed: tuple[float, float] | None,
    velocity: tuple[float, float] | None,
    trajectory_state: TrajectoryPointState | None,
    rmse: float | None = None,
    uncertainty: float | None = None,
) -> SpatialSnapshot:
    geometry_valid = geometry_state is not GeometryQualityState.INVALID
    quality = GeometryQualityObservation(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        state=geometry_state,
        reason=geometry_state.value,
        homography_source=None if not geometry_valid else (
            "direct_keypoints"
            if geometry_state is GeometryQualityState.DIRECT_CALIBRATION
            else "optical_flow_hold"
        ),
        last_direct_calibration_ms=timestamp_ms if geometry_valid else None,
        homography_age_frames=0 if geometry_state is GeometryQualityState.DIRECT_CALIBRATION else (
            2 if geometry_valid else None
        ),
        anchor_count=8 if geometry_valid else None,
        inlier_count=7 if geometry_valid else None,
        reprojection_rmse_px=rmse,
        projection_uncertainty_m=uncertainty,
        policy=GEOMETRY_POLICY,
        provenance={},
    )
    eligibility = SpatialEligibilityObservation(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        state=eligibility_state,
        reasons=(
            (SpatialEligibilityReason.ELIGIBLE,)
            if eligibility_state is SpatialEligibilityState.ELIGIBLE
            else (SpatialEligibilityReason.AGING_GEOMETRY,)
            if eligibility_state is SpatialEligibilityState.DEGRADED
            else (SpatialEligibilityReason.GEOMETRY_INVALID,)
        ),
        geometry_quality_state=geometry_state,
        visible_player_count=1,
        metric_player_count=1 if raw is not None else 0,
        missing_metric_player_count=0 if raw is not None else 1,
        out_of_bounds_metric_player_count=0,
        metric_player_fraction=1.0 if raw is not None else 0.0,
        policy=ELIGIBILITY_POLICY,
        provenance={},
    )
    player = SpatialSnapshotPlayer(
        track_id=7,
        local_team_id="local_a",
        visible=True,
        image_position_px=(100.0, 200.0),
        court_position_m=raw,
        smoothed_position_m=smoothed,
        trajectory_state=trajectory_state,
        kinematics_state=None,
        velocity_mps=velocity,
        speed_mps=None if velocity is None else (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5,
        observation_confidence=0.9,
        position_confidence=0.8,
        provenance={},
    )
    return SpatialSnapshot(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        source_runtime_version="p1-runtime-v1",
        source_spatial_contract_version="p4.spatial_frame.v1",
        geometry=SpatialGeometryEvidence(
            geometry_valid=geometry_valid,
            court_visible=geometry_valid,
            homography_source=quality.homography_source,
            reprojection_rmse_px=rmse,
            homography_matrix=(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                if geometry_valid
                else None
            ),
            court_coordinate_system="p4.fiba_28x15_xy.v1",
            projection_uncertainty_m=uncertainty,
            homography_age_frames=quality.homography_age_frames,
            provenance={},
        ),
        geometry_quality=quality,
        spatial_eligibility=eligibility,
        players=(player,),
        relationships=(),
        team_shapes=(),
        motion_primitives=(),
        provenance={},
    )


def test_build_spatial_diagnostics_aggregates_geometry_and_eligibility() -> None:
    snapshots = (
        _snapshot(
            1, 100,
            geometry_state=GeometryQualityState.DIRECT_CALIBRATION,
            eligibility_state=SpatialEligibilityState.ELIGIBLE,
            raw=(1.0, 1.0), smoothed=(1.0, 1.0), velocity=None,
            trajectory_state=TrajectoryPointState.OBSERVED,
            rmse=1.0, uncertainty=0.1,
        ),
        _snapshot(
            2, 200,
            geometry_state=GeometryQualityState.OPTICAL_FLOW_HOLD_FRESH,
            eligibility_state=SpatialEligibilityState.DEGRADED,
            raw=(2.0, 1.0), smoothed=(1.8, 1.0), velocity=(8.0, 0.0),
            trajectory_state=TrajectoryPointState.OBSERVED,
            rmse=3.0, uncertainty=0.3,
        ),
        _snapshot(
            3, 300,
            geometry_state=GeometryQualityState.INVALID,
            eligibility_state=SpatialEligibilityState.INVALID,
            raw=None, smoothed=None, velocity=None,
            trajectory_state=TrajectoryPointState.INVALID_INPUT,
        ),
    )
    report = build_spatial_diagnostics(snapshots)
    assert report.total_frames == 3
    assert report.geometry_state_counts["direct_calibration"] == 1
    assert report.geometry_state_counts["optical_flow_hold_fresh"] == 1
    assert report.geometry_state_counts["invalid"] == 1
    assert report.eligibility_state_counts == {"eligible": 1, "degraded": 1, "invalid": 1}
    assert report.geometry_valid_frames == 2
    assert report.court_visible_frames == 2
    assert report.mean_reprojection_rmse_px == pytest.approx(2.0)
    assert report.mean_projection_uncertainty_m == pytest.approx(0.2)


def test_player_diagnostics_measure_coverage_gaps_and_smoothing() -> None:
    report = build_spatial_diagnostics((
        _snapshot(
            1, 100,
            geometry_state=GeometryQualityState.DIRECT_CALIBRATION,
            eligibility_state=SpatialEligibilityState.ELIGIBLE,
            raw=(1.0, 1.0), smoothed=(1.0, 1.0), velocity=None,
            trajectory_state=TrajectoryPointState.OBSERVED,
        ),
        _snapshot(
            2, 200,
            geometry_state=GeometryQualityState.DIRECT_CALIBRATION,
            eligibility_state=SpatialEligibilityState.ELIGIBLE,
            raw=(3.0, 1.0), smoothed=(2.0, 1.0), velocity=(10.0, 0.0),
            trajectory_state=TrajectoryPointState.OBSERVED,
        ),
        _snapshot(
            3, 300,
            geometry_state=GeometryQualityState.INVALID,
            eligibility_state=SpatialEligibilityState.INVALID,
            raw=None, smoothed=None, velocity=None,
            trajectory_state=TrajectoryPointState.GAP,
        ),
    ))
    player = report.player_diagnostics[0]
    assert player.frame_presence_count == 3
    assert player.raw_metric_fraction == pytest.approx(2 / 3)
    assert player.smoothed_metric_fraction == pytest.approx(2 / 3)
    assert player.velocity_fraction == pytest.approx(1 / 3)
    assert player.gap_state_count == 1
    assert player.mean_smoothing_displacement_m == pytest.approx(0.5)
    assert player.max_smoothing_displacement_m == pytest.approx(1.0)
    assert player.mean_raw_step_m == pytest.approx(2.0)
    assert player.mean_smoothed_step_m == pytest.approx(1.0)


def test_empty_report_is_valid_and_threshold_free() -> None:
    report = build_spatial_diagnostics(())
    assert report.total_frames == 0
    assert all(value == 0.0 for value in report.geometry_state_fractions.values())
    assert all(value == 0.0 for value in report.eligibility_state_fractions.values())
    assert report.player_diagnostics == ()
    assert report.mean_reprojection_rmse_px is None
    assert report.provenance["thresholds_applied"] is False


def test_duplicate_frame_timestamp_is_rejected() -> None:
    snapshot = _snapshot(
        1, 100,
        geometry_state=GeometryQualityState.DIRECT_CALIBRATION,
        eligibility_state=SpatialEligibilityState.ELIGIBLE,
        raw=(1.0, 1.0), smoothed=(1.0, 1.0), velocity=None,
        trajectory_state=TrajectoryPointState.OBSERVED,
    )
    with pytest.raises(P4SpatialDiagnosticsError):
        build_spatial_diagnostics((snapshot, snapshot))


def test_manifest_freezes_foundation_boundary() -> None:
    manifest = p4_spatial_diagnostics_manifest()
    assert manifest["contract_version"] == P4_SPATIAL_DIAGNOSTICS_CONTRACT_VERSION
    assert manifest["source_contract"] == "p4.spatial_snapshot.v1"
    assert manifest["thresholds_applied"] is False
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_semantics"] is False


def test_diagnostics_module_has_no_p2_or_p3_imports() -> None:
    path = Path(__file__).resolve().parents[1] / "core" / "p4_spatial_diagnostics.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(
        module.startswith(("services.p2", "services.p3", "core.p3"))
        for module in imported
    )
