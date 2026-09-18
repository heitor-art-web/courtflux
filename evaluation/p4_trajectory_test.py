from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.p4_court_coordinates import P4_FIBA_COURT_COORDINATE_SYSTEM
from core.p4_geometry_quality import GeometryQualityState
from core.p4_spatial_contract import SpatialFrame, SpatialGeometryEvidence, SpatialPlayerObservation
from core.p4_spatial_eligibility import (
    SpatialEligibilityObservation,
    SpatialEligibilityPolicy,
    SpatialEligibilityReason,
    SpatialEligibilityState,
)
from core.p4_trajectory import (
    P4_TRAJECTORY_CONTRACT_VERSION,
    P4TrajectoryError,
    PlayerTrajectory,
    TrajectoryPointReason,
    TrajectoryPointState,
    TrajectoryPolicy,
    build_player_trajectories,
    p4_trajectory_manifest,
)


ELIGIBILITY_POLICY = SpatialEligibilityPolicy(
    min_metric_player_count=0,
    min_metric_player_fraction=0.0,
    court_extent_tolerance_m=0.0,
    allow_aging_geometry=True,
    require_court_visible=False,
)


def _frame(frame_id: int, position: tuple[float, float] | None, *, present: bool = True) -> SpatialFrame:
    players = ()
    if present:
        players = (
            SpatialPlayerObservation(
                track_id=7,
                local_team_id=None,
                local_team_confidence=None,
                image_center_px=(100.0, 100.0),
                court_position_m=position,
                observation_confidence=0.9,
                observation_uncertainty=0.1,
                position_confidence=None,
                visible=True,
                source="p1",
            ),
        )
    return SpatialFrame(
        frame_id=frame_id,
        timestamp_ms=(frame_id - 1) * 100,
        geometry=SpatialGeometryEvidence(
            geometry_valid=True,
            court_visible=True,
            homography_source="direct_keypoints",
            homography_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            court_coordinate_system=P4_FIBA_COURT_COORDINATE_SYSTEM,
        ),
        players=players,
    )


def _eligibility(
    frame: SpatialFrame,
    state: SpatialEligibilityState = SpatialEligibilityState.ELIGIBLE,
) -> SpatialEligibilityObservation:
    return SpatialEligibilityObservation(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        state=state,
        reasons=(
            (SpatialEligibilityReason.ELIGIBLE,)
            if state is SpatialEligibilityState.ELIGIBLE
            else (SpatialEligibilityReason.AGING_GEOMETRY,)
        ),
        geometry_quality_state=GeometryQualityState.DIRECT_CALIBRATION,
        visible_player_count=len(frame.players),
        metric_player_count=sum(player.court_position_m is not None for player in frame.players),
        missing_metric_player_count=sum(player.court_position_m is None for player in frame.players),
        out_of_bounds_metric_player_count=0,
        metric_player_fraction=1.0 if frame.players and frame.players[0].court_position_m is not None else 0.0,
        policy=ELIGIBILITY_POLICY,
    )


def _policy(**overrides: object) -> TrajectoryPolicy:
    values = dict(
        smoothing_window_points=1,
        max_interpolation_gap_frames=1,
        max_step_speed_mps=20.0,
        accept_degraded_input=True,
    )
    values.update(overrides)
    return TrajectoryPolicy(**values)


def test_linear_track_and_round_trip() -> None:
    frames = tuple(_frame(index, (float(index - 1), 0.0)) for index in range(1, 4))
    trajectory = build_player_trajectories(
        frames,
        tuple(_eligibility(frame) for frame in frames),
        policy=_policy(),
    )[0]
    assert trajectory.contract_version == P4_TRAJECTORY_CONTRACT_VERSION
    assert [point.smoothed_position_m for point in trajectory.points] == [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    ]
    assert PlayerTrajectory.from_dict(trajectory.as_dict()).as_dict() == trajectory.as_dict()


def test_short_bounded_gap_is_interpolated() -> None:
    frames = (_frame(1, (0.0, 0.0)), _frame(2, None), _frame(3, (2.0, 0.0)))
    trajectory = build_player_trajectories(
        frames,
        tuple(_eligibility(frame) for frame in frames),
        policy=_policy(),
    )[0]
    middle = trajectory.points[1]
    assert middle.state is TrajectoryPointState.INTERPOLATED_GAP
    assert middle.reason is TrajectoryPointReason.INTERPOLATED_GAP
    assert middle.resolved_position_m == (1.0, 0.0)


def test_long_gap_remains_explicit() -> None:
    frames = (
        _frame(1, (0.0, 0.0)),
        _frame(2, None),
        _frame(3, None),
        _frame(4, (3.0, 0.0)),
    )
    trajectory = build_player_trajectories(
        frames,
        tuple(_eligibility(frame) for frame in frames),
        policy=_policy(max_interpolation_gap_frames=1),
    )[0]
    assert trajectory.points[1].state is TrajectoryPointState.GAP
    assert trajectory.points[2].state is TrajectoryPointState.GAP
    assert trajectory.points[1].smoothed_position_m is None


def test_invalid_spatial_frame_is_never_interpolated() -> None:
    frames = (_frame(1, (0.0, 0.0)), _frame(2, None), _frame(3, (2.0, 0.0)))
    eligibilities = (
        _eligibility(frames[0]),
        _eligibility(frames[1], SpatialEligibilityState.INVALID),
        _eligibility(frames[2]),
    )
    trajectory = build_player_trajectories(frames, eligibilities, policy=_policy())[0]
    middle = trajectory.points[1]
    assert middle.state is TrajectoryPointState.INVALID_INPUT
    assert middle.reason is TrajectoryPointReason.SPATIAL_INPUT_INVALID
    assert middle.resolved_position_m is None


def test_degraded_input_requires_explicit_opt_in() -> None:
    frames = (_frame(1, (0.0, 0.0)), _frame(2, (1.0, 0.0)), _frame(3, (2.0, 0.0)))
    eligibilities = (
        _eligibility(frames[0]),
        _eligibility(frames[1], SpatialEligibilityState.DEGRADED),
        _eligibility(frames[2]),
    )
    trajectory = build_player_trajectories(
        frames,
        eligibilities,
        policy=_policy(accept_degraded_input=False),
    )[0]
    assert trajectory.points[1].reason is TrajectoryPointReason.DEGRADED_INPUT_REJECTED


def test_teleport_is_rejected_without_mutating_raw_position() -> None:
    frames = (_frame(1, (0.0, 0.0)), _frame(2, (10.0, 0.0)), _frame(3, (0.2, 0.0)))
    trajectory = build_player_trajectories(
        frames,
        tuple(_eligibility(frame) for frame in frames),
        policy=_policy(max_step_speed_mps=5.0),
    )[0]
    point = trajectory.points[1]
    assert point.raw_position_m == (10.0, 0.0)
    assert point.state is TrajectoryPointState.INTERPOLATED_OUTLIER
    assert point.reason is TrajectoryPointReason.INTERPOLATED_OUTLIER


def test_smoothing_is_causal_and_resets_across_unresolved_gap() -> None:
    frames = (
        _frame(1, (0.0, 0.0)),
        _frame(2, (2.0, 0.0)),
        _frame(3, None),
        _frame(4, (4.0, 0.0)),
    )
    trajectory = build_player_trajectories(
        frames,
        tuple(_eligibility(frame) for frame in frames),
        policy=_policy(smoothing_window_points=2, max_interpolation_gap_frames=0),
    )[0]
    assert trajectory.points[1].smoothed_position_m == (1.0, 0.0)
    assert trajectory.points[2].smoothed_position_m is None
    assert trajectory.points[3].smoothed_position_m == (4.0, 0.0)


def test_sequence_and_policy_are_strict() -> None:
    with pytest.raises(P4TrajectoryError):
        _policy(smoothing_window_points=0)
    frames = (_frame(2, (0.0, 0.0)), _frame(1, (1.0, 0.0)))
    with pytest.raises(P4TrajectoryError, match="frame_id"):
        build_player_trajectories(
            frames,
            tuple(_eligibility(frame) for frame in frames),
            policy=_policy(),
        )


def test_manifest_and_import_boundary() -> None:
    manifest = p4_trajectory_manifest()
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_semantics"] is False
    assert manifest["invalid_spatial_input_interpolated"] is False

    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / "core/p4_trajectory.py").read_text())
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any(name.startswith("services.p2") or name.startswith("core.p3") for name in imports)
