from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.p4_kinematics import (
    KinematicsPointState,
    KinematicsPolicy,
    P4_KINEMATICS_CONTRACT_VERSION,
    P4KinematicsError,
    PlayerKinematics,
    build_player_kinematics,
    p4_kinematics_manifest,
)
from core.p4_spatial_eligibility import SpatialEligibilityState
from core.p4_trajectory import (
    P4_TRAJECTORY_CONTRACT_VERSION,
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


def _trajectory(
    positions: tuple[tuple[float, float] | None, ...],
    *,
    states: tuple[TrajectoryPointState, ...] | None = None,
) -> PlayerTrajectory:
    resolved_states = states or tuple(
        TrajectoryPointState.OBSERVED if position is not None else TrajectoryPointState.GAP
        for position in positions
    )
    points = []
    for index, (position, state) in enumerate(zip(positions, resolved_states), start=1):
        reason = (
            TrajectoryPointReason.OBSERVED
            if state is TrajectoryPointState.OBSERVED
            else (
                TrajectoryPointReason.INTERPOLATED_GAP
                if state is TrajectoryPointState.INTERPOLATED_GAP
                else TrajectoryPointReason.METRIC_POSITION_MISSING
            )
        )
        points.append(
            TrajectoryPoint(
                track_id=7,
                frame_id=index,
                timestamp_ms=(index - 1) * 100,
                state=state,
                reason=reason,
                raw_position_m=position if state is TrajectoryPointState.OBSERVED else None,
                resolved_position_m=position,
                smoothed_position_m=position,
                observation_confidence=0.9 if position is not None else None,
                spatial_eligibility_state=SpatialEligibilityState.ELIGIBLE,
                provenance={},
            )
        )
    return PlayerTrajectory(
        track_id=7,
        points=tuple(points),
        policy=TRAJECTORY_POLICY,
        source_spatial_contract_version="p4.spatial_frame.v1",
        source_eligibility_contract_version="p4.spatial_eligibility.v1",
    )


def test_constant_velocity_and_round_trip() -> None:
    trajectory = _trajectory(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)))
    result = build_player_kinematics(
        trajectory,
        policy=KinematicsPolicy(max_derivative_gap_ms=150, use_interpolated_points=True),
    )
    assert result.contract_version == P4_KINEMATICS_CONTRACT_VERSION
    assert result.source_trajectory_contract_version == P4_TRAJECTORY_CONTRACT_VERSION
    assert result.points[1].velocity_mps == (10.0, 0.0)
    assert result.points[2].acceleration_mps2 == (0.0, 0.0)
    assert result.points[2].state is KinematicsPointState.FULL
    assert PlayerKinematics.from_dict(result.as_dict()).as_dict() == result.as_dict()


def test_heading_uses_canonical_xy_axes() -> None:
    trajectory = _trajectory(((0.0, 0.0), (0.0, 1.0), (0.0, 2.0)))
    result = build_player_kinematics(
        trajectory,
        policy=KinematicsPolicy(max_derivative_gap_ms=150, use_interpolated_points=True),
    )
    assert result.points[1].velocity_mps == (0.0, 10.0)
    assert result.points[1].heading_rad == pytest.approx(1.5707963267948966)


def test_stationary_speed_is_zero_without_inventing_heading() -> None:
    trajectory = _trajectory(((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)))
    result = build_player_kinematics(
        trajectory,
        policy=KinematicsPolicy(max_derivative_gap_ms=150, use_interpolated_points=True),
    )
    assert result.points[1].speed_mps == 0.0
    assert result.points[1].heading_rad is None


def test_unresolved_gap_resets_derivatives() -> None:
    trajectory = _trajectory(((0.0, 0.0), None, (2.0, 0.0), (3.0, 0.0)))
    result = build_player_kinematics(
        trajectory,
        policy=KinematicsPolicy(max_derivative_gap_ms=150, use_interpolated_points=True),
    )
    assert result.points[1].state is KinematicsPointState.UNAVAILABLE
    assert result.points[2].state is KinematicsPointState.UNAVAILABLE
    assert result.points[3].state is KinematicsPointState.VELOCITY_ONLY


def test_large_time_gap_does_not_create_velocity() -> None:
    trajectory = _trajectory(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    policy = KinematicsPolicy(max_derivative_gap_ms=50, use_interpolated_points=True)
    result = build_player_kinematics(trajectory, policy=policy)
    assert all(point.state is KinematicsPointState.UNAVAILABLE for point in result.points)


def test_interpolated_points_are_policy_controlled() -> None:
    states = (
        TrajectoryPointState.OBSERVED,
        TrajectoryPointState.INTERPOLATED_GAP,
        TrajectoryPointState.OBSERVED,
    )
    trajectory = _trajectory(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), states=states)

    reject = build_player_kinematics(
        trajectory,
        policy=KinematicsPolicy(max_derivative_gap_ms=150, use_interpolated_points=False),
    )
    assert all(point.state is KinematicsPointState.UNAVAILABLE for point in reject.points)

    accept = build_player_kinematics(
        trajectory,
        policy=KinematicsPolicy(max_derivative_gap_ms=150, use_interpolated_points=True),
    )
    assert accept.points[1].state is KinematicsPointState.VELOCITY_ONLY
    assert accept.points[2].state is KinematicsPointState.FULL


def test_policy_is_strict() -> None:
    with pytest.raises(P4KinematicsError):
        KinematicsPolicy(max_derivative_gap_ms=0, use_interpolated_points=True)
    with pytest.raises(P4KinematicsError):
        KinematicsPolicy(max_derivative_gap_ms=100, use_interpolated_points=1)  # type: ignore[arg-type]


def test_manifest_and_import_boundary() -> None:
    manifest = p4_kinematics_manifest()
    assert manifest["stationary_classification"] is False
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_semantics"] is False

    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / "core/p4_kinematics.py").read_text())
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any(name.startswith("services.p2") or name.startswith("core.p3") for name in imports)
