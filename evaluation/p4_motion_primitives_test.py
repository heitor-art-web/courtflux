from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.p4_kinematics import KinematicsPoint, KinematicsPointState, KinematicsPolicy, PlayerKinematics
from core.p4_motion_primitives import (
    MotionPrimitiveObservation,
    MotionPrimitivePolicy,
    MotionPrimitiveType,
    P4_MOTION_PRIMITIVE_CONTRACT_VERSION,
    P4MotionPrimitiveError,
    classify_kinematic_motion,
    classify_pair_radial_motion,
    classify_path_crossing,
    classify_stationary_window,
    classify_target_motion,
    p4_motion_primitive_manifest,
)
from core.p4_spatial_relationships import (
    LocalTeamRelation,
    PairwiseSpatialRelationship,
    SpatialActorState,
)

POLICY = MotionPrimitivePolicy(
    pair_radial_speed_min_mps=0.5,
    stationary_speed_max_mps=0.2,
    stationary_window_min_ms=200,
    tangential_acceleration_min_mps2=0.5,
    movement_speed_min_mps=0.5,
    lateral_dominance_ratio=1.5,
    target_radial_speed_min_mps=0.5,
    path_crossing_epsilon_m=1e-9,
)


def _point(
    *,
    track_id: int = 7,
    timestamp_ms: int,
    position: tuple[float, float],
    velocity: tuple[float, float] | None,
    speed: float | None,
    acceleration: tuple[float, float] | None = None,
    state: KinematicsPointState = KinematicsPointState.FULL,
) -> KinematicsPoint:
    return KinematicsPoint(
        track_id=track_id,
        frame_id=timestamp_ms // 100 + 1,
        timestamp_ms=timestamp_ms,
        state=state,
        trajectory_point_state="observed",
        position_m=position,
        velocity_mps=velocity,
        speed_mps=speed,
        acceleration_mps2=acceleration,
        acceleration_magnitude_mps2=(
            None if acceleration is None else (acceleration[0] ** 2 + acceleration[1] ** 2) ** 0.5
        ),
        heading_rad=None,
        provenance={},
    )


def test_pair_radial_motion_distinguishes_approaching_and_separating() -> None:
    approaching = PairwiseSpatialRelationship(
        timestamp_ms=100,
        track_a_id=1,
        track_b_id=2,
        distance_m=2.0,
        delta_position_m=(2.0, 0.0),
        local_team_relation=LocalTeamRelation.DIFFERENT,
        relative_velocity_mps=(-1.0, 0.0),
        closing_speed_mps=1.0,
        provenance={},
    )
    separating = PairwiseSpatialRelationship(
        timestamp_ms=200,
        track_a_id=1,
        track_b_id=2,
        distance_m=2.5,
        delta_position_m=(2.5, 0.0),
        local_team_relation=LocalTeamRelation.DIFFERENT,
        relative_velocity_mps=(1.0, 0.0),
        closing_speed_mps=-1.0,
        provenance={},
    )
    assert classify_pair_radial_motion(approaching, policy=POLICY)[0].primitive_type is MotionPrimitiveType.APPROACHING
    assert classify_pair_radial_motion(separating, policy=POLICY)[0].primitive_type is MotionPrimitiveType.SEPARATING


def test_pair_radial_motion_abstains_below_explicit_threshold() -> None:
    relationship = PairwiseSpatialRelationship(
        timestamp_ms=100,
        track_a_id=1,
        track_b_id=2,
        distance_m=2.0,
        delta_position_m=(2.0, 0.0),
        local_team_relation=LocalTeamRelation.UNKNOWN,
        relative_velocity_mps=(-0.2, 0.0),
        closing_speed_mps=0.2,
        provenance={},
    )
    assert classify_pair_radial_motion(relationship, policy=POLICY) == ()


def test_stationary_window_requires_full_explicit_duration() -> None:
    points = (
        _point(timestamp_ms=0, position=(1.0, 1.0), velocity=(0.0, 0.0), speed=0.0),
        _point(timestamp_ms=100, position=(1.0, 1.0), velocity=(0.0, 0.0), speed=0.0),
        _point(timestamp_ms=200, position=(1.0, 1.0), velocity=(0.0, 0.0), speed=0.0),
    )
    result = classify_stationary_window(
        PlayerKinematics(
            track_id=7,
            points=points,
            policy=KinematicsPolicy(max_derivative_gap_ms=150, use_interpolated_points=True),
            source_trajectory_contract_version="p4.trajectory.v1",
            provenance={},
        ),
        policy=POLICY,
    )
    assert result[0].primitive_type is MotionPrimitiveType.STATIONARY_WINDOW
    assert result[0].start_ms == 0
    assert result[0].end_ms == 200


def test_accelerating_decelerating_and_lateral_are_observable_only() -> None:
    accelerating = _point(timestamp_ms=100, position=(1.0, 1.0), velocity=(2.0, 0.0), speed=2.0, acceleration=(1.0, 0.0))
    decelerating = _point(timestamp_ms=200, position=(2.0, 1.0), velocity=(2.0, 0.0), speed=2.0, acceleration=(-1.0, 0.0))
    lateral = _point(timestamp_ms=300, position=(2.0, 2.0), velocity=(0.5, 2.0), speed=(4.25 ** 0.5), acceleration=(0.0, 0.0))
    assert MotionPrimitiveType.ACCELERATING in {item.primitive_type for item in classify_kinematic_motion(accelerating, policy=POLICY)}
    assert MotionPrimitiveType.DECELERATING in {item.primitive_type for item in classify_kinematic_motion(decelerating, policy=POLICY)}
    assert MotionPrimitiveType.LATERAL_MOVEMENT in {item.primitive_type for item in classify_kinematic_motion(lateral, policy=POLICY)}


def test_target_motion_uses_caller_supplied_point_not_attack_direction() -> None:
    previous = _point(timestamp_ms=0, position=(1.0, 1.0), velocity=(1.0, 0.0), speed=1.0)
    current = _point(timestamp_ms=1000, position=(2.0, 1.0), velocity=(1.0, 0.0), speed=1.0)
    toward = classify_target_motion(previous, current, target_m=(5.0, 1.0), policy=POLICY)
    away = classify_target_motion(previous, current, target_m=(0.0, 1.0), policy=POLICY)
    assert toward[0].primitive_type is MotionPrimitiveType.MOVING_TOWARD_TARGET
    assert away[0].primitive_type is MotionPrimitiveType.MOVING_AWAY_FROM_TARGET


def test_path_crossing_detects_segment_intersection() -> None:
    result = classify_path_crossing(
        SpatialActorState(track_id=1, timestamp_ms=0, position_m=(1.0, 1.0)),
        SpatialActorState(track_id=1, timestamp_ms=100, position_m=(3.0, 3.0)),
        SpatialActorState(track_id=2, timestamp_ms=0, position_m=(1.0, 3.0)),
        SpatialActorState(track_id=2, timestamp_ms=100, position_m=(3.0, 1.0)),
        policy=POLICY,
    )
    assert result[0].primitive_type is MotionPrimitiveType.PATH_CROSSING
    assert result[0].track_ids == (1, 2)


def test_path_crossing_abstains_for_parallel_paths() -> None:
    result = classify_path_crossing(
        SpatialActorState(track_id=1, timestamp_ms=0, position_m=(1.0, 1.0)),
        SpatialActorState(track_id=1, timestamp_ms=100, position_m=(3.0, 1.0)),
        SpatialActorState(track_id=2, timestamp_ms=0, position_m=(1.0, 3.0)),
        SpatialActorState(track_id=2, timestamp_ms=100, position_m=(3.0, 3.0)),
        policy=POLICY,
    )
    assert result == ()


def test_observation_round_trip_and_policy_strictness() -> None:
    item = MotionPrimitiveObservation(
        primitive_type=MotionPrimitiveType.APPROACHING,
        start_ms=100,
        end_ms=100,
        track_ids=(1, 2),
        target_m=None,
        support={"closing_speed_mps": 1.0},
        provenance={},
    )
    assert item.contract_version == P4_MOTION_PRIMITIVE_CONTRACT_VERSION
    assert MotionPrimitiveObservation.from_dict(item.as_dict()).as_dict() == item.as_dict()
    with pytest.raises(P4MotionPrimitiveError):
        MotionPrimitivePolicy(
            pair_radial_speed_min_mps=0.0,
            stationary_speed_max_mps=0.2,
            stationary_window_min_ms=200,
            tangential_acceleration_min_mps2=0.5,
            movement_speed_min_mps=0.5,
            lateral_dominance_ratio=1.5,
            target_radial_speed_min_mps=0.5,
            path_crossing_epsilon_m=0.0,
        )


def test_manifest_and_import_boundary() -> None:
    manifest = p4_motion_primitive_manifest()
    assert manifest["attack_direction_inference"] is False
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_semantics"] is False

    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / "core/p4_motion_primitives.py").read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
    assert not any(name.startswith("services.p2") or name.startswith("core.p3") for name in imports)
