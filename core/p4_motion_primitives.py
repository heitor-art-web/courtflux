from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import hypot, isfinite
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from core.p4_kinematics import KinematicsPoint, KinematicsPointState, PlayerKinematics
from core.p4_spatial_relationships import PairwiseSpatialRelationship, SpatialActorState

P4_MOTION_PRIMITIVE_CONTRACT_VERSION = "p4.motion_primitives.v1"
P4_MOTION_PRIMITIVE_POLICY_VERSION = "p4.motion_primitives.policy.v1"

P4_MOTION_PRIMITIVE_POLICY_FIELDS = frozenset({
    "pair_radial_speed_min_mps",
    "stationary_speed_max_mps",
    "stationary_window_min_ms",
    "tangential_acceleration_min_mps2",
    "movement_speed_min_mps",
    "lateral_dominance_ratio",
    "target_radial_speed_min_mps",
    "path_crossing_epsilon_m",
    "policy_version",
})
P4_MOTION_PRIMITIVE_FIELDS = frozenset({
    "primitive_type",
    "start_ms",
    "end_ms",
    "track_ids",
    "target_m",
    "support",
    "provenance",
    "contract_version",
})


class MotionPrimitiveType(str, Enum):
    APPROACHING = "approaching"
    SEPARATING = "separating"
    STATIONARY_WINDOW = "stationary_window"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    LATERAL_MOVEMENT = "lateral_movement"
    MOVING_TOWARD_TARGET = "moving_toward_target"
    MOVING_AWAY_FROM_TARGET = "moving_away_from_target"
    PATH_CROSSING = "path_crossing"


class P4MotionPrimitiveError(ValueError):
    """Raised when generic motion primitive inputs violate the frozen P4 contract."""


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise P4MotionPrimitiveError(f"{field_name} must be a non-negative integer")
    number = int(value)
    if number < 0:
        raise P4MotionPrimitiveError(f"{field_name} must be a non-negative integer")
    return number


def _positive_int(value: Any, field_name: str) -> int:
    number = _nonnegative_int(value, field_name)
    if number < 1:
        raise P4MotionPrimitiveError(f"{field_name} must be >= 1")
    return number


def _nonnegative_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4MotionPrimitiveError(f"{field_name} must be a non-negative finite number")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise P4MotionPrimitiveError(f"{field_name} must be a non-negative finite number")
    return number


def _positive_number(value: Any, field_name: str) -> float:
    number = _nonnegative_number(value, field_name)
    if number <= 0.0:
        raise P4MotionPrimitiveError(f"{field_name} must be > 0")
    return number


def _point2(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise P4MotionPrimitiveError(f"{field_name} must contain exactly two coordinates")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise P4MotionPrimitiveError(f"{field_name} must contain finite numbers")
        number = float(item)
        if not isfinite(number):
            raise P4MotionPrimitiveError(f"{field_name} must contain finite numbers")
        result.append(number)
    return result[0], result[1]


def _optional_point2(value: Any, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    return _point2(value, field_name)


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P4MotionPrimitiveError(f"{field_name} must be a JSON object")
    normalized = dict(value)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise P4MotionPrimitiveError(f"{field_name} must be JSON serializable") from exc
    return normalized


def _exact_fields(payload: Mapping[str, Any], expected: frozenset[str], object_name: str) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise P4MotionPrimitiveError(
            f"{object_name} fields must match frozen v1 schema "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )


@dataclass(frozen=True, slots=True)
class MotionPrimitivePolicy:
    pair_radial_speed_min_mps: float
    stationary_speed_max_mps: float
    stationary_window_min_ms: int
    tangential_acceleration_min_mps2: float
    movement_speed_min_mps: float
    lateral_dominance_ratio: float
    target_radial_speed_min_mps: float
    path_crossing_epsilon_m: float
    policy_version: str = P4_MOTION_PRIMITIVE_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_radial_speed_min_mps", _positive_number(self.pair_radial_speed_min_mps, "pair_radial_speed_min_mps"))
        object.__setattr__(self, "stationary_speed_max_mps", _nonnegative_number(self.stationary_speed_max_mps, "stationary_speed_max_mps"))
        object.__setattr__(self, "stationary_window_min_ms", _positive_int(self.stationary_window_min_ms, "stationary_window_min_ms"))
        object.__setattr__(self, "tangential_acceleration_min_mps2", _positive_number(self.tangential_acceleration_min_mps2, "tangential_acceleration_min_mps2"))
        object.__setattr__(self, "movement_speed_min_mps", _positive_number(self.movement_speed_min_mps, "movement_speed_min_mps"))
        object.__setattr__(self, "lateral_dominance_ratio", _positive_number(self.lateral_dominance_ratio, "lateral_dominance_ratio"))
        object.__setattr__(self, "target_radial_speed_min_mps", _positive_number(self.target_radial_speed_min_mps, "target_radial_speed_min_mps"))
        object.__setattr__(self, "path_crossing_epsilon_m", _nonnegative_number(self.path_crossing_epsilon_m, "path_crossing_epsilon_m"))
        if self.policy_version != P4_MOTION_PRIMITIVE_POLICY_VERSION:
            raise P4MotionPrimitiveError(f"policy_version must be {P4_MOTION_PRIMITIVE_POLICY_VERSION!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_radial_speed_min_mps": self.pair_radial_speed_min_mps,
            "stationary_speed_max_mps": self.stationary_speed_max_mps,
            "stationary_window_min_ms": self.stationary_window_min_ms,
            "tangential_acceleration_min_mps2": self.tangential_acceleration_min_mps2,
            "movement_speed_min_mps": self.movement_speed_min_mps,
            "lateral_dominance_ratio": self.lateral_dominance_ratio,
            "target_radial_speed_min_mps": self.target_radial_speed_min_mps,
            "path_crossing_epsilon_m": self.path_crossing_epsilon_m,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MotionPrimitivePolicy":
        _exact_fields(payload, P4_MOTION_PRIMITIVE_POLICY_FIELDS, "MotionPrimitivePolicy")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class MotionPrimitiveObservation:
    primitive_type: MotionPrimitiveType
    start_ms: int
    end_ms: int
    track_ids: tuple[int, ...]
    target_m: tuple[float, float] | None
    support: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_MOTION_PRIMITIVE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.primitive_type, MotionPrimitiveType):
            try:
                object.__setattr__(self, "primitive_type", MotionPrimitiveType(str(self.primitive_type)))
            except ValueError as exc:
                raise P4MotionPrimitiveError("primitive_type is not supported") from exc
        object.__setattr__(self, "start_ms", _nonnegative_int(self.start_ms, "start_ms"))
        object.__setattr__(self, "end_ms", _nonnegative_int(self.end_ms, "end_ms"))
        if self.end_ms < self.start_ms:
            raise P4MotionPrimitiveError("end_ms must be >= start_ms")
        track_ids = tuple(self.track_ids)
        if not track_ids or len(track_ids) != len(set(track_ids)) or any(isinstance(track_id, bool) or not isinstance(track_id, Integral) or int(track_id) < 0 for track_id in track_ids):
            raise P4MotionPrimitiveError("track_ids must contain unique non-negative integer ids")
        object.__setattr__(self, "track_ids", tuple(int(value) for value in track_ids))
        object.__setattr__(self, "target_m", _optional_point2(self.target_m, "target_m"))
        object.__setattr__(self, "support", _json_object(self.support, "support"))
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))
        if self.contract_version != P4_MOTION_PRIMITIVE_CONTRACT_VERSION:
            raise P4MotionPrimitiveError(f"contract_version must be {P4_MOTION_PRIMITIVE_CONTRACT_VERSION!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "primitive_type": self.primitive_type.value,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "track_ids": list(self.track_ids),
            "target_m": list(self.target_m) if self.target_m is not None else None,
            "support": dict(self.support),
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MotionPrimitiveObservation":
        _exact_fields(payload, P4_MOTION_PRIMITIVE_FIELDS, "MotionPrimitiveObservation")
        normalized = dict(payload)
        normalized["track_ids"] = tuple(normalized["track_ids"])
        return cls(**normalized)


def p4_motion_primitive_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_MOTION_PRIMITIVE_CONTRACT_VERSION,
        "policy_version": P4_MOTION_PRIMITIVE_POLICY_VERSION,
        "primitive_types": [item.value for item in MotionPrimitiveType],
        "policy_fields": sorted(P4_MOTION_PRIMITIVE_POLICY_FIELDS),
        "pair_semantics": "track_relationship_only_no_offense_defense",
        "lateral_axis": "canonical_fiba_y_axis_relative_to_longitudinal_x",
        "target_semantics": "caller_supplied_generic_court_point",
        "attack_direction_inference": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_semantics": False,
    }


def _primitive(primitive_type: MotionPrimitiveType, *, start_ms: int, end_ms: int, track_ids: tuple[int, ...], policy: MotionPrimitivePolicy, support: Mapping[str, Any], target_m: tuple[float, float] | None = None, evaluator: str) -> MotionPrimitiveObservation:
    return MotionPrimitiveObservation(
        primitive_type=primitive_type,
        start_ms=start_ms,
        end_ms=end_ms,
        track_ids=track_ids,
        target_m=target_m,
        support={**dict(support), "policy_version": policy.policy_version},
        provenance={"evaluator": evaluator, "p2_consumed": False, "p3_consumed": False, "tactical_semantics": False},
    )


def classify_pair_radial_motion(relationship: PairwiseSpatialRelationship, *, policy: MotionPrimitivePolicy) -> tuple[MotionPrimitiveObservation, ...]:
    if not isinstance(relationship, PairwiseSpatialRelationship):
        raise P4MotionPrimitiveError("relationship must be PairwiseSpatialRelationship")
    if relationship.closing_speed_mps is None:
        return ()
    closing = float(relationship.closing_speed_mps)
    if closing >= policy.pair_radial_speed_min_mps:
        kind = MotionPrimitiveType.APPROACHING
    elif closing <= -policy.pair_radial_speed_min_mps:
        kind = MotionPrimitiveType.SEPARATING
    else:
        return ()
    return (_primitive(kind, start_ms=relationship.timestamp_ms, end_ms=relationship.timestamp_ms, track_ids=(relationship.track_a_id, relationship.track_b_id), policy=policy, support={"distance_m": relationship.distance_m, "closing_speed_mps": closing, "local_team_relation": relationship.local_team_relation.value}, evaluator="core.p4_motion_primitives.classify_pair_radial_motion"),)


def _available_points(kinematics: PlayerKinematics) -> tuple[KinematicsPoint, ...]:
    return tuple(point for point in kinematics.points if point.position_m is not None and point.state is not KinematicsPointState.UNAVAILABLE and point.speed_mps is not None)


def classify_stationary_window(kinematics: PlayerKinematics, *, policy: MotionPrimitivePolicy) -> tuple[MotionPrimitiveObservation, ...]:
    if not isinstance(kinematics, PlayerKinematics):
        raise P4MotionPrimitiveError("kinematics must be PlayerKinematics")
    points = _available_points(kinematics)
    if not points:
        return ()
    end = points[-1]
    window: list[KinematicsPoint] = [end]
    for point in reversed(points[:-1]):
        if end.timestamp_ms - point.timestamp_ms > policy.stationary_window_min_ms:
            break
        window.append(point)
    window.reverse()
    if end.timestamp_ms - window[0].timestamp_ms < policy.stationary_window_min_ms:
        return ()
    if any(point.speed_mps is None or point.speed_mps > policy.stationary_speed_max_mps for point in window):
        return ()
    return (_primitive(MotionPrimitiveType.STATIONARY_WINDOW, start_ms=window[0].timestamp_ms, end_ms=end.timestamp_ms, track_ids=(kinematics.track_id,), policy=policy, support={"max_speed_mps": max(float(point.speed_mps or 0.0) for point in window), "sample_count": len(window), "window_ms": end.timestamp_ms - window[0].timestamp_ms}, evaluator="core.p4_motion_primitives.classify_stationary_window"),)


def classify_kinematic_motion(point: KinematicsPoint, *, policy: MotionPrimitivePolicy) -> tuple[MotionPrimitiveObservation, ...]:
    if not isinstance(point, KinematicsPoint):
        raise P4MotionPrimitiveError("point must be KinematicsPoint")
    if point.state is KinematicsPointState.UNAVAILABLE or point.velocity_mps is None or point.speed_mps is None:
        return ()
    results: list[MotionPrimitiveObservation] = []
    vx, vy = point.velocity_mps
    speed = float(point.speed_mps)
    if point.acceleration_mps2 is not None and speed >= policy.movement_speed_min_mps and speed > 0.0:
        ax, ay = point.acceleration_mps2
        tangential_acceleration = (vx * ax + vy * ay) / speed
        kind = MotionPrimitiveType.ACCELERATING if tangential_acceleration >= policy.tangential_acceleration_min_mps2 else MotionPrimitiveType.DECELERATING if tangential_acceleration <= -policy.tangential_acceleration_min_mps2 else None
        if kind is not None:
            results.append(_primitive(kind, start_ms=point.timestamp_ms, end_ms=point.timestamp_ms, track_ids=(point.track_id,), policy=policy, support={"speed_mps": speed, "tangential_acceleration_mps2": tangential_acceleration}, evaluator="core.p4_motion_primitives.classify_kinematic_motion"))
    if speed >= policy.movement_speed_min_mps and abs(vy) >= policy.lateral_dominance_ratio * abs(vx):
        results.append(_primitive(MotionPrimitiveType.LATERAL_MOVEMENT, start_ms=point.timestamp_ms, end_ms=point.timestamp_ms, track_ids=(point.track_id,), policy=policy, support={"velocity_x_mps": vx, "velocity_y_mps": vy, "speed_mps": speed, "lateral_dominance_ratio": policy.lateral_dominance_ratio}, evaluator="core.p4_motion_primitives.classify_kinematic_motion"))
    return tuple(results)


def classify_target_motion(previous: KinematicsPoint, current: KinematicsPoint, *, target_m: Sequence[float], policy: MotionPrimitivePolicy) -> tuple[MotionPrimitiveObservation, ...]:
    if not isinstance(previous, KinematicsPoint) or not isinstance(current, KinematicsPoint):
        raise P4MotionPrimitiveError("previous/current must be KinematicsPoint values")
    if previous.track_id != current.track_id:
        raise P4MotionPrimitiveError("previous/current must describe the same track")
    if current.timestamp_ms <= previous.timestamp_ms:
        raise P4MotionPrimitiveError("current must be later than previous")
    if previous.position_m is None or current.position_m is None:
        return ()
    target = _point2(target_m, "target_m")
    previous_distance = hypot(target[0] - previous.position_m[0], target[1] - previous.position_m[1])
    current_distance = hypot(target[0] - current.position_m[0], target[1] - current.position_m[1])
    dt_sec = (current.timestamp_ms - previous.timestamp_ms) / 1000.0
    radial_speed = (previous_distance - current_distance) / dt_sec
    if radial_speed >= policy.target_radial_speed_min_mps:
        kind = MotionPrimitiveType.MOVING_TOWARD_TARGET
    elif radial_speed <= -policy.target_radial_speed_min_mps:
        kind = MotionPrimitiveType.MOVING_AWAY_FROM_TARGET
    else:
        return ()
    return (_primitive(kind, start_ms=previous.timestamp_ms, end_ms=current.timestamp_ms, track_ids=(current.track_id,), target_m=target, policy=policy, support={"previous_distance_m": previous_distance, "current_distance_m": current_distance, "radial_speed_mps": radial_speed}, evaluator="core.p4_motion_primitives.classify_target_motion"),)


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float], epsilon: float) -> bool:
    return bool(min(a[0], b[0]) - epsilon <= p[0] <= max(a[0], b[0]) + epsilon and min(a[1], b[1]) - epsilon <= p[1] <= max(a[1], b[1]) + epsilon and abs(_orientation(a, b, p)) <= epsilon)


def _segments_cross(a0: tuple[float, float], a1: tuple[float, float], b0: tuple[float, float], b1: tuple[float, float], epsilon: float) -> bool:
    o1 = _orientation(a0, a1, b0)
    o2 = _orientation(a0, a1, b1)
    o3 = _orientation(b0, b1, a0)
    o4 = _orientation(b0, b1, a1)
    if (((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon)) and ((o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon))):
        return True
    return any((abs(o1) <= epsilon and _on_segment(a0, a1, b0, epsilon), abs(o2) <= epsilon and _on_segment(a0, a1, b1, epsilon), abs(o3) <= epsilon and _on_segment(b0, b1, a0, epsilon), abs(o4) <= epsilon and _on_segment(b0, b1, a1, epsilon)))


def classify_path_crossing(previous_a: SpatialActorState, current_a: SpatialActorState, previous_b: SpatialActorState, current_b: SpatialActorState, *, policy: MotionPrimitivePolicy) -> tuple[MotionPrimitiveObservation, ...]:
    actors = (previous_a, current_a, previous_b, current_b)
    if not all(isinstance(actor, SpatialActorState) for actor in actors):
        raise P4MotionPrimitiveError("path crossing inputs must be SpatialActorState values")
    if previous_a.track_id != current_a.track_id or previous_b.track_id != current_b.track_id:
        raise P4MotionPrimitiveError("each actor path must preserve track_id")
    if previous_a.track_id == previous_b.track_id:
        raise P4MotionPrimitiveError("path crossing requires two different tracks")
    if previous_a.timestamp_ms != previous_b.timestamp_ms:
        raise P4MotionPrimitiveError("previous actor states must share timestamp")
    if current_a.timestamp_ms != current_b.timestamp_ms:
        raise P4MotionPrimitiveError("current actor states must share timestamp")
    if current_a.timestamp_ms <= previous_a.timestamp_ms:
        raise P4MotionPrimitiveError("current states must be later than previous states")
    if not _segments_cross(previous_a.position_m, current_a.position_m, previous_b.position_m, current_b.position_m, policy.path_crossing_epsilon_m):
        return ()
    return (_primitive(MotionPrimitiveType.PATH_CROSSING, start_ms=previous_a.timestamp_ms, end_ms=current_a.timestamp_ms, track_ids=(previous_a.track_id, previous_b.track_id), policy=policy, support={"path_a_start_m": list(previous_a.position_m), "path_a_end_m": list(current_a.position_m), "path_b_start_m": list(previous_b.position_m), "path_b_end_m": list(current_b.position_m), "epsilon_m": policy.path_crossing_epsilon_m}, evaluator="core.p4_motion_primitives.classify_path_crossing"),)
