from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import atan2, hypot, isfinite
from numbers import Integral
from typing import Any, Mapping

from core.p4_trajectory import (
    P4_TRAJECTORY_CONTRACT_VERSION,
    PlayerTrajectory,
    TrajectoryPointState,
)

P4_KINEMATICS_CONTRACT_VERSION = "p4.kinematics.v1"
P4_KINEMATICS_POLICY_VERSION = "p4.kinematics.policy.v1"

P4_KINEMATICS_POLICY_FIELDS = frozenset({
    "max_derivative_gap_ms",
    "use_interpolated_points",
    "policy_version",
})
P4_KINEMATICS_POINT_FIELDS = frozenset({
    "track_id",
    "frame_id",
    "timestamp_ms",
    "state",
    "trajectory_point_state",
    "position_m",
    "velocity_mps",
    "speed_mps",
    "acceleration_mps2",
    "acceleration_magnitude_mps2",
    "heading_rad",
    "provenance",
})
P4_PLAYER_KINEMATICS_FIELDS = frozenset({
    "track_id",
    "points",
    "policy",
    "source_trajectory_contract_version",
    "provenance",
    "contract_version",
})


class KinematicsPointState(str, Enum):
    UNAVAILABLE = "unavailable"
    VELOCITY_ONLY = "velocity_only"
    FULL = "full"


class P4KinematicsError(ValueError):
    """Raised when kinematic inputs violate the frozen P4 kinematics contract."""


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise P4KinematicsError(f"{field_name} must be a non-negative integer")
    number = int(value)
    if number < 0:
        raise P4KinematicsError(f"{field_name} must be a non-negative integer")
    return number


def _positive_int(value: Any, field_name: str) -> int:
    number = _nonnegative_int(value, field_name)
    if number < 1:
        raise P4KinematicsError(f"{field_name} must be >= 1")
    return number


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise P4KinematicsError(f"{field_name} must be a boolean")
    return value


def _optional_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise P4KinematicsError(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise P4KinematicsError(f"{field_name} must be a finite number") from exc
    if not isfinite(number):
        raise P4KinematicsError(f"{field_name} must be a finite number")
    return number


def _optional_nonnegative_number(value: Any, field_name: str) -> float | None:
    number = _optional_number(value, field_name)
    if number is not None and number < 0.0:
        raise P4KinematicsError(f"{field_name} must be >= 0")
    return number


def _finite_number(value: Any, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        raise P4KinematicsError(f"{field_name} must contain finite numbers")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise P4KinematicsError(f"{field_name} must contain finite numbers") from exc
    if not isfinite(number):
        raise P4KinematicsError(f"{field_name} must contain finite numbers")
    return number


def _optional_vector2(value: Any, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise P4KinematicsError(f"{field_name} must contain exactly two numbers")
    return (
        _finite_number(value[0], field_name),
        _finite_number(value[1], field_name),
    )


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P4KinematicsError(f"{field_name} must be a JSON object")
    normalized = dict(value)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise P4KinematicsError(f"{field_name} must be JSON serializable") from exc
    return normalized


def _exact_fields(payload: Mapping[str, Any], expected: frozenset[str], object_name: str) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise P4KinematicsError(
            f"{object_name} fields must match frozen v1 schema "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )


@dataclass(frozen=True, slots=True)
class KinematicsPolicy:
    max_derivative_gap_ms: int
    use_interpolated_points: bool
    policy_version: str = P4_KINEMATICS_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_derivative_gap_ms",
            _positive_int(self.max_derivative_gap_ms, "max_derivative_gap_ms"),
        )
        object.__setattr__(
            self,
            "use_interpolated_points",
            _boolean(self.use_interpolated_points, "use_interpolated_points"),
        )
        if self.policy_version != P4_KINEMATICS_POLICY_VERSION:
            raise P4KinematicsError(
                f"policy_version must be {P4_KINEMATICS_POLICY_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_derivative_gap_ms": self.max_derivative_gap_ms,
            "use_interpolated_points": self.use_interpolated_points,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KinematicsPolicy":
        _exact_fields(payload, P4_KINEMATICS_POLICY_FIELDS, "KinematicsPolicy")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class KinematicsPoint:
    track_id: int
    frame_id: int
    timestamp_ms: int
    state: KinematicsPointState
    trajectory_point_state: TrajectoryPointState
    position_m: tuple[float, float] | None
    velocity_mps: tuple[float, float] | None
    speed_mps: float | None
    acceleration_mps2: tuple[float, float] | None
    acceleration_magnitude_mps2: float | None
    heading_rad: float | None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonnegative_int(self.track_id, "track_id"))
        object.__setattr__(self, "frame_id", _nonnegative_int(self.frame_id, "frame_id"))
        object.__setattr__(
            self, "timestamp_ms", _nonnegative_int(self.timestamp_ms, "timestamp_ms")
        )
        if not isinstance(self.state, KinematicsPointState):
            try:
                object.__setattr__(self, "state", KinematicsPointState(str(self.state)))
            except ValueError as exc:
                raise P4KinematicsError("state is not supported") from exc
        if not isinstance(self.trajectory_point_state, TrajectoryPointState):
            try:
                object.__setattr__(
                    self,
                    "trajectory_point_state",
                    TrajectoryPointState(str(self.trajectory_point_state)),
                )
            except ValueError as exc:
                raise P4KinematicsError("trajectory_point_state is not supported") from exc
        object.__setattr__(self, "position_m", _optional_vector2(self.position_m, "position_m"))
        object.__setattr__(
            self, "velocity_mps", _optional_vector2(self.velocity_mps, "velocity_mps")
        )
        object.__setattr__(
            self, "speed_mps", _optional_nonnegative_number(self.speed_mps, "speed_mps")
        )
        object.__setattr__(
            self,
            "acceleration_mps2",
            _optional_vector2(self.acceleration_mps2, "acceleration_mps2"),
        )
        object.__setattr__(
            self,
            "acceleration_magnitude_mps2",
            _optional_nonnegative_number(
                self.acceleration_magnitude_mps2,
                "acceleration_magnitude_mps2",
            ),
        )
        object.__setattr__(
            self, "heading_rad", _optional_number(self.heading_rad, "heading_rad")
        )
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))

        if self.state is KinematicsPointState.UNAVAILABLE:
            if any(
                value is not None
                for value in (
                    self.velocity_mps,
                    self.speed_mps,
                    self.acceleration_mps2,
                    self.acceleration_magnitude_mps2,
                    self.heading_rad,
                )
            ):
                raise P4KinematicsError("UNAVAILABLE point cannot contain derivatives")
        elif self.velocity_mps is None or self.speed_mps is None:
            raise P4KinematicsError("available kinematics requires velocity and speed")
        if self.state is KinematicsPointState.FULL and (
            self.acceleration_mps2 is None or self.acceleration_magnitude_mps2 is None
        ):
            raise P4KinematicsError("FULL kinematics requires acceleration")

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "state": self.state.value,
            "trajectory_point_state": self.trajectory_point_state.value,
            "position_m": list(self.position_m) if self.position_m is not None else None,
            "velocity_mps": list(self.velocity_mps) if self.velocity_mps is not None else None,
            "speed_mps": self.speed_mps,
            "acceleration_mps2": (
                list(self.acceleration_mps2)
                if self.acceleration_mps2 is not None
                else None
            ),
            "acceleration_magnitude_mps2": self.acceleration_magnitude_mps2,
            "heading_rad": self.heading_rad,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KinematicsPoint":
        _exact_fields(payload, P4_KINEMATICS_POINT_FIELDS, "KinematicsPoint")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class PlayerKinematics:
    track_id: int
    points: tuple[KinematicsPoint, ...]
    policy: KinematicsPolicy
    source_trajectory_contract_version: str
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_KINEMATICS_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonnegative_int(self.track_id, "track_id"))
        points = tuple(self.points)
        if not points:
            raise P4KinematicsError("points must not be empty")
        if not all(isinstance(point, KinematicsPoint) for point in points):
            raise P4KinematicsError("points must contain KinematicsPoint values")
        if any(point.track_id != self.track_id for point in points):
            raise P4KinematicsError("all kinematics points must share track_id")
        for previous, current in zip(points, points[1:]):
            if current.frame_id <= previous.frame_id:
                raise P4KinematicsError("frame_id values must be strictly increasing")
            if current.timestamp_ms <= previous.timestamp_ms:
                raise P4KinematicsError("timestamps must be strictly increasing")
        object.__setattr__(self, "points", points)
        if not isinstance(self.policy, KinematicsPolicy):
            raise P4KinematicsError("policy must be KinematicsPolicy")
        source_version = str(self.source_trajectory_contract_version).strip()
        if not source_version:
            raise P4KinematicsError("source_trajectory_contract_version must be non-empty")
        object.__setattr__(
            self, "source_trajectory_contract_version", source_version
        )
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))
        if self.contract_version != P4_KINEMATICS_CONTRACT_VERSION:
            raise P4KinematicsError(
                f"contract_version must be {P4_KINEMATICS_CONTRACT_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "points": [point.as_dict() for point in self.points],
            "policy": self.policy.as_dict(),
            "source_trajectory_contract_version": self.source_trajectory_contract_version,
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlayerKinematics":
        _exact_fields(payload, P4_PLAYER_KINEMATICS_FIELDS, "PlayerKinematics")
        normalized = dict(payload)
        normalized["policy"] = KinematicsPolicy.from_dict(normalized["policy"])
        normalized["points"] = tuple(
            KinematicsPoint.from_dict(point) for point in normalized["points"]
        )
        return cls(**normalized)


def p4_kinematics_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_KINEMATICS_CONTRACT_VERSION,
        "policy_version": P4_KINEMATICS_POLICY_VERSION,
        "states": [state.value for state in KinematicsPointState],
        "policy_fields": sorted(P4_KINEMATICS_POLICY_FIELDS),
        "position_source": "trajectory.smoothed_position_m",
        "velocity_method": "causal_finite_difference",
        "acceleration_method": "difference_of_consecutive_velocity_estimates",
        "heading_units": "radians",
        "stationary_classification": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_semantics": False,
    }


_INTERPOLATED_STATES = {
    TrajectoryPointState.INTERPOLATED_GAP,
    TrajectoryPointState.INTERPOLATED_OUTLIER,
}


def _point_allowed(point: Any, policy: KinematicsPolicy) -> bool:
    if point.smoothed_position_m is None:
        return False
    if point.state in _INTERPOLATED_STATES and not policy.use_interpolated_points:
        return False
    return True


def build_player_kinematics(
    trajectory: PlayerTrajectory,
    *,
    policy: KinematicsPolicy,
) -> PlayerKinematics:
    """Derive velocity, speed, acceleration and heading from one smoothed trajectory."""

    if not isinstance(trajectory, PlayerTrajectory):
        raise P4KinematicsError("trajectory must be PlayerTrajectory")
    if not isinstance(policy, KinematicsPolicy):
        raise P4KinematicsError("policy must be KinematicsPolicy")

    results: list[KinematicsPoint] = []
    previous_usable = None
    previous_velocity: tuple[float, float] | None = None

    for point in trajectory.points:
        position = point.smoothed_position_m
        state = KinematicsPointState.UNAVAILABLE
        velocity = None
        speed = None
        acceleration = None
        acceleration_magnitude = None
        heading = None

        usable = _point_allowed(point, policy)
        if usable and previous_usable is not None:
            dt_ms = point.timestamp_ms - previous_usable.timestamp_ms
            if 0 < dt_ms <= policy.max_derivative_gap_ms:
                dt_sec = dt_ms / 1000.0
                previous_position = previous_usable.smoothed_position_m
                if previous_position is not None and position is not None:
                    velocity = (
                        (position[0] - previous_position[0]) / dt_sec,
                        (position[1] - previous_position[1]) / dt_sec,
                    )
                    speed = hypot(*velocity)
                    heading = atan2(velocity[1], velocity[0]) if speed > 0.0 else None
                    state = KinematicsPointState.VELOCITY_ONLY

                    if previous_velocity is not None:
                        acceleration = (
                            (velocity[0] - previous_velocity[0]) / dt_sec,
                            (velocity[1] - previous_velocity[1]) / dt_sec,
                        )
                        acceleration_magnitude = hypot(*acceleration)
                        state = KinematicsPointState.FULL

        results.append(
            KinematicsPoint(
                track_id=trajectory.track_id,
                frame_id=point.frame_id,
                timestamp_ms=point.timestamp_ms,
                state=state,
                trajectory_point_state=point.state,
                position_m=position,
                velocity_mps=velocity,
                speed_mps=speed,
                acceleration_mps2=acceleration,
                acceleration_magnitude_mps2=acceleration_magnitude,
                heading_rad=heading,
                provenance={
                    "builder": "core.p4_kinematics.build_player_kinematics",
                    "source_trajectory_contract_version": trajectory.contract_version,
                    "p2_consumed": False,
                    "p3_consumed": False,
                    "tactical_semantics": False,
                },
            )
        )

        if usable:
            if velocity is None:
                previous_velocity = None
            else:
                previous_velocity = velocity
            previous_usable = point
        else:
            previous_usable = None
            previous_velocity = None

    return PlayerKinematics(
        track_id=trajectory.track_id,
        points=tuple(results),
        policy=policy,
        source_trajectory_contract_version=trajectory.contract_version,
        provenance={
            "builder": "core.p4_kinematics.build_player_kinematics",
            "coordinate_space": "canonical_fiba_metres",
            "p2_consumed": False,
            "p3_consumed": False,
            "tactical_semantics": False,
        },
    )
