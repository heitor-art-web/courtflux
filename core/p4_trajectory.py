from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import hypot, isfinite
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from core.p4_court_coordinates import within_court_extent
from core.p4_spatial_contract import SpatialFrame
from core.p4_spatial_eligibility import (
    SpatialEligibilityObservation,
    SpatialEligibilityState,
)

P4_TRAJECTORY_CONTRACT_VERSION = "p4.trajectory.v1"
P4_TRAJECTORY_POLICY_VERSION = "p4.trajectory.policy.v1"

P4_TRAJECTORY_POLICY_FIELDS = frozenset({
    "smoothing_window_points",
    "max_interpolation_gap_frames",
    "max_step_speed_mps",
    "accept_degraded_input",
    "policy_version",
})
P4_TRAJECTORY_POINT_FIELDS = frozenset({
    "track_id",
    "frame_id",
    "timestamp_ms",
    "state",
    "reason",
    "raw_position_m",
    "resolved_position_m",
    "smoothed_position_m",
    "observation_confidence",
    "spatial_eligibility_state",
    "provenance",
})
P4_PLAYER_TRAJECTORY_FIELDS = frozenset({
    "track_id",
    "points",
    "policy",
    "source_spatial_contract_version",
    "source_eligibility_contract_version",
    "provenance",
    "contract_version",
})


class TrajectoryPointState(str, Enum):
    OBSERVED = "observed"
    INTERPOLATED_GAP = "interpolated_gap"
    INTERPOLATED_OUTLIER = "interpolated_outlier"
    GAP = "gap"
    OUTLIER_REJECTED = "outlier_rejected"
    INVALID_INPUT = "invalid_input"


class TrajectoryPointReason(str, Enum):
    OBSERVED = "observed"
    TRACK_NOT_VISIBLE = "track_not_visible"
    METRIC_POSITION_MISSING = "metric_position_missing"
    OUT_OF_BOUNDS_METRIC_POSITION = "out_of_bounds_metric_position"
    SPATIAL_INPUT_INVALID = "spatial_input_invalid"
    DEGRADED_INPUT_REJECTED = "degraded_input_rejected"
    TELEPORT_SPEED_EXCEEDED = "teleport_speed_exceeded"
    INTERPOLATED_GAP = "interpolated_gap"
    INTERPOLATED_OUTLIER = "interpolated_outlier"


class P4TrajectoryError(ValueError):
    """Raised when trajectory inputs violate the frozen P4 trajectory contract."""


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise P4TrajectoryError(f"{field_name} must be a non-negative integer")
    number = int(value)
    if number < 0:
        raise P4TrajectoryError(f"{field_name} must be a non-negative integer")
    return number


def _positive_int(value: Any, field_name: str) -> int:
    number = _nonnegative_int(value, field_name)
    if number < 1:
        raise P4TrajectoryError(f"{field_name} must be >= 1")
    return number


def _positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4TrajectoryError(f"{field_name} must be a positive finite number")
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise P4TrajectoryError(f"{field_name} must be a positive finite number")
    return number


def _probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4TrajectoryError(f"{field_name} must be a finite probability")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise P4TrajectoryError(f"{field_name} must be within [0, 1]")
    return number


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise P4TrajectoryError(f"{field_name} must be a boolean")
    return value


def _optional_point(value: Any, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise P4TrajectoryError(f"{field_name} must contain exactly two coordinates")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise P4TrajectoryError(f"{field_name} must contain finite numbers")
        number = float(item)
        if not isfinite(number):
            raise P4TrajectoryError(f"{field_name} must contain finite numbers")
        result.append(number)
    return result[0], result[1]


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P4TrajectoryError(f"{field_name} must be a JSON object")
    normalized = dict(value)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise P4TrajectoryError(f"{field_name} must be JSON serializable") from exc
    return normalized


def _exact_fields(payload: Mapping[str, Any], expected: frozenset[str], object_name: str) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise P4TrajectoryError(
            f"{object_name} fields must match frozen v1 schema "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )


@dataclass(frozen=True, slots=True)
class TrajectoryPolicy:
    smoothing_window_points: int
    max_interpolation_gap_frames: int
    max_step_speed_mps: float
    accept_degraded_input: bool
    policy_version: str = P4_TRAJECTORY_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "smoothing_window_points",
            _positive_int(self.smoothing_window_points, "smoothing_window_points"),
        )
        object.__setattr__(
            self,
            "max_interpolation_gap_frames",
            _nonnegative_int(
                self.max_interpolation_gap_frames,
                "max_interpolation_gap_frames",
            ),
        )
        object.__setattr__(
            self,
            "max_step_speed_mps",
            _positive_number(self.max_step_speed_mps, "max_step_speed_mps"),
        )
        object.__setattr__(
            self,
            "accept_degraded_input",
            _boolean(self.accept_degraded_input, "accept_degraded_input"),
        )
        if self.policy_version != P4_TRAJECTORY_POLICY_VERSION:
            raise P4TrajectoryError(
                f"policy_version must be {P4_TRAJECTORY_POLICY_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "smoothing_window_points": self.smoothing_window_points,
            "max_interpolation_gap_frames": self.max_interpolation_gap_frames,
            "max_step_speed_mps": self.max_step_speed_mps,
            "accept_degraded_input": self.accept_degraded_input,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryPolicy":
        _exact_fields(payload, P4_TRAJECTORY_POLICY_FIELDS, "TrajectoryPolicy")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    track_id: int
    frame_id: int
    timestamp_ms: int
    state: TrajectoryPointState
    reason: TrajectoryPointReason
    raw_position_m: tuple[float, float] | None
    resolved_position_m: tuple[float, float] | None
    smoothed_position_m: tuple[float, float] | None
    observation_confidence: float | None
    spatial_eligibility_state: SpatialEligibilityState
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonnegative_int(self.track_id, "track_id"))
        object.__setattr__(self, "frame_id", _nonnegative_int(self.frame_id, "frame_id"))
        object.__setattr__(
            self, "timestamp_ms", _nonnegative_int(self.timestamp_ms, "timestamp_ms")
        )
        if not isinstance(self.state, TrajectoryPointState):
            try:
                object.__setattr__(self, "state", TrajectoryPointState(str(self.state)))
            except ValueError as exc:
                raise P4TrajectoryError("state is not supported") from exc
        if not isinstance(self.reason, TrajectoryPointReason):
            try:
                object.__setattr__(self, "reason", TrajectoryPointReason(str(self.reason)))
            except ValueError as exc:
                raise P4TrajectoryError("reason is not supported") from exc
        for field_name in ("raw_position_m", "resolved_position_m", "smoothed_position_m"):
            object.__setattr__(
                self,
                field_name,
                _optional_point(getattr(self, field_name), field_name),
            )
        if self.observation_confidence is not None:
            object.__setattr__(
                self,
                "observation_confidence",
                _probability(self.observation_confidence, "observation_confidence"),
            )
        if not isinstance(self.spatial_eligibility_state, SpatialEligibilityState):
            try:
                object.__setattr__(
                    self,
                    "spatial_eligibility_state",
                    SpatialEligibilityState(str(self.spatial_eligibility_state)),
                )
            except ValueError as exc:
                raise P4TrajectoryError("spatial_eligibility_state is not supported") from exc
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))

    @property
    def usable(self) -> bool:
        return self.smoothed_position_m is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "state": self.state.value,
            "reason": self.reason.value,
            "raw_position_m": (
                list(self.raw_position_m) if self.raw_position_m is not None else None
            ),
            "resolved_position_m": (
                list(self.resolved_position_m)
                if self.resolved_position_m is not None
                else None
            ),
            "smoothed_position_m": (
                list(self.smoothed_position_m)
                if self.smoothed_position_m is not None
                else None
            ),
            "observation_confidence": self.observation_confidence,
            "spatial_eligibility_state": self.spatial_eligibility_state.value,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryPoint":
        _exact_fields(payload, P4_TRAJECTORY_POINT_FIELDS, "TrajectoryPoint")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class PlayerTrajectory:
    track_id: int
    points: tuple[TrajectoryPoint, ...]
    policy: TrajectoryPolicy
    source_spatial_contract_version: str
    source_eligibility_contract_version: str
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_TRAJECTORY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonnegative_int(self.track_id, "track_id"))
        points = tuple(self.points)
        if not points:
            raise P4TrajectoryError("points must not be empty")
        if not all(isinstance(point, TrajectoryPoint) for point in points):
            raise P4TrajectoryError("points must contain TrajectoryPoint values")
        if any(point.track_id != self.track_id for point in points):
            raise P4TrajectoryError("all trajectory points must share track_id")
        for previous, current in zip(points, points[1:]):
            if current.frame_id <= previous.frame_id:
                raise P4TrajectoryError("trajectory frame_id values must be strictly increasing")
            if current.timestamp_ms <= previous.timestamp_ms:
                raise P4TrajectoryError("trajectory timestamps must be strictly increasing")
        object.__setattr__(self, "points", points)
        if not isinstance(self.policy, TrajectoryPolicy):
            raise P4TrajectoryError("policy must be TrajectoryPolicy")
        for field_name in (
            "source_spatial_contract_version",
            "source_eligibility_contract_version",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise P4TrajectoryError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))
        if self.contract_version != P4_TRAJECTORY_CONTRACT_VERSION:
            raise P4TrajectoryError(
                f"contract_version must be {P4_TRAJECTORY_CONTRACT_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "points": [point.as_dict() for point in self.points],
            "policy": self.policy.as_dict(),
            "source_spatial_contract_version": self.source_spatial_contract_version,
            "source_eligibility_contract_version": self.source_eligibility_contract_version,
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlayerTrajectory":
        _exact_fields(payload, P4_PLAYER_TRAJECTORY_FIELDS, "PlayerTrajectory")
        normalized = dict(payload)
        normalized["policy"] = TrajectoryPolicy.from_dict(normalized["policy"])
        normalized["points"] = tuple(
            TrajectoryPoint.from_dict(point) for point in normalized["points"]
        )
        return cls(**normalized)


def p4_trajectory_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_TRAJECTORY_CONTRACT_VERSION,
        "policy_version": P4_TRAJECTORY_POLICY_VERSION,
        "point_states": [state.value for state in TrajectoryPointState],
        "point_reasons": [reason.value for reason in TrajectoryPointReason],
        "policy_fields": sorted(P4_TRAJECTORY_POLICY_FIELDS),
        "smoothing": "causal_mean_over_resolved_points",
        "interpolation": "timestamp_linear_for_short_bounded_noninvalid_gaps",
        "teleport_policy": "explicit_max_step_speed_mps",
        "invalid_spatial_input_interpolated": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_semantics": False,
    }


@dataclass(slots=True)
class _WorkingPoint:
    frame_id: int
    timestamp_ms: int
    eligibility_state: SpatialEligibilityState
    raw_position_m: tuple[float, float] | None
    observation_confidence: float | None
    state: TrajectoryPointState
    reason: TrajectoryPointReason
    resolved_position_m: tuple[float, float] | None = None
    smoothed_position_m: tuple[float, float] | None = None


def _validate_sequence(
    frames: Sequence[SpatialFrame],
    eligibilities: Sequence[SpatialEligibilityObservation],
) -> None:
    if len(frames) != len(eligibilities):
        raise P4TrajectoryError("frames and eligibilities must have identical length")
    for index, (frame, eligibility) in enumerate(zip(frames, eligibilities)):
        if not isinstance(frame, SpatialFrame):
            raise P4TrajectoryError("frames must contain SpatialFrame values")
        if not isinstance(eligibility, SpatialEligibilityObservation):
            raise P4TrajectoryError(
                "eligibilities must contain SpatialEligibilityObservation values"
            )
        if frame.frame_id != eligibility.frame_id or frame.timestamp_ms != eligibility.timestamp_ms:
            raise P4TrajectoryError(
                "each eligibility observation must match its SpatialFrame"
            )
        if index:
            previous = frames[index - 1]
            if frame.frame_id <= previous.frame_id:
                raise P4TrajectoryError("frame_id values must be strictly increasing")
            if frame.timestamp_ms <= previous.timestamp_ms:
                raise P4TrajectoryError("timestamps must be strictly increasing")


def _frame_allowed(
    eligibility: SpatialEligibilityObservation,
    policy: TrajectoryPolicy,
) -> bool:
    if eligibility.state is SpatialEligibilityState.INVALID:
        return False
    if eligibility.state is SpatialEligibilityState.DEGRADED:
        return policy.accept_degraded_input
    return True


def _reject_teleports(points: list[_WorkingPoint], policy: TrajectoryPolicy) -> None:
    last_accepted: _WorkingPoint | None = None
    for point in points:
        if point.raw_position_m is None or point.state is not TrajectoryPointState.OBSERVED:
            continue
        if last_accepted is not None and last_accepted.raw_position_m is not None:
            dt_sec = (point.timestamp_ms - last_accepted.timestamp_ms) / 1000.0
            if dt_sec <= 0.0:
                raise P4TrajectoryError("trajectory timestamps must be strictly increasing")
            distance_m = hypot(
                point.raw_position_m[0] - last_accepted.raw_position_m[0],
                point.raw_position_m[1] - last_accepted.raw_position_m[1],
            )
            if distance_m / dt_sec > policy.max_step_speed_mps:
                point.state = TrajectoryPointState.OUTLIER_REJECTED
                point.reason = TrajectoryPointReason.TELEPORT_SPEED_EXCEEDED
                continue
        point.resolved_position_m = point.raw_position_m
        last_accepted = point


def _interpolate_short_gaps(points: list[_WorkingPoint], policy: TrajectoryPolicy) -> None:
    if policy.max_interpolation_gap_frames == 0:
        return
    index = 0
    while index < len(points):
        if points[index].resolved_position_m is not None:
            index += 1
            continue
        start = index
        while index < len(points) and points[index].resolved_position_m is None:
            index += 1
        end = index
        gap = points[start:end]
        if (
            start == 0
            or end >= len(points)
            or len(gap) > policy.max_interpolation_gap_frames
            or any(item.state is TrajectoryPointState.INVALID_INPUT for item in gap)
        ):
            continue
        left = points[start - 1]
        right = points[end]
        if left.resolved_position_m is None or right.resolved_position_m is None:
            continue
        total_ms = right.timestamp_ms - left.timestamp_ms
        if total_ms <= 0:
            continue
        for item in gap:
            if item.state not in (
                TrajectoryPointState.GAP,
                TrajectoryPointState.OUTLIER_REJECTED,
            ):
                continue
            ratio = (item.timestamp_ms - left.timestamp_ms) / total_ms
            item.resolved_position_m = (
                left.resolved_position_m[0]
                + ratio * (right.resolved_position_m[0] - left.resolved_position_m[0]),
                left.resolved_position_m[1]
                + ratio * (right.resolved_position_m[1] - left.resolved_position_m[1]),
            )
            if item.state is TrajectoryPointState.OUTLIER_REJECTED:
                item.state = TrajectoryPointState.INTERPOLATED_OUTLIER
                item.reason = TrajectoryPointReason.INTERPOLATED_OUTLIER
            else:
                item.state = TrajectoryPointState.INTERPOLATED_GAP
                item.reason = TrajectoryPointReason.INTERPOLATED_GAP


def _smooth(points: list[_WorkingPoint], policy: TrajectoryPolicy) -> None:
    history: list[tuple[float, float]] = []
    for point in points:
        if point.resolved_position_m is None:
            history.clear()
            point.smoothed_position_m = None
            continue
        history.append(point.resolved_position_m)
        if len(history) > policy.smoothing_window_points:
            history.pop(0)
        point.smoothed_position_m = (
            sum(value[0] for value in history) / len(history),
            sum(value[1] for value in history) / len(history),
        )


def build_player_trajectories(
    frames: Sequence[SpatialFrame],
    eligibilities: Sequence[SpatialEligibilityObservation],
    *,
    policy: TrajectoryPolicy,
) -> tuple[PlayerTrajectory, ...]:
    """Build deterministic track trajectories from P4 spatial frames only."""

    if not isinstance(policy, TrajectoryPolicy):
        raise P4TrajectoryError("policy must be TrajectoryPolicy")
    frames = tuple(frames)
    eligibilities = tuple(eligibilities)
    _validate_sequence(frames, eligibilities)
    if not frames:
        return ()

    track_ids = sorted(
        {
            player.track_id
            for frame in frames
            for player in frame.players
            if player.visible
        }
    )
    trajectories: list[PlayerTrajectory] = []

    for track_id in track_ids:
        visible_indices = [
            index
            for index, frame in enumerate(frames)
            if any(player.track_id == track_id and player.visible for player in frame.players)
        ]
        if not visible_indices:
            continue
        first_index, last_index = min(visible_indices), max(visible_indices)
        working: list[_WorkingPoint] = []

        for frame, eligibility in zip(
            frames[first_index : last_index + 1],
            eligibilities[first_index : last_index + 1],
        ):
            player = next(
                (
                    candidate
                    for candidate in frame.players
                    if candidate.track_id == track_id and candidate.visible
                ),
                None,
            )
            raw_position: tuple[float, float] | None = None
            confidence: float | None = None
            if eligibility.state is SpatialEligibilityState.INVALID:
                state = TrajectoryPointState.INVALID_INPUT
                reason = TrajectoryPointReason.SPATIAL_INPUT_INVALID
            elif (
                eligibility.state is SpatialEligibilityState.DEGRADED
                and not policy.accept_degraded_input
            ):
                state = TrajectoryPointState.INVALID_INPUT
                reason = TrajectoryPointReason.DEGRADED_INPUT_REJECTED
            elif player is None:
                state = TrajectoryPointState.GAP
                reason = TrajectoryPointReason.TRACK_NOT_VISIBLE
            else:
                confidence = player.observation_confidence
                if player.court_position_m is None:
                    state = TrajectoryPointState.GAP
                    reason = TrajectoryPointReason.METRIC_POSITION_MISSING
                elif not within_court_extent(
                    player.court_position_m,
                    tolerance_m=eligibility.policy.court_extent_tolerance_m,
                ):
                    state = TrajectoryPointState.GAP
                    reason = TrajectoryPointReason.OUT_OF_BOUNDS_METRIC_POSITION
                else:
                    raw_position = tuple(player.court_position_m)
                    state = TrajectoryPointState.OBSERVED
                    reason = TrajectoryPointReason.OBSERVED

            working.append(
                _WorkingPoint(
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    eligibility_state=eligibility.state,
                    raw_position_m=raw_position,
                    observation_confidence=confidence,
                    state=state,
                    reason=reason,
                )
            )

        _reject_teleports(working, policy)
        _interpolate_short_gaps(working, policy)
        _smooth(working, policy)

        points = tuple(
            TrajectoryPoint(
                track_id=track_id,
                frame_id=item.frame_id,
                timestamp_ms=item.timestamp_ms,
                state=item.state,
                reason=item.reason,
                raw_position_m=item.raw_position_m,
                resolved_position_m=item.resolved_position_m,
                smoothed_position_m=item.smoothed_position_m,
                observation_confidence=item.observation_confidence,
                spatial_eligibility_state=item.eligibility_state,
                provenance={
                    "builder": "core.p4_trajectory.build_player_trajectories",
                    "p2_consumed": False,
                    "p3_consumed": False,
                    "tactical_semantics": False,
                },
            )
            for item in working
        )
        trajectories.append(
            PlayerTrajectory(
                track_id=track_id,
                points=points,
                policy=policy,
                source_spatial_contract_version=frames[0].contract_version,
                source_eligibility_contract_version=eligibilities[0].contract_version,
                provenance={
                    "builder": "core.p4_trajectory.build_player_trajectories",
                    "coordinate_space": "canonical_fiba_metres",
                    "p2_consumed": False,
                    "p3_consumed": False,
                    "tactical_semantics": False,
                },
            )
        )
    return tuple(trajectories)
