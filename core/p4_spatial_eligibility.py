from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite
from numbers import Integral, Real
from typing import Any, Mapping

from core.p4_court_coordinates import P4_FIBA_COURT_COORDINATE_SYSTEM, within_court_extent
from core.p4_geometry_quality import GeometryQualityObservation, GeometryQualityState
from core.p4_spatial_contract import SpatialFrame

P4_SPATIAL_ELIGIBILITY_CONTRACT_VERSION = "p4.spatial_eligibility.v1"
P4_SPATIAL_ELIGIBILITY_POLICY_VERSION = "p4.spatial_eligibility.policy.v1"

P4_SPATIAL_ELIGIBILITY_FIELDS = frozenset({
    "frame_id",
    "timestamp_ms",
    "state",
    "reasons",
    "geometry_quality_state",
    "visible_player_count",
    "metric_player_count",
    "missing_metric_player_count",
    "out_of_bounds_metric_player_count",
    "metric_player_fraction",
    "policy",
    "provenance",
    "contract_version",
})
P4_SPATIAL_ELIGIBILITY_POLICY_FIELDS = frozenset({
    "min_metric_player_count",
    "min_metric_player_fraction",
    "court_extent_tolerance_m",
    "allow_aging_geometry",
    "require_court_visible",
    "policy_version",
})


class SpatialEligibilityState(str, Enum):
    ELIGIBLE = "eligible"
    DEGRADED = "degraded"
    INVALID = "invalid"


class SpatialEligibilityReason(str, Enum):
    ELIGIBLE = "eligible"
    GEOMETRY_INVALID = "geometry_invalid"
    COURT_NOT_VISIBLE = "court_not_visible"
    COORDINATE_SYSTEM_MISMATCH = "coordinate_system_mismatch"
    METRIC_PLAYER_COUNT_BELOW_MIN = "metric_player_count_below_min"
    METRIC_PLAYER_FRACTION_BELOW_MIN = "metric_player_fraction_below_min"
    AGING_GEOMETRY = "aging_geometry"
    MISSING_METRIC_PLAYER_POSITION = "missing_metric_player_position"
    OUT_OF_BOUNDS_METRIC_POSITION = "out_of_bounds_metric_position"


class P4SpatialEligibilityError(ValueError):
    """Raised when eligibility inputs violate the frozen P4 spatial gate contract."""


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise P4SpatialEligibilityError(f"{field_name} must be a non-negative integer")
    number = int(value)
    if number < 0:
        raise P4SpatialEligibilityError(f"{field_name} must be a non-negative integer")
    return number


def _probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4SpatialEligibilityError(f"{field_name} must be a finite probability")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise P4SpatialEligibilityError(f"{field_name} must be within [0, 1]")
    return number


def _nonnegative_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4SpatialEligibilityError(f"{field_name} must be a non-negative finite number")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise P4SpatialEligibilityError(f"{field_name} must be a non-negative finite number")
    return number


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise P4SpatialEligibilityError(f"{field_name} must be a boolean")
    return value


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P4SpatialEligibilityError(f"{field_name} must be a JSON object")
    normalized = dict(value)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise P4SpatialEligibilityError(f"{field_name} must be JSON serializable") from exc
    return normalized


def _exact_fields(payload: Mapping[str, Any], expected: frozenset[str], object_name: str) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise P4SpatialEligibilityError(
            f"{object_name} fields must match frozen v1 schema "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )


@dataclass(frozen=True, slots=True)
class SpatialEligibilityPolicy:
    """Explicit, versioned policy for deciding whether metric spatial state is usable."""

    min_metric_player_count: int
    min_metric_player_fraction: float
    court_extent_tolerance_m: float
    allow_aging_geometry: bool
    require_court_visible: bool
    policy_version: str = P4_SPATIAL_ELIGIBILITY_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_metric_player_count", _nonnegative_int(
            self.min_metric_player_count, "min_metric_player_count"
        ))
        object.__setattr__(self, "min_metric_player_fraction", _probability(
            self.min_metric_player_fraction, "min_metric_player_fraction"
        ))
        object.__setattr__(self, "court_extent_tolerance_m", _nonnegative_number(
            self.court_extent_tolerance_m, "court_extent_tolerance_m"
        ))
        object.__setattr__(self, "allow_aging_geometry", _boolean(
            self.allow_aging_geometry, "allow_aging_geometry"
        ))
        object.__setattr__(self, "require_court_visible", _boolean(
            self.require_court_visible, "require_court_visible"
        ))
        if self.policy_version != P4_SPATIAL_ELIGIBILITY_POLICY_VERSION:
            raise P4SpatialEligibilityError(
                f"policy_version must be {P4_SPATIAL_ELIGIBILITY_POLICY_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_metric_player_count": self.min_metric_player_count,
            "min_metric_player_fraction": self.min_metric_player_fraction,
            "court_extent_tolerance_m": self.court_extent_tolerance_m,
            "allow_aging_geometry": self.allow_aging_geometry,
            "require_court_visible": self.require_court_visible,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialEligibilityPolicy":
        _exact_fields(payload, P4_SPATIAL_ELIGIBILITY_POLICY_FIELDS, "SpatialEligibilityPolicy")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class SpatialEligibilityObservation:
    frame_id: int
    timestamp_ms: int
    state: SpatialEligibilityState
    reasons: tuple[SpatialEligibilityReason, ...]
    geometry_quality_state: GeometryQualityState
    visible_player_count: int
    metric_player_count: int
    missing_metric_player_count: int
    out_of_bounds_metric_player_count: int
    metric_player_fraction: float
    policy: SpatialEligibilityPolicy
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_SPATIAL_ELIGIBILITY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _nonnegative_int(self.frame_id, "frame_id"))
        object.__setattr__(self, "timestamp_ms", _nonnegative_int(self.timestamp_ms, "timestamp_ms"))
        if not isinstance(self.state, SpatialEligibilityState):
            try:
                object.__setattr__(self, "state", SpatialEligibilityState(str(self.state)))
            except ValueError as exc:
                raise P4SpatialEligibilityError("state is not supported") from exc
        reasons: list[SpatialEligibilityReason] = []
        for reason in tuple(self.reasons):
            try:
                normalized = reason if isinstance(reason, SpatialEligibilityReason) else SpatialEligibilityReason(str(reason))
            except ValueError as exc:
                raise P4SpatialEligibilityError("reason is not supported") from exc
            if normalized not in reasons:
                reasons.append(normalized)
        if not reasons:
            raise P4SpatialEligibilityError("reasons must not be empty")
        object.__setattr__(self, "reasons", tuple(reasons))
        if not isinstance(self.geometry_quality_state, GeometryQualityState):
            try:
                object.__setattr__(
                    self, "geometry_quality_state", GeometryQualityState(str(self.geometry_quality_state))
                )
            except ValueError as exc:
                raise P4SpatialEligibilityError("geometry_quality_state is not supported") from exc
        for field_name in (
            "visible_player_count",
            "metric_player_count",
            "missing_metric_player_count",
            "out_of_bounds_metric_player_count",
        ):
            object.__setattr__(self, field_name, _nonnegative_int(getattr(self, field_name), field_name))
        object.__setattr__(
            self, "metric_player_fraction", _probability(self.metric_player_fraction, "metric_player_fraction")
        )
        if not isinstance(self.policy, SpatialEligibilityPolicy):
            raise P4SpatialEligibilityError("policy must be SpatialEligibilityPolicy")
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))
        if self.contract_version != P4_SPATIAL_ELIGIBILITY_CONTRACT_VERSION:
            raise P4SpatialEligibilityError(
                f"contract_version must be {P4_SPATIAL_ELIGIBILITY_CONTRACT_VERSION!r}"
            )

    @property
    def usable(self) -> bool:
        return self.state is not SpatialEligibilityState.INVALID

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "state": self.state.value,
            "reasons": [reason.value for reason in self.reasons],
            "geometry_quality_state": self.geometry_quality_state.value,
            "visible_player_count": self.visible_player_count,
            "metric_player_count": self.metric_player_count,
            "missing_metric_player_count": self.missing_metric_player_count,
            "out_of_bounds_metric_player_count": self.out_of_bounds_metric_player_count,
            "metric_player_fraction": self.metric_player_fraction,
            "policy": self.policy.as_dict(),
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialEligibilityObservation":
        _exact_fields(payload, P4_SPATIAL_ELIGIBILITY_FIELDS, "SpatialEligibilityObservation")
        normalized = dict(payload)
        policy = normalized["policy"]
        if not isinstance(policy, Mapping):
            raise P4SpatialEligibilityError("policy must be an object")
        normalized["policy"] = SpatialEligibilityPolicy.from_dict(policy)
        normalized["reasons"] = tuple(normalized["reasons"])
        return cls(**normalized)


def p4_spatial_eligibility_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_SPATIAL_ELIGIBILITY_CONTRACT_VERSION,
        "policy_version": P4_SPATIAL_ELIGIBILITY_POLICY_VERSION,
        "states": [state.value for state in SpatialEligibilityState],
        "reasons": [reason.value for reason in SpatialEligibilityReason],
        "fields": sorted(P4_SPATIAL_ELIGIBILITY_FIELDS),
        "policy_fields": sorted(P4_SPATIAL_ELIGIBILITY_POLICY_FIELDS),
        "canonical_coordinate_system": P4_FIBA_COURT_COORDINATE_SYSTEM,
        "confidence_thresholds": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_semantics": False,
    }


def evaluate_spatial_eligibility(
    frame: SpatialFrame,
    geometry_quality: GeometryQualityObservation,
    *,
    policy: SpatialEligibilityPolicy,
) -> SpatialEligibilityObservation:
    """Classify frame-level metric spatial usability without basketball semantics."""

    if not isinstance(frame, SpatialFrame):
        raise P4SpatialEligibilityError("frame must be SpatialFrame")
    if not isinstance(geometry_quality, GeometryQualityObservation):
        raise P4SpatialEligibilityError("geometry_quality must be GeometryQualityObservation")
    if not isinstance(policy, SpatialEligibilityPolicy):
        raise P4SpatialEligibilityError("policy must be SpatialEligibilityPolicy")
    if geometry_quality.frame_id != frame.frame_id or geometry_quality.timestamp_ms != frame.timestamp_ms:
        raise P4SpatialEligibilityError("geometry quality must describe the same frame and timestamp")

    visible_players = [player for player in frame.players if player.visible]
    metric_count = 0
    missing_count = 0
    out_of_bounds_count = 0
    for player in visible_players:
        if player.court_position_m is None:
            missing_count += 1
            continue
        if not within_court_extent(
            player.court_position_m,
            tolerance_m=policy.court_extent_tolerance_m,
        ):
            out_of_bounds_count += 1
            continue
        metric_count += 1

    visible_count = len(visible_players)
    fraction = (metric_count / visible_count) if visible_count else 0.0

    invalid_reasons: list[SpatialEligibilityReason] = []
    degraded_reasons: list[SpatialEligibilityReason] = []

    if geometry_quality.state is GeometryQualityState.INVALID:
        invalid_reasons.append(SpatialEligibilityReason.GEOMETRY_INVALID)
    if policy.require_court_visible and not frame.geometry.court_visible:
        invalid_reasons.append(SpatialEligibilityReason.COURT_NOT_VISIBLE)
    if frame.geometry.court_coordinate_system != P4_FIBA_COURT_COORDINATE_SYSTEM:
        invalid_reasons.append(SpatialEligibilityReason.COORDINATE_SYSTEM_MISMATCH)
    if metric_count < policy.min_metric_player_count:
        invalid_reasons.append(SpatialEligibilityReason.METRIC_PLAYER_COUNT_BELOW_MIN)
    if fraction < policy.min_metric_player_fraction:
        invalid_reasons.append(SpatialEligibilityReason.METRIC_PLAYER_FRACTION_BELOW_MIN)

    if (
        geometry_quality.state is GeometryQualityState.OPTICAL_FLOW_HOLD_AGING
        and not policy.allow_aging_geometry
    ):
        degraded_reasons.append(SpatialEligibilityReason.AGING_GEOMETRY)
    if missing_count:
        degraded_reasons.append(SpatialEligibilityReason.MISSING_METRIC_PLAYER_POSITION)
    if out_of_bounds_count:
        degraded_reasons.append(SpatialEligibilityReason.OUT_OF_BOUNDS_METRIC_POSITION)

    if invalid_reasons:
        state = SpatialEligibilityState.INVALID
        reasons = tuple(invalid_reasons + degraded_reasons)
    elif degraded_reasons:
        state = SpatialEligibilityState.DEGRADED
        reasons = tuple(degraded_reasons)
    else:
        state = SpatialEligibilityState.ELIGIBLE
        reasons = (SpatialEligibilityReason.ELIGIBLE,)

    return SpatialEligibilityObservation(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        state=state,
        reasons=reasons,
        geometry_quality_state=geometry_quality.state,
        visible_player_count=visible_count,
        metric_player_count=metric_count,
        missing_metric_player_count=missing_count,
        out_of_bounds_metric_player_count=out_of_bounds_count,
        metric_player_fraction=fraction,
        policy=policy,
        provenance={
            "evaluator": "core.p4_spatial_eligibility.evaluate_spatial_eligibility",
            "source_spatial_contract_version": frame.contract_version,
            "source_geometry_quality_contract_version": geometry_quality.contract_version,
            "p2_consumed": False,
            "p3_consumed": False,
            "tactical_semantics": False,
        },
    )
