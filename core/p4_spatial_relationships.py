from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import hypot, isfinite
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from core.p4_court_coordinates import (
    CourtEnd,
    CourtPoint,
    P4CourtCoordinateError,
    distance_to_basket,
    require_within_court_extent,
)
from core.p4_kinematics import KinematicsPoint

P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION = "p4.spatial_relationships.v1"


class P4SpatialRelationshipError(ValueError):
    pass


class LocalTeamRelation(str, Enum):
    SAME = "same_local_team"
    DIFFERENT = "different_local_team"
    UNKNOWN = "unknown"


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise P4SpatialRelationshipError(f"{field_name} must be a non-negative integer")
    number = int(value)
    if number < 0:
        raise P4SpatialRelationshipError(f"{field_name} must be a non-negative integer")
    return number


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4SpatialRelationshipError(f"{field_name} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise P4SpatialRelationshipError(f"{field_name} must be a finite number")
    return number


def _point2(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise P4SpatialRelationshipError(f"{field_name} must contain exactly two coordinates")
    return _finite_number(value[0], field_name), _finite_number(value[1], field_name)


def _optional_point2(value: Any, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    return _point2(value, field_name)


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise P4SpatialRelationshipError(f"{field_name} must be non-empty when provided")
    return text


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P4SpatialRelationshipError(f"{field_name} must be a JSON object")
    normalized = dict(value)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise P4SpatialRelationshipError(f"{field_name} must be JSON serializable") from exc
    return normalized


@dataclass(frozen=True, slots=True)
class SpatialActorState:
    track_id: int
    timestamp_ms: int
    position_m: tuple[float, float]
    velocity_mps: tuple[float, float] | None = None
    local_team_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonnegative_int(self.track_id, "track_id"))
        object.__setattr__(self, "timestamp_ms", _nonnegative_int(self.timestamp_ms, "timestamp_ms"))
        position = _point2(self.position_m, "position_m")
        try:
            require_within_court_extent(position)
        except P4CourtCoordinateError as exc:
            raise P4SpatialRelationshipError(
                "position_m must be within the canonical FIBA court"
            ) from exc
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_mps", _optional_point2(self.velocity_mps, "velocity_mps"))
        object.__setattr__(self, "local_team_id", _optional_text(self.local_team_id, "local_team_id"))
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "timestamp_ms": self.timestamp_ms,
            "position_m": list(self.position_m),
            "velocity_mps": list(self.velocity_mps) if self.velocity_mps is not None else None,
            "local_team_id": self.local_team_id,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialActorState":
        expected = {
            "track_id",
            "timestamp_ms",
            "position_m",
            "velocity_mps",
            "local_team_id",
            "provenance",
        }
        if set(payload) != expected:
            raise P4SpatialRelationshipError(
                "SpatialActorState fields must match frozen v1 schema"
            )
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class PairwiseSpatialRelationship:
    timestamp_ms: int
    track_a_id: int
    track_b_id: int
    distance_m: float
    delta_position_m: tuple[float, float]
    local_team_relation: LocalTeamRelation
    relative_velocity_mps: tuple[float, float] | None
    closing_speed_mps: float | None
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ms", _nonnegative_int(self.timestamp_ms, "timestamp_ms"))
        object.__setattr__(self, "track_a_id", _nonnegative_int(self.track_a_id, "track_a_id"))
        object.__setattr__(self, "track_b_id", _nonnegative_int(self.track_b_id, "track_b_id"))
        if self.track_a_id == self.track_b_id:
            raise P4SpatialRelationshipError("relationship requires two different track ids")
        distance = _finite_number(self.distance_m, "distance_m")
        if distance < 0.0:
            raise P4SpatialRelationshipError("distance_m must be >= 0")
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "delta_position_m", _point2(self.delta_position_m, "delta_position_m"))
        if not isinstance(self.local_team_relation, LocalTeamRelation):
            try:
                object.__setattr__(
                    self,
                    "local_team_relation",
                    LocalTeamRelation(str(self.local_team_relation)),
                )
            except ValueError as exc:
                raise P4SpatialRelationshipError("local_team_relation is not supported") from exc
        object.__setattr__(
            self,
            "relative_velocity_mps",
            _optional_point2(self.relative_velocity_mps, "relative_velocity_mps"),
        )
        if self.closing_speed_mps is not None:
            object.__setattr__(
                self,
                "closing_speed_mps",
                _finite_number(self.closing_speed_mps, "closing_speed_mps"),
            )
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))
        if self.contract_version != P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION:
            raise P4SpatialRelationshipError("unsupported contract_version")

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "track_a_id": self.track_a_id,
            "track_b_id": self.track_b_id,
            "distance_m": self.distance_m,
            "delta_position_m": list(self.delta_position_m),
            "local_team_relation": self.local_team_relation.value,
            "relative_velocity_mps": (
                list(self.relative_velocity_mps)
                if self.relative_velocity_mps is not None
                else None
            ),
            "closing_speed_mps": self.closing_speed_mps,
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairwiseSpatialRelationship":
        expected = {
            "timestamp_ms",
            "track_a_id",
            "track_b_id",
            "distance_m",
            "delta_position_m",
            "local_team_relation",
            "relative_velocity_mps",
            "closing_speed_mps",
            "provenance",
            "contract_version",
        }
        if set(payload) != expected:
            raise P4SpatialRelationshipError(
                "PairwiseSpatialRelationship fields must match frozen v1 schema"
            )
        return cls(**dict(payload))


def actor_state_from_kinematics(
    point: KinematicsPoint,
    *,
    local_team_id: str | None = None,
) -> SpatialActorState:
    """Build the neutral relationship input from one P4 kinematics point."""
    if not isinstance(point, KinematicsPoint):
        raise P4SpatialRelationshipError("point must be KinematicsPoint")
    if point.position_m is None:
        raise P4SpatialRelationshipError("kinematics point must contain position_m")
    return SpatialActorState(
        track_id=point.track_id,
        timestamp_ms=point.timestamp_ms,
        position_m=point.position_m,
        velocity_mps=point.velocity_mps,
        local_team_id=local_team_id,
        provenance={
            "adapter": "core.p4_spatial_relationships.actor_state_from_kinematics",
            "source_kinematics_contract": "p4.kinematics.v1",
            "p2_consumed": False,
            "p3_consumed": False,
            "tactical_semantics": False,
        },
    )


def _team_relation(a: SpatialActorState, b: SpatialActorState) -> LocalTeamRelation:
    if a.local_team_id is None or b.local_team_id is None:
        return LocalTeamRelation.UNKNOWN
    if a.local_team_id == b.local_team_id:
        return LocalTeamRelation.SAME
    return LocalTeamRelation.DIFFERENT


def distance_between(a: SpatialActorState, b: SpatialActorState) -> float:
    if a.timestamp_ms != b.timestamp_ms:
        raise P4SpatialRelationshipError("actors must describe the same timestamp")
    return hypot(
        b.position_m[0] - a.position_m[0],
        b.position_m[1] - a.position_m[1],
    )


def distance_to_target(actor: SpatialActorState, target: Sequence[float]) -> float:
    target_point = _point2(target, "target")
    return hypot(
        target_point[0] - actor.position_m[0],
        target_point[1] - actor.position_m[1],
    )


def distance_to_physical_basket(
    actor: SpatialActorState,
    end: CourtEnd | str,
) -> float:
    return float(distance_to_basket(CourtPoint(*actor.position_m), end))


def pair_relationship(
    a: SpatialActorState,
    b: SpatialActorState,
) -> PairwiseSpatialRelationship:
    if a.timestamp_ms != b.timestamp_ms:
        raise P4SpatialRelationshipError("actors must describe the same timestamp")
    dx = b.position_m[0] - a.position_m[0]
    dy = b.position_m[1] - a.position_m[1]
    distance = hypot(dx, dy)
    relative_velocity = None
    closing_speed = None
    if a.velocity_mps is not None and b.velocity_mps is not None:
        relative_velocity = (
            b.velocity_mps[0] - a.velocity_mps[0],
            b.velocity_mps[1] - a.velocity_mps[1],
        )
        if distance > 0.0:
            radial_rate = (
                dx * relative_velocity[0] + dy * relative_velocity[1]
            ) / distance
            closing_speed = -radial_rate
    return PairwiseSpatialRelationship(
        timestamp_ms=a.timestamp_ms,
        track_a_id=a.track_id,
        track_b_id=b.track_id,
        distance_m=distance,
        delta_position_m=(dx, dy),
        local_team_relation=_team_relation(a, b),
        relative_velocity_mps=relative_velocity,
        closing_speed_mps=closing_speed,
        provenance={
            "calculator": "core.p4_spatial_relationships.pair_relationship",
            "p2_consumed": False,
            "p3_consumed": False,
            "tactical_semantics": False,
        },
    )


def pairwise_relationships(
    actors: Sequence[SpatialActorState],
) -> tuple[PairwiseSpatialRelationship, ...]:
    normalized = tuple(actors)
    ids = [actor.track_id for actor in normalized]
    if len(ids) != len(set(ids)):
        raise P4SpatialRelationshipError("actors must have unique track_id values")
    if normalized:
        timestamp_ms = normalized[0].timestamp_ms
        if any(actor.timestamp_ms != timestamp_ms for actor in normalized):
            raise P4SpatialRelationshipError("all actors must describe the same timestamp")
    ordered = sorted(normalized, key=lambda actor: actor.track_id)
    return tuple(
        pair_relationship(ordered[i], ordered[j])
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
    )


def pairwise_distance_matrix(
    actors: Sequence[SpatialActorState],
) -> dict[int, dict[int, float]]:
    normalized = tuple(actors)
    relationships = pairwise_relationships(normalized)
    ids = sorted(actor.track_id for actor in normalized)
    matrix = {
        track_id: {
            other: 0.0 if track_id == other else float("nan")
            for other in ids
        }
        for track_id in ids
    }
    for relationship in relationships:
        matrix[relationship.track_a_id][relationship.track_b_id] = relationship.distance_m
        matrix[relationship.track_b_id][relationship.track_a_id] = relationship.distance_m
    return matrix


def nearest_actor(
    actor: SpatialActorState,
    candidates: Sequence[SpatialActorState],
    *,
    relation: LocalTeamRelation | None = None,
) -> PairwiseSpatialRelationship | None:
    relationships: list[PairwiseSpatialRelationship] = []
    for candidate in candidates:
        if candidate.track_id == actor.track_id:
            continue
        relationship = pair_relationship(actor, candidate)
        if relation is not None and relationship.local_team_relation is not relation:
            continue
        relationships.append(relationship)
    if not relationships:
        return None
    return min(
        relationships,
        key=lambda relationship: (relationship.distance_m, relationship.track_b_id),
    )


def p4_spatial_relationship_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION,
        "distance_units": "metres",
        "velocity_units": "metres_per_second",
        "closing_speed_sign": "positive_means_approaching_negative_means_separating",
        "team_scope": "optional_p1_local_team_only",
        "position_source": "canonical_fiba_metres_from_p4_kinematics_or_equivalent",
        "offense_defense_roles": False,
        "matchup_semantics": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_semantics": False,
    }
