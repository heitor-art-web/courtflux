from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import isfinite
from numbers import Integral
from typing import Any, Mapping

P4_SPATIAL_CONTRACT_VERSION = "p4.spatial_frame.v1"
P4_UPSTREAM_COURT_COORDINATE_SYSTEM = "p1.runtime_metric.v1"

P4_SPATIAL_FRAME_FIELDS = frozenset(
    {
        "frame_id",
        "timestamp_ms",
        "geometry",
        "players",
        "source_runtime_version",
        "provenance",
        "contract_version",
    }
)
P4_SPATIAL_GEOMETRY_FIELDS = frozenset(
    {
        "geometry_valid",
        "court_visible",
        "homography_source",
        "reprojection_rmse_px",
        "homography_matrix",
        "court_polygon_px",
        "court_coordinate_system",
        "projection_uncertainty_m",
        "homography_age_frames",
        "provenance",
    }
)
P4_SPATIAL_PLAYER_FIELDS = frozenset(
    {
        "track_id",
        "local_team_id",
        "local_team_confidence",
        "image_center_px",
        "court_position_m",
        "observation_confidence",
        "observation_uncertainty",
        "position_confidence",
        "visible",
        "source",
        "provenance",
    }
)

P4_SPATIAL_FORBIDDEN_SEMANTICS = frozenset(
    {
        "possession_id",
        "game_player_id",
        "ball_handler_game_player_id",
        "ball_control_state",
        "offense_team_id",
        "defense_team_id",
        "p2_events",
        "p3_derivation_id",
        "review_item_id",
        "clip_id",
        "tactical_action",
    }
)


class P4SpatialContractError(ValueError):
    """Raised when a P4 spatial-contract value violates the frozen v1 schema."""


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise P4SpatialContractError(f"{field_name} must be a boolean")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise P4SpatialContractError(f"{field_name} must be a non-negative integer")
    number = int(value)
    if number < 0:
        raise P4SpatialContractError(f"{field_name} must be a non-negative integer")
    return number


def _finite_number(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise P4SpatialContractError(f"{field_name} must be a finite number") from exc
    if not isfinite(number):
        raise P4SpatialContractError(f"{field_name} must be a finite number")
    return number


def _probability(value: Any, field_name: str) -> float:
    number = _finite_number(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise P4SpatialContractError(f"{field_name} must be within [0, 1]")
    return number


def _optional_nonnegative_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, field_name)
    if number < 0.0:
        raise P4SpatialContractError(f"{field_name} must be >= 0")
    return number


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field_name)


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise P4SpatialContractError(f"{field_name} must be non-empty when provided")
    return text


def _point2(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise P4SpatialContractError(f"{field_name} must contain exactly two coordinates")
    return (_finite_number(value[0], field_name), _finite_number(value[1], field_name))


def _optional_point2(value: Any, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    return _point2(value, field_name)


def _matrix3(value: Any, field_name: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise P4SpatialContractError(f"{field_name} must be a 3x3 matrix")
    rows: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise P4SpatialContractError(f"{field_name} must be a 3x3 matrix")
        rows.append(
            (
                _finite_number(row[0], field_name),
                _finite_number(row[1], field_name),
                _finite_number(row[2], field_name),
            )
        )
    return tuple(rows)


def _optional_matrix3(
    value: Any, field_name: str
) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    return _matrix3(value, field_name)


def _polygon(value: Any, field_name: str) -> tuple[tuple[float, float], ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise P4SpatialContractError(f"{field_name} must contain at least three points")
    return tuple(_point2(point, field_name) for point in value)


def _optional_polygon(value: Any, field_name: str) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    return _polygon(value, field_name)


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P4SpatialContractError(f"{field_name} must be a JSON object")
    normalized = dict(value)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise P4SpatialContractError(f"{field_name} must be JSON serializable") from exc
    return normalized


def _require_exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], object_name: str
) -> None:
    actual = frozenset(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"extra={extra}")
        raise P4SpatialContractError(
            f"{object_name} fields must match the frozen v1 schema ({', '.join(detail)})"
        )


@dataclass(frozen=True, slots=True)
class SpatialGeometryEvidence:
    """P1-owned geometry evidence preserved without tactical interpretation."""

    geometry_valid: bool
    court_visible: bool
    homography_source: str | None = None
    reprojection_rmse_px: float | None = None
    homography_matrix: tuple[tuple[float, float, float], ...] | None = None
    court_polygon_px: tuple[tuple[float, float], ...] | None = None
    court_coordinate_system: str | None = None
    projection_uncertainty_m: float | None = None
    homography_age_frames: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_valid", _boolean(self.geometry_valid, "geometry_valid"))
        object.__setattr__(self, "court_visible", _boolean(self.court_visible, "court_visible"))
        object.__setattr__(
            self, "homography_source", _optional_text(self.homography_source, "homography_source")
        )
        object.__setattr__(
            self,
            "reprojection_rmse_px",
            _optional_nonnegative_number(self.reprojection_rmse_px, "reprojection_rmse_px"),
        )
        object.__setattr__(
            self,
            "homography_matrix",
            _optional_matrix3(self.homography_matrix, "homography_matrix"),
        )
        object.__setattr__(
            self,
            "court_polygon_px",
            _optional_polygon(self.court_polygon_px, "court_polygon_px"),
        )
        object.__setattr__(
            self,
            "court_coordinate_system",
            _optional_text(self.court_coordinate_system, "court_coordinate_system"),
        )
        object.__setattr__(
            self,
            "projection_uncertainty_m",
            _optional_nonnegative_number(
                self.projection_uncertainty_m, "projection_uncertainty_m"
            ),
        )
        object.__setattr__(
            self,
            "homography_age_frames",
            _optional_nonnegative_int(self.homography_age_frames, "homography_age_frames"),
        )
        object.__setattr__(self, "provenance", _json_object(self.provenance, "geometry.provenance"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometry_valid": self.geometry_valid,
            "court_visible": self.court_visible,
            "homography_source": self.homography_source,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "homography_matrix": (
                [list(row) for row in self.homography_matrix]
                if self.homography_matrix is not None
                else None
            ),
            "court_polygon_px": (
                [list(point) for point in self.court_polygon_px]
                if self.court_polygon_px is not None
                else None
            ),
            "court_coordinate_system": self.court_coordinate_system,
            "projection_uncertainty_m": self.projection_uncertainty_m,
            "homography_age_frames": self.homography_age_frames,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialGeometryEvidence":
        _require_exact_fields(payload, P4_SPATIAL_GEOMETRY_FIELDS, "SpatialGeometryEvidence")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class SpatialPlayerObservation:
    """One P1 player observation normalized for spatial processing only."""

    track_id: int
    local_team_id: str | None
    local_team_confidence: float | None
    image_center_px: tuple[float, float]
    court_position_m: tuple[float, float] | None
    observation_confidence: float
    observation_uncertainty: float
    position_confidence: float | None
    visible: bool
    source: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonnegative_int(self.track_id, "track_id"))
        object.__setattr__(
            self,
            "local_team_id",
            _optional_text(self.local_team_id, "local_team_id"),
        )
        object.__setattr__(
            self,
            "local_team_confidence",
            None
            if self.local_team_confidence is None
            else _probability(self.local_team_confidence, "local_team_confidence"),
        )
        object.__setattr__(
            self,
            "image_center_px",
            _point2(self.image_center_px, "image_center_px"),
        )
        object.__setattr__(
            self, "court_position_m", _optional_point2(self.court_position_m, "court_position_m")
        )
        object.__setattr__(
            self,
            "observation_confidence",
            _probability(self.observation_confidence, "observation_confidence"),
        )
        object.__setattr__(
            self,
            "observation_uncertainty",
            _probability(self.observation_uncertainty, "observation_uncertainty"),
        )
        object.__setattr__(
            self,
            "position_confidence",
            None
            if self.position_confidence is None
            else _probability(self.position_confidence, "position_confidence"),
        )
        object.__setattr__(self, "visible", _boolean(self.visible, "visible"))
        source = str(self.source).strip()
        if not source:
            raise P4SpatialContractError("source must be non-empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "provenance", _json_object(self.provenance, "player.provenance"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "local_team_id": self.local_team_id,
            "local_team_confidence": self.local_team_confidence,
            "image_center_px": list(self.image_center_px),
            "court_position_m": (
                list(self.court_position_m) if self.court_position_m is not None else None
            ),
            "observation_confidence": self.observation_confidence,
            "observation_uncertainty": self.observation_uncertainty,
            "position_confidence": self.position_confidence,
            "visible": self.visible,
            "source": self.source,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialPlayerObservation":
        _require_exact_fields(payload, P4_SPATIAL_PLAYER_FIELDS, "SpatialPlayerObservation")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class SpatialFrame:
    """Frozen P1-to-P4 spatial envelope with no P2/P3/tactical truth."""

    frame_id: int
    timestamp_ms: int
    geometry: SpatialGeometryEvidence
    players: tuple[SpatialPlayerObservation, ...] = ()
    source_runtime_version: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_SPATIAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _nonnegative_int(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "timestamp_ms",
            _nonnegative_int(self.timestamp_ms, "timestamp_ms"),
        )
        if not isinstance(self.geometry, SpatialGeometryEvidence):
            raise P4SpatialContractError("geometry must be SpatialGeometryEvidence")
        players = tuple(self.players)
        if not all(isinstance(player, SpatialPlayerObservation) for player in players):
            raise P4SpatialContractError("players must contain SpatialPlayerObservation values")
        track_ids = [player.track_id for player in players]
        if len(track_ids) != len(set(track_ids)):
            raise P4SpatialContractError("players must have unique track_id values per frame")
        object.__setattr__(self, "players", players)
        object.__setattr__(
            self,
            "source_runtime_version",
            _optional_text(self.source_runtime_version, "source_runtime_version"),
        )
        object.__setattr__(self, "provenance", _json_object(self.provenance, "frame.provenance"))
        if self.contract_version != P4_SPATIAL_CONTRACT_VERSION:
            raise P4SpatialContractError(
                f"contract_version must be {P4_SPATIAL_CONTRACT_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "geometry": self.geometry.as_dict(),
            "players": [player.as_dict() for player in self.players],
            "source_runtime_version": self.source_runtime_version,
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialFrame":
        _require_exact_fields(payload, P4_SPATIAL_FRAME_FIELDS, "SpatialFrame")
        normalized = dict(payload)
        geometry = normalized["geometry"]
        if not isinstance(geometry, Mapping):
            raise P4SpatialContractError("geometry must be a JSON object")
        players = normalized["players"]
        if not isinstance(players, (list, tuple)):
            raise P4SpatialContractError("players must be an array")
        normalized["geometry"] = SpatialGeometryEvidence.from_dict(geometry)
        normalized["players"] = tuple(
            SpatialPlayerObservation.from_dict(player)
            if isinstance(player, Mapping)
            else _raise_player_object()
            for player in players
        )
        return cls(**normalized)


def _raise_player_object() -> SpatialPlayerObservation:
    raise P4SpatialContractError("each player must be a JSON object")


def p4_spatial_contract_manifest() -> dict[str, Any]:
    """Machine-readable description of the frozen v1 contract boundary."""

    return {
        "contract_version": P4_SPATIAL_CONTRACT_VERSION,
        "time_unit": "source_video_milliseconds",
        "coordinate_units": {
            "image_center_px": "pixels",
            "court_position_m": "metres",
        },
        "upstream_court_coordinate_system": P4_UPSTREAM_COURT_COORDINATE_SYSTEM,
        "frame_fields": sorted(P4_SPATIAL_FRAME_FIELDS),
        "geometry_fields": sorted(P4_SPATIAL_GEOMETRY_FIELDS),
        "player_fields": sorted(P4_SPATIAL_PLAYER_FIELDS),
        "forbidden_semantics": sorted(P4_SPATIAL_FORBIDDEN_SEMANTICS),
        "unknown_representation": "null",
        "identity_scope": "track_id_only",
        "team_scope": "optional_local_p1_team_evidence_only",
        "basketball_semantics": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_labels": False,
        "threshold_policy": "none_in_contract_v1",
    }
