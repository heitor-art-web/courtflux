from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import isfinite
from numbers import Real
from typing import Any, Mapping, Sequence

from core.p4_court_coordinates import CourtEnd
from core.p4_court_zones import CourtZone, zone_memberships
from core.p4_spatial_relationships import (
    P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION,
    SpatialActorState,
    pairwise_relationships,
)

P4_TEAM_SHAPE_CONTRACT_VERSION = "p4.team_shape.v1"


class P4TeamShapeError(ValueError):
    pass


def _finite_nonnegative(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4TeamShapeError(f"{field_name} must be a non-negative finite number")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise P4TeamShapeError(f"{field_name} must be a non-negative finite number")
    return number


def _point2(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise P4TeamShapeError(f"{field_name} must contain exactly two numbers")
    values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise P4TeamShapeError(f"{field_name} must contain finite numbers")
        number = float(item)
        if not isfinite(number):
            raise P4TeamShapeError(f"{field_name} must contain finite numbers")
        values.append(number)
    return values[0], values[1]


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P4TeamShapeError(f"{field_name} must be a JSON object")
    normalized = dict(value)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise P4TeamShapeError(f"{field_name} must be JSON serializable") from exc
    return normalized


@dataclass(frozen=True, slots=True)
class PhysicalEndOccupancy:
    end: CourtEnd
    player_count: int
    paint_count: int
    rim_area_count: int
    perimeter_count: int
    other_half_count: int
    paint_fraction: float
    perimeter_fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.end, CourtEnd):
            try:
                object.__setattr__(self, "end", CourtEnd(str(self.end)))
            except ValueError as exc:
                raise P4TeamShapeError("end is not supported") from exc
        for field_name in (
            "player_count",
            "paint_count",
            "rim_area_count",
            "perimeter_count",
            "other_half_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise P4TeamShapeError(
                    f"{field_name} must be a non-negative integer"
                )
        for field_name in ("paint_fraction", "perimeter_fraction"):
            value = _finite_nonnegative(getattr(self, field_name), field_name)
            if value > 1.0:
                raise P4TeamShapeError(f"{field_name} must be within [0, 1]")
            object.__setattr__(self, field_name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "end": self.end.value,
            "player_count": self.player_count,
            "paint_count": self.paint_count,
            "rim_area_count": self.rim_area_count,
            "perimeter_count": self.perimeter_count,
            "other_half_count": self.other_half_count,
            "paint_fraction": self.paint_fraction,
            "perimeter_fraction": self.perimeter_fraction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalEndOccupancy":
        expected = {
            "end",
            "player_count",
            "paint_count",
            "rim_area_count",
            "perimeter_count",
            "other_half_count",
            "paint_fraction",
            "perimeter_fraction",
        }
        if set(payload) != expected:
            raise P4TeamShapeError(
                "PhysicalEndOccupancy fields must match frozen v1 schema"
            )
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class TeamShapeMetrics:
    timestamp_ms: int
    local_team_id: str
    track_ids: tuple[int, ...]
    player_count: int
    centroid_m: tuple[float, float]
    width_m: float
    depth_m: float
    bounding_box_m: tuple[float, float, float, float]
    convex_hull_m: tuple[tuple[float, float], ...]
    convex_hull_area_m2: float
    mean_pairwise_spacing_m: float | None
    pairwise_spacing_variance_m2: float | None
    min_pairwise_spacing_m: float | None
    max_pairwise_spacing_m: float | None
    occupancy_by_physical_end: tuple[PhysicalEndOccupancy, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_TEAM_SHAPE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ms, bool)
            or not isinstance(self.timestamp_ms, int)
            or self.timestamp_ms < 0
        ):
            raise P4TeamShapeError("timestamp_ms must be a non-negative integer")
        team = str(self.local_team_id).strip()
        if not team:
            raise P4TeamShapeError("local_team_id must be non-empty")
        object.__setattr__(self, "local_team_id", team)
        track_ids = tuple(self.track_ids)
        if (
            not track_ids
            or len(track_ids) != len(set(track_ids))
            or any(
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id < 0
                for track_id in track_ids
            )
        ):
            raise P4TeamShapeError(
                "track_ids must be unique non-negative integers"
            )
        object.__setattr__(self, "track_ids", track_ids)
        if self.player_count != len(track_ids):
            raise P4TeamShapeError("player_count must equal len(track_ids)")
        object.__setattr__(self, "centroid_m", _point2(self.centroid_m, "centroid_m"))
        object.__setattr__(self, "width_m", _finite_nonnegative(self.width_m, "width_m"))
        object.__setattr__(self, "depth_m", _finite_nonnegative(self.depth_m, "depth_m"))
        bounding_box = tuple(self.bounding_box_m)
        if len(bounding_box) != 4:
            raise P4TeamShapeError("bounding_box_m must have four values")
        object.__setattr__(
            self,
            "bounding_box_m",
            tuple(float(value) for value in bounding_box),
        )
        hull = tuple(_point2(point, "convex_hull_m") for point in self.convex_hull_m)
        object.__setattr__(self, "convex_hull_m", hull)
        object.__setattr__(
            self,
            "convex_hull_area_m2",
            _finite_nonnegative(self.convex_hull_area_m2, "convex_hull_area_m2"),
        )
        for field_name in (
            "mean_pairwise_spacing_m",
            "pairwise_spacing_variance_m2",
            "min_pairwise_spacing_m",
            "max_pairwise_spacing_m",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _finite_nonnegative(value, field_name),
                )
        occupancies = tuple(self.occupancy_by_physical_end)
        if {occupancy.end for occupancy in occupancies} != {
            CourtEnd.LEFT,
            CourtEnd.RIGHT,
        }:
            raise P4TeamShapeError(
                "occupancy_by_physical_end must contain left and right"
            )
        object.__setattr__(self, "occupancy_by_physical_end", occupancies)
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))
        if self.contract_version != P4_TEAM_SHAPE_CONTRACT_VERSION:
            raise P4TeamShapeError("unsupported contract_version")

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "local_team_id": self.local_team_id,
            "track_ids": list(self.track_ids),
            "player_count": self.player_count,
            "centroid_m": list(self.centroid_m),
            "width_m": self.width_m,
            "depth_m": self.depth_m,
            "bounding_box_m": list(self.bounding_box_m),
            "convex_hull_m": [list(point) for point in self.convex_hull_m],
            "convex_hull_area_m2": self.convex_hull_area_m2,
            "mean_pairwise_spacing_m": self.mean_pairwise_spacing_m,
            "pairwise_spacing_variance_m2": self.pairwise_spacing_variance_m2,
            "min_pairwise_spacing_m": self.min_pairwise_spacing_m,
            "max_pairwise_spacing_m": self.max_pairwise_spacing_m,
            "occupancy_by_physical_end": [
                occupancy.as_dict()
                for occupancy in self.occupancy_by_physical_end
            ],
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeamShapeMetrics":
        expected = {
            "timestamp_ms",
            "local_team_id",
            "track_ids",
            "player_count",
            "centroid_m",
            "width_m",
            "depth_m",
            "bounding_box_m",
            "convex_hull_m",
            "convex_hull_area_m2",
            "mean_pairwise_spacing_m",
            "pairwise_spacing_variance_m2",
            "min_pairwise_spacing_m",
            "max_pairwise_spacing_m",
            "occupancy_by_physical_end",
            "provenance",
            "contract_version",
        }
        if set(payload) != expected:
            raise P4TeamShapeError(
                "TeamShapeMetrics fields must match frozen v1 schema"
            )
        normalized = dict(payload)
        normalized["track_ids"] = tuple(normalized["track_ids"])
        normalized["convex_hull_m"] = tuple(
            tuple(point) for point in normalized["convex_hull_m"]
        )
        normalized["occupancy_by_physical_end"] = tuple(
            PhysicalEndOccupancy.from_dict(item)
            for item in normalized["occupancy_by_physical_end"]
        )
        return cls(**normalized)


def _cross(
    origin: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return (
        (a[0] - origin[0]) * (b[1] - origin[1])
        - (a[1] - origin[1]) * (b[0] - origin[0])
    )


def convex_hull(
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return tuple(unique)
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    normalized = tuple(points)
    if len(normalized) < 3:
        return 0.0
    return abs(
        sum(
            normalized[index][0] * normalized[(index + 1) % len(normalized)][1]
            - normalized[(index + 1) % len(normalized)][0]
            * normalized[index][1]
            for index in range(len(normalized))
        )
    ) / 2.0


def _occupancy(
    actors: Sequence[SpatialActorState],
    end: CourtEnd,
) -> PhysicalEndOccupancy:
    paint = 0
    rim = 0
    perimeter = 0
    other_half = 0
    for actor in actors:
        zones = zone_memberships(actor.position_m, end=end)
        paint += CourtZone.PAINT in zones
        rim += CourtZone.RIM_AREA in zones
        perimeter += CourtZone.PERIMETER in zones
        other_half += CourtZone.OTHER_HALF in zones
    count = len(actors)
    return PhysicalEndOccupancy(
        end=end,
        player_count=count,
        paint_count=paint,
        rim_area_count=rim,
        perimeter_count=perimeter,
        other_half_count=other_half,
        paint_fraction=paint / count,
        perimeter_fraction=perimeter / count,
    )


def compute_team_shape(
    actors: Sequence[SpatialActorState],
    *,
    local_team_id: str,
) -> TeamShapeMetrics:
    team = str(local_team_id).strip()
    if not team:
        raise P4TeamShapeError("local_team_id must be non-empty")
    selected = tuple(actor for actor in actors if actor.local_team_id == team)
    if not selected:
        raise P4TeamShapeError("no actors available for local_team_id")
    track_ids = [actor.track_id for actor in selected]
    if len(track_ids) != len(set(track_ids)):
        raise P4TeamShapeError("team actors must have unique track_id values")
    timestamp_ms = selected[0].timestamp_ms
    if any(actor.timestamp_ms != timestamp_ms for actor in selected):
        raise P4TeamShapeError("team actors must describe the same timestamp")

    xs = [actor.position_m[0] for actor in selected]
    ys = [actor.position_m[1] for actor in selected]
    centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
    width = max(ys) - min(ys)
    depth = max(xs) - min(xs)
    bounding_box = (min(xs), min(ys), max(xs), max(ys))
    hull = convex_hull(tuple(actor.position_m for actor in selected))
    area = polygon_area(hull)

    relationships = pairwise_relationships(selected)
    distances = [relationship.distance_m for relationship in relationships]
    if distances:
        mean = sum(distances) / len(distances)
        variance = sum((distance - mean) ** 2 for distance in distances) / len(distances)
        min_distance = min(distances)
        max_distance = max(distances)
    else:
        mean = None
        variance = None
        min_distance = None
        max_distance = None

    return TeamShapeMetrics(
        timestamp_ms=timestamp_ms,
        local_team_id=team,
        track_ids=tuple(sorted(track_ids)),
        player_count=len(selected),
        centroid_m=centroid,
        width_m=width,
        depth_m=depth,
        bounding_box_m=bounding_box,
        convex_hull_m=hull,
        convex_hull_area_m2=area,
        mean_pairwise_spacing_m=mean,
        pairwise_spacing_variance_m2=variance,
        min_pairwise_spacing_m=min_distance,
        max_pairwise_spacing_m=max_distance,
        occupancy_by_physical_end=(
            _occupancy(selected, CourtEnd.LEFT),
            _occupancy(selected, CourtEnd.RIGHT),
        ),
        provenance={
            "calculator": "core.p4_team_shape.compute_team_shape",
            "source_relationship_contract_version": (
                P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION
            ),
            "team_scope": "p1_local_team_only",
            "attack_direction_inferred": False,
            "p2_consumed": False,
            "p3_consumed": False,
            "tactical_semantics": False,
        },
    )


def compute_all_team_shapes(
    actors: Sequence[SpatialActorState],
) -> tuple[TeamShapeMetrics, ...]:
    teams = sorted(
        {
            actor.local_team_id
            for actor in actors
            if actor.local_team_id is not None
        }
    )
    return tuple(
        compute_team_shape(actors, local_team_id=team)
        for team in teams
    )


def p4_team_shape_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_TEAM_SHAPE_CONTRACT_VERSION,
        "team_scope": "p1_local_team_only",
        "width_axis": "canonical_y",
        "depth_axis": "canonical_x",
        "attack_relative_shape": False,
        "occupancy_semantics": (
            "reported_for_both_physical_basket_ends_without_attack_inference"
        ),
        "source_relationship_contract_version": (
            P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION
        ),
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_semantics": False,
    }
