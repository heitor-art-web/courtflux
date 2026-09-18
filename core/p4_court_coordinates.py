from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot, isfinite
from numbers import Real
from typing import Any, Mapping, Sequence

from core.p4_spatial_contract import P4_UPSTREAM_COURT_COORDINATE_SYSTEM

P4_FIBA_COURT_CONTRACT_VERSION = "p4.fiba_court.v1"
P4_FIBA_COURT_COORDINATE_SYSTEM = "p4.fiba_28x15_xy.v1"

FIBA_COURT_LENGTH_M = 28.0
FIBA_COURT_WIDTH_M = 15.0
FIBA_HALF_COURT_LENGTH_M = FIBA_COURT_LENGTH_M / 2.0
FIBA_COURT_CENTER_Y_M = FIBA_COURT_WIDTH_M / 2.0

# Official FIBA court-marking geometry used by later zone helpers.
FIBA_BASKET_CENTER_FROM_ENDLINE_M = 1.575
FIBA_FREE_THROW_LINE_FROM_ENDLINE_M = 5.80
FIBA_RESTRICTED_AREA_HALF_WIDTH_M = 2.45
FIBA_RESTRICTED_AREA_Y_MIN_M = FIBA_COURT_CENTER_Y_M - FIBA_RESTRICTED_AREA_HALF_WIDTH_M
FIBA_RESTRICTED_AREA_Y_MAX_M = FIBA_COURT_CENTER_Y_M + FIBA_RESTRICTED_AREA_HALF_WIDTH_M
FIBA_THREE_POINT_ARC_RADIUS_M = 6.75
FIBA_THREE_POINT_LINE_FROM_SIDELINE_M = 0.90
FIBA_NO_CHARGE_SEMICIRCLE_RADIUS_M = 1.30


class P4CourtCoordinateError(ValueError):
    """Raised when a point or transform violates the canonical FIBA court contract."""


class CourtEnd(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class LongitudinalDirection(str, Enum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4CourtCoordinateError(f"{field_name} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise P4CourtCoordinateError(f"{field_name} must be a finite number")
    return number


def _enum_value(value: Any, enum_type: type[Enum], field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        raise P4CourtCoordinateError(f"{field_name} is not supported") from exc


@dataclass(frozen=True, slots=True)
class CourtPoint:
    """One metric point in the canonical 28 m x 15 m FIBA court coordinate system."""

    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _finite_number(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _finite_number(self.y_m, "y_m"))

    def as_tuple(self) -> tuple[float, float]:
        return self.x_m, self.y_m

    def as_dict(self) -> dict[str, float]:
        return {"x_m": self.x_m, "y_m": self.y_m}

    @classmethod
    def from_value(cls, value: "CourtPoint | Sequence[float] | Mapping[str, Any]") -> "CourtPoint":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if frozenset(value) != frozenset({"x_m", "y_m"}):
                raise P4CourtCoordinateError(
                    "CourtPoint mapping must contain exactly x_m and y_m"
                )
            return cls(x_m=value["x_m"], y_m=value["y_m"])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            return cls(x_m=value[0], y_m=value[1])
        raise P4CourtCoordinateError(
            "CourtPoint must be a CourtPoint, [x_m, y_m], or {x_m, y_m}"
        )


def canonicalize_upstream_point(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    *,
    source_coordinate_system: str,
) -> CourtPoint:
    """Convert a supported P1 metric point into the P4 canonical FIBA system.

    `p1.runtime_metric.v1` already uses the same 28 x 15 metre orientation, so the
    current transform is identity after strict numeric validation. The explicit source
    label prevents future upstream coordinate changes from being silently reinterpreted.
    """

    source = str(source_coordinate_system).strip()
    if source != P4_UPSTREAM_COURT_COORDINATE_SYSTEM:
        raise P4CourtCoordinateError(
            f"unsupported upstream court coordinate system: {source!r}"
        )
    return CourtPoint.from_value(point)


def within_court_extent(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    *,
    tolerance_m: float = 0.0,
) -> bool:
    """Return whether a point lies within the geometric court extent, including lines.

    This is intentionally a geometry helper, not a FIBA in/out ruling. Official boundary
    lines are not part of the playing court; P4 stores coordinates on those lines and lets
    later rules/eligibility logic decide how to treat them.
    """

    p = CourtPoint.from_value(point)
    tolerance = _finite_number(tolerance_m, "tolerance_m")
    if tolerance < 0.0:
        raise P4CourtCoordinateError("tolerance_m must be >= 0")
    return bool(
        -tolerance <= p.x_m <= FIBA_COURT_LENGTH_M + tolerance
        and -tolerance <= p.y_m <= FIBA_COURT_WIDTH_M + tolerance
    )


def require_within_court_extent(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    *,
    tolerance_m: float = 0.0,
) -> CourtPoint:
    p = CourtPoint.from_value(point)
    if not within_court_extent(p, tolerance_m=tolerance_m):
        raise P4CourtCoordinateError(
            f"point {p.as_tuple()} is outside the canonical FIBA court extent"
        )
    return p


def normalize_unit(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
) -> tuple[float, float]:
    """Map an in-bounds metric point to [0,1] x [0,1] without changing orientation."""

    p = require_within_court_extent(point)
    return p.x_m / FIBA_COURT_LENGTH_M, p.y_m / FIBA_COURT_WIDTH_M


def denormalize_unit(x_unit: Any, y_unit: Any) -> CourtPoint:
    x = _finite_number(x_unit, "x_unit")
    y = _finite_number(y_unit, "y_unit")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise P4CourtCoordinateError("normalized coordinates must be within [0, 1]")
    return CourtPoint(x * FIBA_COURT_LENGTH_M, y * FIBA_COURT_WIDTH_M)


def mirror_longitudinal(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
) -> CourtPoint:
    p = CourtPoint.from_value(point)
    return CourtPoint(FIBA_COURT_LENGTH_M - p.x_m, p.y_m)


def mirror_lateral(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
) -> CourtPoint:
    p = CourtPoint.from_value(point)
    return CourtPoint(p.x_m, FIBA_COURT_WIDTH_M - p.y_m)


def normalize_longitudinal_direction(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    direction: LongitudinalDirection | str,
) -> CourtPoint:
    """Return a point with motion direction normalized left-to-right.

    The caller must provide direction explicitly. This helper never infers attack direction,
    possession, team role, or any P2/P3 semantics.
    """

    p = CourtPoint.from_value(point)
    resolved = _enum_value(direction, LongitudinalDirection, "direction")
    if resolved is LongitudinalDirection.LEFT_TO_RIGHT:
        return p
    return mirror_longitudinal(p)


def basket_center(end: CourtEnd | str) -> CourtPoint:
    resolved = _enum_value(end, CourtEnd, "end")
    if resolved is CourtEnd.LEFT:
        x_m = FIBA_BASKET_CENTER_FROM_ENDLINE_M
    else:
        x_m = FIBA_COURT_LENGTH_M - FIBA_BASKET_CENTER_FROM_ENDLINE_M
    return CourtPoint(x_m, FIBA_COURT_CENTER_Y_M)


def distance_to_basket(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    end: CourtEnd | str,
) -> float:
    p = CourtPoint.from_value(point)
    basket = basket_center(end)
    return hypot(p.x_m - basket.x_m, p.y_m - basket.y_m)


def distance_from_endline(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    end: CourtEnd | str,
) -> float:
    """Longitudinal depth from a selected physical endline, without role semantics."""

    p = CourtPoint.from_value(point)
    resolved = _enum_value(end, CourtEnd, "end")
    return p.x_m if resolved is CourtEnd.LEFT else FIBA_COURT_LENGTH_M - p.x_m


def to_end_relative(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    end: CourtEnd | str,
) -> CourtPoint:
    """Represent a point with x=0 at the selected endline.

    y is intentionally preserved. `lower`/`upper` therefore always refer to canonical
    y-space, never to offensive left/right.
    """

    p = CourtPoint.from_value(point)
    return CourtPoint(distance_from_endline(p, end), p.y_m)


def from_end_relative(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    end: CourtEnd | str,
) -> CourtPoint:
    relative = CourtPoint.from_value(point)
    resolved = _enum_value(end, CourtEnd, "end")
    if resolved is CourtEnd.LEFT:
        return CourtPoint(relative.x_m, relative.y_m)
    return CourtPoint(FIBA_COURT_LENGTH_M - relative.x_m, relative.y_m)


def p4_fiba_court_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_FIBA_COURT_CONTRACT_VERSION,
        "coordinate_system": P4_FIBA_COURT_COORDINATE_SYSTEM,
        "units": "metres",
        "origin": "left_endline_lower_sideline_inner_edges",
        "x_axis": "left_endline_to_right_endline",
        "y_axis": "lower_sideline_to_upper_sideline",
        "court_length_m": FIBA_COURT_LENGTH_M,
        "court_width_m": FIBA_COURT_WIDTH_M,
        "court_center_m": [FIBA_HALF_COURT_LENGTH_M, FIBA_COURT_CENTER_Y_M],
        "basket_centers_m": {
            "left": list(basket_center(CourtEnd.LEFT).as_tuple()),
            "right": list(basket_center(CourtEnd.RIGHT).as_tuple()),
        },
        "markings": {
            "free_throw_line_from_endline_m": FIBA_FREE_THROW_LINE_FROM_ENDLINE_M,
            "restricted_area_half_width_m": FIBA_RESTRICTED_AREA_HALF_WIDTH_M,
            "three_point_arc_radius_m": FIBA_THREE_POINT_ARC_RADIUS_M,
            "three_point_line_from_sideline_m": FIBA_THREE_POINT_LINE_FROM_SIDELINE_M,
            "no_charge_semicircle_radius_m": FIBA_NO_CHARGE_SEMICIRCLE_RADIUS_M,
        },
        "boundary_storage_policy": "extent_includes_boundary_lines",
        "upstream_compatibility": {
            P4_UPSTREAM_COURT_COORDINATE_SYSTEM: "identity_transform_validated",
        },
        "direction_inference": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_semantics": False,
    }
