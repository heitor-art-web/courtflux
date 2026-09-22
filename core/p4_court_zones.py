from __future__ import annotations

from enum import Enum
from math import sqrt
from typing import Any, Mapping, Sequence

from core.p4_court_coordinates import (
    CourtEnd,
    CourtPoint,
    FIBA_BASKET_CENTER_FROM_ENDLINE_M,
    FIBA_COURT_CENTER_Y_M,
    FIBA_COURT_LENGTH_M,
    FIBA_COURT_WIDTH_M,
    FIBA_FREE_THROW_LINE_FROM_ENDLINE_M,
    FIBA_NO_CHARGE_SEMICIRCLE_RADIUS_M,
    FIBA_RESTRICTED_AREA_Y_MAX_M,
    FIBA_RESTRICTED_AREA_Y_MIN_M,
    FIBA_THREE_POINT_ARC_RADIUS_M,
    FIBA_THREE_POINT_LINE_FROM_SIDELINE_M,
    P4CourtCoordinateError,
    distance_from_endline,
    distance_to_basket,
    require_within_court_extent,
)

P4_FIBA_ZONE_CONTRACT_VERSION = "p4.fiba_zones.v1"
P4_ZONE_GEOMETRY_EPSILON_M = 1e-9
FIBA_THREE_POINT_CORNER_CONNECTION_DEPTH_M = (
    FIBA_BASKET_CENTER_FROM_ENDLINE_M
    + sqrt(
        FIBA_THREE_POINT_ARC_RADIUS_M**2
        - (FIBA_COURT_CENTER_Y_M - FIBA_THREE_POINT_LINE_FROM_SIDELINE_M) ** 2
    )
)


class CourtZone(str, Enum):
    OUTSIDE = "outside"
    OTHER_HALF = "other_half"
    INTERIOR = "interior"
    PAINT = "paint"
    RIM_AREA = "rim_area"
    PERIMETER = "perimeter"
    LOWER_CORNER = "lower_corner"
    UPPER_CORNER = "upper_corner"
    LOWER_WING = "lower_wing"
    UPPER_WING = "upper_wing"
    TOP = "top"


def _end(value: CourtEnd | str) -> CourtEnd:
    if isinstance(value, CourtEnd):
        return value
    try:
        return CourtEnd(str(value))
    except ValueError as exc:
        raise P4CourtCoordinateError("end is not supported") from exc


def is_paint(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    *,
    end: CourtEnd | str,
) -> bool:
    p = require_within_court_extent(point)
    depth = distance_from_endline(p, _end(end))
    return bool(
        depth <= FIBA_FREE_THROW_LINE_FROM_ENDLINE_M + P4_ZONE_GEOMETRY_EPSILON_M
        and FIBA_RESTRICTED_AREA_Y_MIN_M - P4_ZONE_GEOMETRY_EPSILON_M
        <= p.y_m
        <= FIBA_RESTRICTED_AREA_Y_MAX_M + P4_ZONE_GEOMETRY_EPSILON_M
    )


def is_rim_area(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    *,
    end: CourtEnd | str,
) -> bool:
    """CourtFlux v1 rim-area grouping using the official 1.30 m no-charge radius.

    This is a spatial grouping, not rules adjudication. The circle is clipped by the court
    extent and used symmetrically for either selected physical basket.
    """

    p = require_within_court_extent(point)
    return bool(
        distance_to_basket(p, _end(end))
        <= FIBA_NO_CHARGE_SEMICIRCLE_RADIUS_M + P4_ZONE_GEOMETRY_EPSILON_M
    )


def is_three_point_region(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    *,
    end: CourtEnd | str,
) -> bool:
    """Whether a point is beyond the FIBA 3-point boundary for a selected basket.

    Boundary-line points remain in the two-point region. A tiny metric epsilon prevents
    floating-point roundoff from turning an exact arc point into a three-point point.
    """

    p = require_within_court_extent(point)
    selected_end = _end(end)
    depth = distance_from_endline(p, selected_end)
    if depth > FIBA_COURT_LENGTH_M / 2.0 + P4_ZONE_GEOMETRY_EPSILON_M:
        return False
    if p.y_m < FIBA_THREE_POINT_LINE_FROM_SIDELINE_M - P4_ZONE_GEOMETRY_EPSILON_M:
        return True
    if (
        p.y_m
        > FIBA_COURT_WIDTH_M
        - FIBA_THREE_POINT_LINE_FROM_SIDELINE_M
        + P4_ZONE_GEOMETRY_EPSILON_M
    ):
        return True
    return bool(
        distance_to_basket(p, selected_end)
        > FIBA_THREE_POINT_ARC_RADIUS_M + P4_ZONE_GEOMETRY_EPSILON_M
    )


def perimeter_subzone(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    *,
    end: CourtEnd | str,
) -> CourtZone | None:
    p = require_within_court_extent(point)
    selected_end = _end(end)
    if not is_three_point_region(p, end=selected_end):
        return None

    depth = distance_from_endline(p, selected_end)
    lower_line = FIBA_THREE_POINT_LINE_FROM_SIDELINE_M
    upper_line = FIBA_COURT_WIDTH_M - FIBA_THREE_POINT_LINE_FROM_SIDELINE_M

    if depth <= FIBA_THREE_POINT_CORNER_CONNECTION_DEPTH_M + P4_ZONE_GEOMETRY_EPSILON_M:
        if p.y_m < lower_line - P4_ZONE_GEOMETRY_EPSILON_M:
            return CourtZone.LOWER_CORNER
        if p.y_m > upper_line + P4_ZONE_GEOMETRY_EPSILON_M:
            return CourtZone.UPPER_CORNER

    if p.y_m < FIBA_RESTRICTED_AREA_Y_MIN_M:
        return CourtZone.LOWER_WING
    if p.y_m > FIBA_RESTRICTED_AREA_Y_MAX_M:
        return CourtZone.UPPER_WING
    return CourtZone.TOP


def zone_memberships(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    *,
    end: CourtEnd | str,
) -> frozenset[CourtZone]:
    """Return overlapping geometry memberships for one selected physical basket."""

    try:
        p = require_within_court_extent(point)
    except P4CourtCoordinateError:
        return frozenset({CourtZone.OUTSIDE})

    selected_end = _end(end)
    depth = distance_from_endline(p, selected_end)
    if depth > FIBA_COURT_LENGTH_M / 2.0 + P4_ZONE_GEOMETRY_EPSILON_M:
        return frozenset({CourtZone.OTHER_HALF})

    memberships: set[CourtZone] = set()
    if is_paint(p, end=selected_end):
        memberships.add(CourtZone.PAINT)
    if is_rim_area(p, end=selected_end):
        memberships.add(CourtZone.RIM_AREA)
    subzone = perimeter_subzone(p, end=selected_end)
    if subzone is not None:
        memberships.add(CourtZone.PERIMETER)
        memberships.add(subzone)
    if not memberships:
        memberships.add(CourtZone.INTERIOR)
    return frozenset(memberships)


def primary_zone(
    point: CourtPoint | Sequence[float] | Mapping[str, Any],
    *,
    end: CourtEnd | str,
) -> CourtZone:
    memberships = zone_memberships(point, end=end)
    for zone in (
        CourtZone.OUTSIDE,
        CourtZone.OTHER_HALF,
        CourtZone.RIM_AREA,
        CourtZone.PAINT,
        CourtZone.LOWER_CORNER,
        CourtZone.UPPER_CORNER,
        CourtZone.LOWER_WING,
        CourtZone.UPPER_WING,
        CourtZone.TOP,
        CourtZone.INTERIOR,
    ):
        if zone in memberships:
            return zone
    return CourtZone.INTERIOR


def p4_fiba_zone_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_FIBA_ZONE_CONTRACT_VERSION,
        "zones": [zone.value for zone in CourtZone],
        "selected_end_required": True,
        "selected_end_semantics": "physical_basket_only_not_attack_direction",
        "geometry_epsilon_m": P4_ZONE_GEOMETRY_EPSILON_M,
        "paint": {
            "depth_m": FIBA_FREE_THROW_LINE_FROM_ENDLINE_M,
            "y_min_m": FIBA_RESTRICTED_AREA_Y_MIN_M,
            "y_max_m": FIBA_RESTRICTED_AREA_Y_MAX_M,
        },
        "rim_area": {
            "radius_m": FIBA_NO_CHARGE_SEMICIRCLE_RADIUS_M,
            "semantics": "courtflux_spatial_grouping_not_rules_adjudication",
        },
        "three_point": {
            "arc_radius_m": FIBA_THREE_POINT_ARC_RADIUS_M,
            "straight_line_from_sideline_m": FIBA_THREE_POINT_LINE_FROM_SIDELINE_M,
            "corner_connection_depth_m": FIBA_THREE_POINT_CORNER_CONNECTION_DEPTH_M,
            "boundary_line_is_two_point": True,
        },
        "p2_dependency": False,
        "p3_dependency": False,
        "attack_direction_inference": False,
        "tactical_semantics": False,
    }
