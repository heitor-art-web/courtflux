from __future__ import annotations

import ast
from pathlib import Path

from core.p4_court_coordinates import CourtEnd, CourtPoint
from core.p4_court_zones import (
    CourtZone,
    FIBA_THREE_POINT_CORNER_CONNECTION_DEPTH_M,
    is_paint,
    is_rim_area,
    is_three_point_region,
    p4_fiba_zone_manifest,
    perimeter_subzone,
    primary_zone,
    zone_memberships,
)


def test_paint_is_symmetric_for_both_physical_ends() -> None:
    left = CourtPoint(3.0, 7.5)
    right = CourtPoint(25.0, 7.5)
    assert is_paint(left, end=CourtEnd.LEFT)
    assert is_paint(right, end=CourtEnd.RIGHT)
    assert not is_paint(left, end=CourtEnd.RIGHT)


def test_rim_area_is_symmetric_and_more_specific_than_paint() -> None:
    left = CourtPoint(1.575, 7.5)
    right = CourtPoint(26.425, 7.5)
    assert is_rim_area(left, end=CourtEnd.LEFT)
    assert is_rim_area(right, end=CourtEnd.RIGHT)
    assert primary_zone(left, end=CourtEnd.LEFT) is CourtZone.RIM_AREA
    assert CourtZone.PAINT in zone_memberships(left, end=CourtEnd.LEFT)


def test_three_point_line_itself_is_not_three_point_region() -> None:
    # Point on the 6.75 m arc at centre-line y.
    point = CourtPoint(1.575 + 6.75, 7.5)
    assert not is_three_point_region(point, end=CourtEnd.LEFT)
    assert is_three_point_region(CourtPoint(point.x_m + 0.01, point.y_m), end=CourtEnd.LEFT)


def test_corner_zone_uses_official_straight_line_geometry() -> None:
    assert 2.9 < FIBA_THREE_POINT_CORNER_CONNECTION_DEPTH_M < 3.1
    lower = CourtPoint(2.0, 0.5)
    upper = CourtPoint(2.0, 14.5)
    assert perimeter_subzone(lower, end=CourtEnd.LEFT) is CourtZone.LOWER_CORNER
    assert perimeter_subzone(upper, end=CourtEnd.LEFT) is CourtZone.UPPER_CORNER


def test_wing_and_top_subzones_are_canonical_y_regions() -> None:
    lower = CourtPoint(7.5, 3.0)
    top = CourtPoint(8.5, 7.5)
    upper = CourtPoint(7.5, 12.0)
    assert perimeter_subzone(lower, end=CourtEnd.LEFT) is CourtZone.LOWER_WING
    assert perimeter_subzone(top, end=CourtEnd.LEFT) is CourtZone.TOP
    assert perimeter_subzone(upper, end=CourtEnd.LEFT) is CourtZone.UPPER_WING


def test_right_end_uses_same_zone_logic_without_attack_semantics() -> None:
    point = CourtPoint(26.0, 0.5)
    memberships = zone_memberships(point, end=CourtEnd.RIGHT)
    assert CourtZone.PERIMETER in memberships
    assert CourtZone.LOWER_CORNER in memberships


def test_other_half_is_explicit_for_selected_physical_end() -> None:
    point = CourtPoint(20.0, 7.5)
    assert primary_zone(point, end=CourtEnd.LEFT) is CourtZone.OTHER_HALF
    assert primary_zone(point, end=CourtEnd.RIGHT) is not CourtZone.OTHER_HALF


def test_outside_points_return_outside_without_repair() -> None:
    assert primary_zone(CourtPoint(-0.1, 7.5), end=CourtEnd.LEFT) is CourtZone.OUTSIDE
    assert zone_memberships(CourtPoint(14.0, 15.1), end=CourtEnd.LEFT) == frozenset(
        {CourtZone.OUTSIDE}
    )


def test_manifest_excludes_p2_p3_and_attack_direction_inference() -> None:
    manifest = p4_fiba_zone_manifest()
    assert manifest["selected_end_required"] is True
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["attack_direction_inference"] is False
    assert manifest["tactical_semantics"] is False
    assert manifest["three_point"]["boundary_line_is_two_point"] is True


def test_zone_module_does_not_import_p2_or_p3() -> None:
    module = Path(__file__).parents[1] / "core" / "p4_court_zones.py"
    tree = ast.parse(module.read_text())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any("p2" in name or "p3" in name for name in imports)
