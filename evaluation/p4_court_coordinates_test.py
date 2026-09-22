from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.p4_court_coordinates import (
    CourtEnd,
    CourtPoint,
    FIBA_BASKET_CENTER_FROM_ENDLINE_M,
    FIBA_COURT_LENGTH_M,
    FIBA_COURT_WIDTH_M,
    LongitudinalDirection,
    P4CourtCoordinateError,
    basket_center,
    canonicalize_upstream_point,
    denormalize_unit,
    distance_from_endline,
    mirror_lateral,
    mirror_longitudinal,
    normalize_longitudinal_direction,
    normalize_unit,
    p4_fiba_court_manifest,
    to_end_relative,
    from_end_relative,
    within_court_extent,
)


def test_dimensions_and_basket_centres_are_frozen() -> None:
    assert FIBA_COURT_LENGTH_M == 28.0
    assert FIBA_COURT_WIDTH_M == 15.0
    assert FIBA_BASKET_CENTER_FROM_ENDLINE_M == 1.575
    assert basket_center(CourtEnd.LEFT) == CourtPoint(1.575, 7.5)
    assert basket_center(CourtEnd.RIGHT) == CourtPoint(26.425, 7.5)


def test_unit_normalization_round_trips_without_orientation_change() -> None:
    point = CourtPoint(7.0, 3.75)
    normalized = normalize_unit(point)
    assert normalized == (0.25, 0.25)
    assert denormalize_unit(*normalized) == point


def test_mirrors_are_involutions() -> None:
    point = CourtPoint(3.5, 2.25)
    assert mirror_longitudinal(mirror_longitudinal(point)) == point
    assert mirror_lateral(mirror_lateral(point)) == point
    assert mirror_longitudinal(point) == CourtPoint(24.5, 2.25)
    assert mirror_lateral(point) == CourtPoint(3.5, 12.75)


def test_direction_normalization_is_explicit_not_inferred() -> None:
    point = CourtPoint(4.0, 6.0)
    assert normalize_longitudinal_direction(
        point, LongitudinalDirection.LEFT_TO_RIGHT
    ) == point
    assert normalize_longitudinal_direction(
        point, LongitudinalDirection.RIGHT_TO_LEFT
    ) == CourtPoint(24.0, 6.0)


def test_end_relative_coordinates_are_symmetric_and_reversible() -> None:
    point = CourtPoint(23.0, 4.0)
    relative = to_end_relative(point, CourtEnd.RIGHT)
    assert relative == CourtPoint(5.0, 4.0)
    assert from_end_relative(relative, CourtEnd.RIGHT) == point
    assert distance_from_endline(point, CourtEnd.RIGHT) == 5.0


def test_upstream_p1_metric_points_are_explicit_identity_transform() -> None:
    point = canonicalize_upstream_point(
        (8.0, 4.0),
        source_coordinate_system="p1.runtime_metric.v1",
    )
    assert point == CourtPoint(8.0, 4.0)
    with pytest.raises(P4CourtCoordinateError, match="unsupported upstream"):
        canonicalize_upstream_point(
            (8.0, 4.0),
            source_coordinate_system="future.unknown",
        )


def test_extent_validation_includes_lines_but_rejects_outside_values() -> None:
    assert within_court_extent(CourtPoint(0.0, 0.0))
    assert within_court_extent(CourtPoint(28.0, 15.0))
    assert not within_court_extent(CourtPoint(-0.001, 7.5))
    assert not within_court_extent(CourtPoint(14.0, 15.001))


def test_invalid_numeric_inputs_fail_closed() -> None:
    with pytest.raises(P4CourtCoordinateError):
        CourtPoint(True, 2.0)  # type: ignore[arg-type]
    with pytest.raises(P4CourtCoordinateError):
        CourtPoint(float("nan"), 2.0)
    with pytest.raises(P4CourtCoordinateError):
        denormalize_unit(1.1, 0.5)
    with pytest.raises(P4CourtCoordinateError):
        normalize_longitudinal_direction((1.0, 1.0), "unknown")


def test_manifest_has_no_p2_p3_or_attack_inference() -> None:
    manifest = p4_fiba_court_manifest()
    assert manifest["court_length_m"] == 28.0
    assert manifest["court_width_m"] == 15.0
    assert manifest["direction_inference"] is False
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_semantics"] is False


def test_spatial_adapter_uses_explicit_canonical_bridge() -> None:
    adapter = (
        Path(__file__).parents[1] / "services" / "p4" / "spatial_adapter.py"
    ).read_text()
    assert "canonicalize_upstream_point" in adapter
    assert "P4_FIBA_COURT_COORDINATE_SYSTEM" in adapter
    assert "court_coordinate_system=P4_FIBA_COURT_COORDINATE_SYSTEM" in adapter


def test_coordinate_module_does_not_import_p2_or_p3() -> None:
    module = Path(__file__).parents[1] / "core" / "p4_court_coordinates.py"
    tree = ast.parse(module.read_text())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any("p2" in name or "p3" in name for name in imports)
