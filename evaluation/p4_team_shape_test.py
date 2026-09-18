from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.p4_court_coordinates import CourtEnd
from core.p4_spatial_relationships import SpatialActorState
from core.p4_team_shape import (
    P4_TEAM_SHAPE_CONTRACT_VERSION,
    P4TeamShapeError,
    TeamShapeMetrics,
    compute_all_team_shapes,
    compute_team_shape,
    convex_hull,
    p4_team_shape_manifest,
    polygon_area,
)


def _actor(
    track_id: int,
    position: tuple[float, float],
    *,
    team: str | None,
    timestamp_ms: int = 1000,
) -> SpatialActorState:
    return SpatialActorState(
        track_id=track_id,
        timestamp_ms=timestamp_ms,
        position_m=position,
        velocity_mps=None,
        local_team_id=team,
    )


def test_rectangle_shape_metrics_and_round_trip() -> None:
    actors = (
        _actor(1, (1.0, 6.0), team="A"),
        _actor(2, (1.0, 9.0), team="A"),
        _actor(3, (4.0, 9.0), team="A"),
        _actor(4, (4.0, 6.0), team="A"),
    )
    shape = compute_team_shape(actors, local_team_id="A")
    assert shape.contract_version == P4_TEAM_SHAPE_CONTRACT_VERSION
    assert shape.centroid_m == pytest.approx((2.5, 7.5))
    assert shape.width_m == pytest.approx(3.0)
    assert shape.depth_m == pytest.approx(3.0)
    assert shape.bounding_box_m == pytest.approx((1.0, 6.0, 4.0, 9.0))
    assert shape.convex_hull_area_m2 == pytest.approx(9.0)
    assert TeamShapeMetrics.from_dict(shape.as_dict()).as_dict() == shape.as_dict()


def test_pairwise_spacing_statistics_are_descriptive_only() -> None:
    actors = (
        _actor(1, (0.0, 0.0), team="A"),
        _actor(2, (3.0, 0.0), team="A"),
        _actor(3, (0.0, 4.0), team="A"),
    )
    shape = compute_team_shape(actors, local_team_id="A")
    assert shape.mean_pairwise_spacing_m == pytest.approx(4.0)
    assert shape.pairwise_spacing_variance_m2 == pytest.approx(2.0 / 3.0)
    assert shape.min_pairwise_spacing_m == pytest.approx(3.0)
    assert shape.max_pairwise_spacing_m == pytest.approx(5.0)


def test_single_player_keeps_valid_zero_extent_shape() -> None:
    shape = compute_team_shape(
        (_actor(1, (7.0, 8.0), team="A"),),
        local_team_id="A",
    )
    assert shape.centroid_m == (7.0, 8.0)
    assert shape.width_m == 0.0
    assert shape.depth_m == 0.0
    assert shape.convex_hull_area_m2 == 0.0
    assert shape.mean_pairwise_spacing_m is None


def test_convex_hull_collinear_points_have_zero_area() -> None:
    hull = convex_hull(((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)))
    assert polygon_area(hull) == 0.0


def test_shapes_group_by_local_team_only() -> None:
    actors = (
        _actor(1, (1.0, 1.0), team="A"),
        _actor(2, (2.0, 1.0), team="A"),
        _actor(3, (20.0, 10.0), team="B"),
        _actor(4, (21.0, 10.0), team=None),
    )
    shapes = compute_all_team_shapes(actors)
    assert [shape.local_team_id for shape in shapes] == ["A", "B"]
    assert [shape.player_count for shape in shapes] == [2, 1]


def test_physical_end_occupancy_does_not_infer_attack_direction() -> None:
    actors = (
        _actor(1, (1.575, 7.5), team="A"),
        _actor(2, (3.0, 7.5), team="A"),
    )
    shape = compute_team_shape(actors, local_team_id="A")
    by_end = {item.end: item for item in shape.occupancy_by_physical_end}
    assert by_end[CourtEnd.LEFT].paint_count == 2
    assert by_end[CourtEnd.RIGHT].paint_count == 0
    assert by_end[CourtEnd.RIGHT].other_half_count == 2


def test_longitudinal_mirror_preserves_shape_geometry_and_swaps_end_occupancy() -> None:
    original = (
        _actor(1, (1.575, 7.5), team="A"),
        _actor(2, (4.0, 6.0), team="A"),
        _actor(3, (4.0, 9.0), team="A"),
    )
    mirrored = tuple(
        _actor(
            actor.track_id,
            (28.0 - actor.position_m[0], actor.position_m[1]),
            team="A",
        )
        for actor in original
    )
    left = compute_team_shape(original, local_team_id="A")
    right = compute_team_shape(mirrored, local_team_id="A")
    assert right.width_m == pytest.approx(left.width_m)
    assert right.depth_m == pytest.approx(left.depth_m)
    assert right.convex_hull_area_m2 == pytest.approx(left.convex_hull_area_m2)

    left_occ = {item.end: item for item in left.occupancy_by_physical_end}
    right_occ = {item.end: item for item in right.occupancy_by_physical_end}
    assert (
        left_occ[CourtEnd.LEFT].paint_count
        == right_occ[CourtEnd.RIGHT].paint_count
    )


def test_team_shape_requires_consistent_timestamp() -> None:
    actors = (
        _actor(1, (1.0, 1.0), team="A", timestamp_ms=100),
        _actor(2, (2.0, 1.0), team="A", timestamp_ms=200),
    )
    with pytest.raises(P4TeamShapeError):
        compute_team_shape(actors, local_team_id="A")


def test_manifest_and_import_boundary() -> None:
    manifest = p4_team_shape_manifest()
    assert manifest["team_scope"] == "p1_local_team_only"
    assert manifest["attack_relative_shape"] is False
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_semantics"] is False

    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / "core/p4_team_shape.py").read_text())
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any(
        name.startswith("services.p2") or name.startswith("core.p3")
        for name in imports
    )
