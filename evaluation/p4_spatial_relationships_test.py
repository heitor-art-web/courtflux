from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.p4_kinematics import KinematicsPoint, KinematicsPointState
from core.p4_spatial_relationships import (
    LocalTeamRelation,
    P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION,
    P4SpatialRelationshipError,
    PairwiseSpatialRelationship,
    SpatialActorState,
    actor_state_from_kinematics,
    distance_between,
    distance_to_physical_basket,
    nearest_actor,
    pair_relationship,
    pairwise_distance_matrix,
    pairwise_relationships,
    p4_spatial_relationship_manifest,
)
from core.p4_trajectory import TrajectoryPointState
from core.p4_court_coordinates import CourtEnd


def _actor(
    track_id: int,
    position: tuple[float, float],
    *,
    velocity: tuple[float, float] | None = None,
    team: str | None = None,
    timestamp_ms: int = 1000,
) -> SpatialActorState:
    return SpatialActorState(
        track_id=track_id,
        timestamp_ms=timestamp_ms,
        position_m=position,
        velocity_mps=velocity,
        local_team_id=team,
    )


def test_distance_symmetry_and_round_trip() -> None:
    a = _actor(1, (0.0, 0.0), team="A")
    b = _actor(2, (3.0, 4.0), team="B")
    ab = pair_relationship(a, b)
    ba = pair_relationship(b, a)
    assert ab.contract_version == P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION
    assert ab.distance_m == pytest.approx(5.0)
    assert ba.distance_m == pytest.approx(ab.distance_m)
    assert ba.delta_position_m == pytest.approx((-3.0, -4.0))
    assert PairwiseSpatialRelationship.from_dict(ab.as_dict()).as_dict() == ab.as_dict()


def test_closing_speed_sign_is_explicit() -> None:
    a = _actor(1, (0.0, 0.0), velocity=(1.0, 0.0))
    b = _actor(2, (3.0, 4.0), velocity=(0.0, 0.0))
    relationship = pair_relationship(a, b)
    assert relationship.relative_velocity_mps == (-1.0, 0.0)
    assert relationship.closing_speed_mps == pytest.approx(0.6)

    separating = pair_relationship(
        _actor(1, (0.0, 0.0), velocity=(-1.0, 0.0)),
        b,
    )
    assert separating.closing_speed_mps == pytest.approx(-0.6)


def test_zero_distance_has_no_radial_closing_speed() -> None:
    relationship = pair_relationship(
        _actor(1, (2.0, 2.0), velocity=(1.0, 0.0)),
        _actor(2, (2.0, 2.0), velocity=(-1.0, 0.0)),
    )
    assert relationship.distance_m == 0.0
    assert relationship.closing_speed_mps is None


def test_local_team_relation_is_not_offense_defense() -> None:
    assert pair_relationship(
        _actor(1, (1.0, 1.0), team="red"),
        _actor(2, (2.0, 1.0), team="red"),
    ).local_team_relation is LocalTeamRelation.SAME
    assert pair_relationship(
        _actor(1, (1.0, 1.0), team="red"),
        _actor(2, (2.0, 1.0), team="blue"),
    ).local_team_relation is LocalTeamRelation.DIFFERENT
    assert pair_relationship(
        _actor(1, (1.0, 1.0), team=None),
        _actor(2, (2.0, 1.0), team="blue"),
    ).local_team_relation is LocalTeamRelation.UNKNOWN


def test_pairwise_matrix_and_nearest_are_deterministic() -> None:
    actors = (
        _actor(4, (4.0, 1.0), team="A"),
        _actor(1, (0.0, 1.0), team="A"),
        _actor(3, (2.0, 1.0), team="B"),
    )
    relationships = pairwise_relationships(actors)
    assert [(r.track_a_id, r.track_b_id) for r in relationships] == [
        (1, 3),
        (1, 4),
        (3, 4),
    ]
    matrix = pairwise_distance_matrix(actors)
    assert matrix[1][3] == matrix[3][1] == 2.0
    nearest = nearest_actor(actors[1], actors)
    assert nearest is not None and nearest.track_b_id == 3
    same_team = nearest_actor(actors[1], actors, relation=LocalTeamRelation.SAME)
    assert same_team is not None and same_team.track_b_id == 4


def test_relationship_inputs_must_share_timestamp() -> None:
    with pytest.raises(P4SpatialRelationshipError):
        distance_between(
            _actor(1, (1.0, 1.0), timestamp_ms=100),
            _actor(2, (2.0, 1.0), timestamp_ms=200),
        )


def test_position_must_remain_on_canonical_court() -> None:
    with pytest.raises(P4SpatialRelationshipError):
        _actor(1, (28.1, 7.5))


def test_adapter_from_kinematics_is_neutral() -> None:
    point = KinematicsPoint(
        track_id=7,
        frame_id=10,
        timestamp_ms=1000,
        state=KinematicsPointState.VELOCITY_ONLY,
        trajectory_point_state=TrajectoryPointState.OBSERVED,
        position_m=(4.0, 5.0),
        velocity_mps=(1.0, -0.5),
        speed_mps=1.11803398875,
        acceleration_mps2=None,
        acceleration_magnitude_mps2=None,
        heading_rad=0.0,
        provenance={},
    )
    actor = actor_state_from_kinematics(point, local_team_id="local-a")
    assert actor.track_id == 7
    assert actor.position_m == (4.0, 5.0)
    assert actor.local_team_id == "local-a"


def test_distance_to_physical_basket_requires_no_attack_direction() -> None:
    actor = _actor(1, (1.575, 7.5))
    assert distance_to_physical_basket(actor, CourtEnd.LEFT) == pytest.approx(0.0)
    assert distance_to_physical_basket(actor, CourtEnd.RIGHT) > 20.0


def test_manifest_and_import_boundary() -> None:
    manifest = p4_spatial_relationship_manifest()
    assert manifest["offense_defense_roles"] is False
    assert manifest["matchup_semantics"] is False
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_semantics"] is False

    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / "core/p4_spatial_relationships.py").read_text())
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any(
        name.startswith("services.p2") or name.startswith("core.p3")
        for name in imports
    )
