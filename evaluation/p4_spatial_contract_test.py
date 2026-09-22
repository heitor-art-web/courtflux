from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.court_state import CourtState
from core.p4_spatial_contract import (
    P4_SPATIAL_CONTRACT_VERSION,
    P4_SPATIAL_FORBIDDEN_SEMANTICS,
    P4_UPSTREAM_COURT_COORDINATE_SYSTEM,
    P4SpatialContractError,
    SpatialFrame,
    SpatialGeometryEvidence,
    SpatialPlayerObservation,
    p4_spatial_contract_manifest,
)
from core.player_state import PlayerState
from services.p4.spatial_adapter import spatial_frame_from_p1


def _player(track_id: int = 7, **overrides: object) -> SpatialPlayerObservation:
    values = {
        "track_id": track_id,
        "local_team_id": None,
        "local_team_confidence": None,
        "image_center_px": (320.0, 180.0),
        "court_position_m": (8.0, 4.0),
        "observation_confidence": 0.9,
        "observation_uncertainty": 0.1,
        "position_confidence": None,
        "visible": True,
        "source": "tracked_player_detection",
        "provenance": {"source": "p1"},
    }
    values.update(overrides)
    return SpatialPlayerObservation(**values)


def _frame(**overrides: object) -> SpatialFrame:
    values = {
        "frame_id": 10,
        "timestamp_ms": 333,
        "geometry": SpatialGeometryEvidence(
            geometry_valid=True,
            court_visible=True,
            homography_source="direct",
            reprojection_rmse_px=1.2,
            homography_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            court_polygon_px=((0.0, 0.0), (640.0, 0.0), (640.0, 360.0), (0.0, 360.0)),
            court_coordinate_system=P4_UPSTREAM_COURT_COORDINATE_SYSTEM,
        ),
        "players": (_player(),),
        "source_runtime_version": "p1-runtime-test",
        "provenance": {"fixture": True},
    }
    values.update(overrides)
    return SpatialFrame(**values)


def test_contract_round_trip_is_exact_and_versioned() -> None:
    frame = _frame()
    assert frame.contract_version == P4_SPATIAL_CONTRACT_VERSION
    assert SpatialFrame.from_dict(frame.as_dict()).as_dict() == frame.as_dict()


def test_contract_rejects_invalid_values_instead_of_clamping() -> None:
    for payload, expected in (
        ({"observation_confidence": 1.2}, "observation_confidence"),
        ({"observation_uncertainty": -0.1}, "observation_uncertainty"),
        ({"track_id": -1}, "track_id"),
        ({"image_center_px": (float("nan"), 10.0)}, "image_center_px"),
    ):
        with pytest.raises(P4SpatialContractError, match=expected):
            _player(**payload)

    with pytest.raises(P4SpatialContractError, match="reprojection_rmse_px"):
        SpatialGeometryEvidence(
            geometry_valid=True,
            court_visible=True,
            reprojection_rmse_px=-0.1,
        )
    with pytest.raises(P4SpatialContractError, match="geometry_valid"):
        SpatialGeometryEvidence(geometry_valid="true", court_visible=True)  # type: ignore[arg-type]
    with pytest.raises(P4SpatialContractError, match="frame_id"):
        _frame(frame_id="10")  # type: ignore[arg-type]


def test_contract_requires_unique_tracks_and_exact_serialized_fields() -> None:
    with pytest.raises(P4SpatialContractError, match="unique track_id"):
        _frame(players=(_player(1), _player(1)))

    payload = _frame().as_dict()
    payload["possession_id"] = "forbidden"
    with pytest.raises(P4SpatialContractError, match="extra"):
        SpatialFrame.from_dict(payload)


def test_manifest_explicitly_excludes_p2_p3_and_tactical_truth() -> None:
    manifest = p4_spatial_contract_manifest()
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_labels"] is False
    assert manifest["identity_scope"] == "track_id_only"
    for forbidden in (
        "possession_id",
        "game_player_id",
        "ball_handler_game_player_id",
        "offense_team_id",
        "defense_team_id",
        "review_item_id",
    ):
        assert forbidden in P4_SPATIAL_FORBIDDEN_SEMANTICS


def test_adapter_consumes_only_p1_state_and_ignores_basketball_fields() -> None:
    player = PlayerState(
        frame_id=12,
        timestamp_sec=0.4,
        track_id=3,
        bbox=(10, 20, 30, 60),
        center_px=(20.0, 40.0),
        confidence=0.85,
        uncertainty=0.15,
        court_position_m=(5.0, 6.0),
        provenance={"runtime": "p1"},
    )
    court = CourtState(
        frame_id=12,
        timestamp_sec=0.4,
        geometry_valid=True,
        court_visible=True,
        homography_source="direct",
        reprojection_rmse_px=0.9,
        court_polygon_px=((0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)),
        provenance={"camera": "p1"},
    )
    base = {
        "frame_id": 12,
        "timestamp": 0.4,
        "homography_matrix": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "court_polygon": None,
        "court_state": court,
        "player_states": [player],
    }
    frame_a = spatial_frame_from_p1(
        SimpleNamespace(
            **base,
            p2_events=[{"event": "must_be_ignored"}],
            ball_handler={"track_id": 3},
            offensive_team="home",
        ),
        source_runtime_version="p1-test",
        local_team_by_track_id={3: ("local-A", 0.8)},
    )
    frame_b = spatial_frame_from_p1(
        SimpleNamespace(
            **base,
            p2_events=[],
            ball_handler=None,
            offensive_team=None,
        ),
        source_runtime_version="p1-test",
        local_team_by_track_id={3: ("local-A", 0.8)},
    )
    assert frame_a.as_dict() == frame_b.as_dict()
    assert frame_a.timestamp_ms == 400
    assert frame_a.players[0].local_team_id == "local-A"
    assert frame_a.players[0].local_team_confidence == 0.8
    assert frame_a.provenance["p2_consumed"] is False
    assert frame_a.provenance["p3_consumed"] is False


def test_p4_spatial_modules_do_not_import_p2_or_p3() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for relative in (
        Path("core/p4_spatial_contract.py"),
        Path("services/p4/spatial_adapter.py"),
    ):
        tree = ast.parse((repo_root / relative).read_text())
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            name == "services.p2"
            or name.startswith("services.p2.")
            or name == "services.p3"
            or name.startswith("services.p3.")
            or name.startswith("core.p3")
            for name in imports
        ), (relative, imports)
