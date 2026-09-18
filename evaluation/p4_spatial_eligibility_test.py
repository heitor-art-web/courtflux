from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.p4_geometry_quality import (
    GeometryQualityObservation,
    GeometryQualityPolicy,
    GeometryQualityState,
)
from core.p4_spatial_contract import SpatialFrame, SpatialGeometryEvidence, SpatialPlayerObservation
from core.p4_spatial_eligibility import (
    P4SpatialEligibilityError,
    SpatialEligibilityPolicy,
    SpatialEligibilityReason,
    SpatialEligibilityState,
    evaluate_spatial_eligibility,
    p4_spatial_eligibility_manifest,
)


def _player(track_id: int, *, court_position=(5.0, 5.0), visible=True):
    return SpatialPlayerObservation(
        track_id=track_id,
        local_team_id=None,
        local_team_confidence=None,
        image_center_px=(100.0 + track_id, 200.0),
        court_position_m=court_position,
        observation_confidence=0.9,
        observation_uncertainty=0.1,
        position_confidence=None,
        visible=visible,
        source="fixture",
    )


def _frame(*players, court_visible=True, coordinate_system="p4.fiba_28x15_xy.v1"):
    return SpatialFrame(
        frame_id=10,
        timestamp_ms=333,
        geometry=SpatialGeometryEvidence(
            geometry_valid=True,
            court_visible=court_visible,
            homography_source="direct_keypoints",
            homography_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            court_coordinate_system=coordinate_system,
        ),
        players=tuple(players),
    )


def _quality(state=GeometryQualityState.DIRECT_CALIBRATION):
    return GeometryQualityObservation(
        frame_id=10,
        timestamp_ms=333,
        state=state,
        reason="fixture",
        homography_source="direct_keypoints",
        last_direct_calibration_ms=333,
        homography_age_frames=0,
        anchor_count=8,
        inlier_count=8,
        reprojection_rmse_px=1.0,
        projection_uncertainty_m=None,
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=15),
    )


def _policy(**overrides):
    values = {
        "min_metric_player_count": 2,
        "min_metric_player_fraction": 0.5,
        "court_extent_tolerance_m": 0.25,
        "allow_aging_geometry": False,
        "require_court_visible": True,
    }
    values.update(overrides)
    return SpatialEligibilityPolicy(**values)


def test_fully_supported_frame_is_eligible() -> None:
    result = evaluate_spatial_eligibility(
        _frame(_player(1), _player(2)), _quality(), policy=_policy()
    )
    assert result.state is SpatialEligibilityState.ELIGIBLE
    assert result.reasons == (SpatialEligibilityReason.ELIGIBLE,)
    assert result.metric_player_fraction == 1.0


def test_invalid_geometry_and_invisible_court_fail_closed() -> None:
    result = evaluate_spatial_eligibility(
        _frame(_player(1), _player(2), court_visible=False),
        _quality(GeometryQualityState.INVALID),
        policy=_policy(),
    )
    assert result.state is SpatialEligibilityState.INVALID
    assert SpatialEligibilityReason.GEOMETRY_INVALID in result.reasons
    assert SpatialEligibilityReason.COURT_NOT_VISIBLE in result.reasons


def test_coordinate_system_mismatch_is_invalid() -> None:
    result = evaluate_spatial_eligibility(
        _frame(_player(1), _player(2), coordinate_system="legacy-space"),
        _quality(),
        policy=_policy(),
    )
    assert result.state is SpatialEligibilityState.INVALID
    assert SpatialEligibilityReason.COORDINATE_SYSTEM_MISMATCH in result.reasons


def test_insufficient_metric_coverage_is_invalid() -> None:
    result = evaluate_spatial_eligibility(
        _frame(_player(1), _player(2, court_position=None), _player(3, court_position=None)),
        _quality(),
        policy=_policy(min_metric_player_count=2, min_metric_player_fraction=0.75),
    )
    assert result.state is SpatialEligibilityState.INVALID
    assert SpatialEligibilityReason.METRIC_PLAYER_COUNT_BELOW_MIN in result.reasons
    assert SpatialEligibilityReason.METRIC_PLAYER_FRACTION_BELOW_MIN in result.reasons
    assert result.metric_player_count == 1


def test_partial_positions_are_degraded_when_minimums_still_pass() -> None:
    result = evaluate_spatial_eligibility(
        _frame(_player(1), _player(2), _player(3, court_position=None)),
        _quality(),
        policy=_policy(min_metric_player_count=2, min_metric_player_fraction=0.5),
    )
    assert result.state is SpatialEligibilityState.DEGRADED
    assert result.reasons == (SpatialEligibilityReason.MISSING_METRIC_PLAYER_POSITION,)


def test_out_of_bounds_position_is_degraded_when_coverage_still_passes() -> None:
    result = evaluate_spatial_eligibility(
        _frame(_player(1), _player(2), _player(3, court_position=(40.0, 7.0))),
        _quality(),
        policy=_policy(min_metric_player_count=2, min_metric_player_fraction=0.5),
    )
    assert result.state is SpatialEligibilityState.DEGRADED
    assert SpatialEligibilityReason.OUT_OF_BOUNDS_METRIC_POSITION in result.reasons
    assert result.out_of_bounds_metric_player_count == 1


def test_aging_geometry_is_policy_controlled_not_silently_rejected() -> None:
    aging = _quality(GeometryQualityState.OPTICAL_FLOW_HOLD_AGING)
    degraded = evaluate_spatial_eligibility(
        _frame(_player(1), _player(2)), aging, policy=_policy(allow_aging_geometry=False)
    )
    accepted = evaluate_spatial_eligibility(
        _frame(_player(1), _player(2)), aging, policy=_policy(allow_aging_geometry=True)
    )
    assert degraded.state is SpatialEligibilityState.DEGRADED
    assert SpatialEligibilityReason.AGING_GEOMETRY in degraded.reasons
    assert accepted.state is SpatialEligibilityState.ELIGIBLE


def test_policy_and_observation_round_trip_and_manifest() -> None:
    result = evaluate_spatial_eligibility(
        _frame(_player(1), _player(2)), _quality(), policy=_policy()
    )
    restored = type(result).from_dict(result.as_dict())
    assert restored.as_dict() == result.as_dict()
    manifest = p4_spatial_eligibility_manifest()
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["tactical_semantics"] is False
    assert manifest["confidence_thresholds"] is False


def test_policy_is_strict_and_frame_alignment_is_required() -> None:
    with pytest.raises(P4SpatialEligibilityError, match="min_metric_player_fraction"):
        _policy(min_metric_player_fraction=1.1)
    wrong = GeometryQualityObservation(
        frame_id=11,
        timestamp_ms=333,
        state=GeometryQualityState.DIRECT_CALIBRATION,
        reason="fixture",
        homography_source="direct_keypoints",
        last_direct_calibration_ms=333,
        homography_age_frames=0,
        anchor_count=None,
        inlier_count=None,
        reprojection_rmse_px=None,
        projection_uncertainty_m=None,
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=15),
    )
    with pytest.raises(P4SpatialEligibilityError, match="same frame"):
        evaluate_spatial_eligibility(_frame(_player(1), _player(2)), wrong, policy=_policy())


def test_module_has_no_p2_or_p3_imports() -> None:
    path = Path(__file__).parents[1] / "core" / "p4_spatial_eligibility.py"
    tree = ast.parse(path.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(
        name.startswith("services.p2")
        or name.startswith("services.p3")
        or name.startswith("core.p3")
        for name in imports
    )
