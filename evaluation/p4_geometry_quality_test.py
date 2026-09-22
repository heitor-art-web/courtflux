from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.court_state import CourtState
from core.p4_geometry_quality import (
    GeometryQualityObservation,
    GeometryQualityPolicy,
    GeometryQualityState,
    GeometryQualityTracker,
    P4_GEOMETRY_QUALITY_CONTRACT_VERSION,
    P4_GEOMETRY_QUALITY_POLICY_VERSION,
    P4GeometryQualityError,
    p4_geometry_quality_manifest,
)
from core.p4_spatial_contract import SpatialFrame, SpatialGeometryEvidence
from services.p4.spatial_adapter import spatial_frame_from_p1


MATRIX = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _frame(
    frame_id: int,
    source: str | None,
    *,
    valid: bool = True,
    timestamp_ms: int | None = None,
    diagnostics: dict[str, object] | None = None,
    matrix: tuple[tuple[float, float, float], ...] | None = MATRIX,
) -> SpatialFrame:
    return SpatialFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 40 if timestamp_ms is None else timestamp_ms,
        geometry=SpatialGeometryEvidence(
            geometry_valid=valid,
            court_visible=True,
            homography_source=source,
            reprojection_rmse_px=2.5,
            homography_matrix=matrix,
            court_polygon_px=None,
            court_coordinate_system="p1.runtime_metric.v1",
            projection_uncertainty_m=None,
            homography_age_frames=None,
            provenance={"p1_camera_diagnostics": dict(diagnostics or {})},
        ),
        players=(),
        source_runtime_version="p1-test",
        provenance={"fixture": "geometry-quality"},
    )


def test_direct_calibration_resets_age_and_preserves_diagnostics() -> None:
    tracker = GeometryQualityTracker(
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=3)
    )
    quality = tracker.update(
        _frame(
            10,
            "direct_keypoints",
            diagnostics={"anchor_count": 8, "inlier_count": 7},
        )
    )
    assert quality.state is GeometryQualityState.DIRECT_CALIBRATION
    assert quality.reason == "direct_keypoint_calibration"
    assert quality.homography_age_frames == 0
    assert quality.last_direct_calibration_ms == 400
    assert quality.anchor_count == 8
    assert quality.inlier_count == 7


def test_optical_flow_is_fresh_then_aging_using_explicit_policy() -> None:
    tracker = GeometryQualityTracker(
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=2)
    )
    tracker.update(_frame(10, "direct_keypoints"))
    assert (
        tracker.update(_frame(11, "optical_flow_hold")).state
        is GeometryQualityState.OPTICAL_FLOW_HOLD_FRESH
    )
    assert (
        tracker.update(_frame(12, "optical_flow_hold")).state
        is GeometryQualityState.OPTICAL_FLOW_HOLD_FRESH
    )
    aging = tracker.update(_frame(13, "optical_flow_hold"))
    assert aging.state is GeometryQualityState.OPTICAL_FLOW_HOLD_AGING
    assert aging.homography_age_frames == 3
    assert aging.last_direct_calibration_ms == 400


def test_optical_flow_without_direct_history_fails_closed() -> None:
    tracker = GeometryQualityTracker(
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=5)
    )
    quality = tracker.update(_frame(1, "optical_flow_hold"))
    assert quality.state is GeometryQualityState.INVALID
    assert quality.reason == "optical_flow_without_direct_history"


def test_invalid_missing_matrix_and_unknown_source_fail_closed() -> None:
    tracker = GeometryQualityTracker(
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=5)
    )
    invalid = tracker.update(_frame(1, "direct_keypoints", valid=False))
    assert invalid.state is GeometryQualityState.INVALID
    assert invalid.reason == "geometry_marked_invalid"

    tracker = GeometryQualityTracker(
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=5)
    )
    missing = tracker.update(_frame(1, "direct_keypoints", matrix=None))
    assert missing.state is GeometryQualityState.INVALID
    assert missing.reason == "missing_homography_matrix"

    tracker = GeometryQualityTracker(
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=5)
    )
    unknown = tracker.update(_frame(1, "future_geometry_source"))
    assert unknown.state is GeometryQualityState.INVALID
    assert unknown.reason == "unsupported_homography_source"


def test_contract_round_trip_and_manifest_are_explicit() -> None:
    tracker = GeometryQualityTracker(
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=4)
    )
    quality = tracker.update(_frame(1, "direct_keypoints"))
    assert GeometryQualityObservation.from_dict(quality.as_dict()).as_dict() == quality.as_dict()

    manifest = p4_geometry_quality_manifest()
    assert manifest["contract_version"] == P4_GEOMETRY_QUALITY_CONTRACT_VERSION
    assert manifest["policy_version"] == P4_GEOMETRY_QUALITY_POLICY_VERSION
    assert manifest["numeric_confidence_model"] is False
    assert manifest["p2_dependency"] is False
    assert manifest["p3_dependency"] is False
    assert manifest["spatial_eligibility_decision"] is False


def test_policy_and_sequence_order_reject_invalid_inputs() -> None:
    for value in (-1, True, 1.5):
        with pytest.raises(P4GeometryQualityError):
            GeometryQualityPolicy(fresh_hold_max_age_frames=value)  # type: ignore[arg-type]

    tracker = GeometryQualityTracker(
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=4)
    )
    tracker.update(_frame(2, "direct_keypoints"))
    with pytest.raises(P4GeometryQualityError, match="strictly increasing"):
        tracker.update(_frame(2, "optical_flow_hold"))


def test_spatial_adapter_preserves_selected_p1_camera_diagnostics() -> None:
    court = CourtState(
        frame_id=5,
        timestamp_sec=0.2,
        geometry_valid=True,
        court_visible=True,
        homography_source="direct_keypoints",
        reprojection_rmse_px=1.25,
        provenance={"version": "court_state_v1"},
    )
    frame = spatial_frame_from_p1(
        SimpleNamespace(
            frame_id=5,
            timestamp=0.2,
            homography_matrix=MATRIX,
            court_polygon=None,
            court_state=court,
            player_states=[],
            diagnostics={
                "camera": {
                    "valid": True,
                    "source": "direct_keypoints",
                    "anchor_count": 9,
                    "inlier_count": 8,
                    "reprojection_rmse_px": 1.25,
                    "keypoint_backend": "serverless",
                    "unrelated_debug_blob": {"must": "not leak"},
                }
            },
            p2_events=[{"ignored": True}],
        ),
        source_runtime_version="p1-test",
    )
    camera = frame.geometry.provenance["p1_camera_diagnostics"]
    assert camera["anchor_count"] == 9
    assert camera["inlier_count"] == 8
    assert camera["keypoint_backend"] == "serverless"
    assert "unrelated_debug_blob" not in camera
    assert frame.provenance["p2_consumed"] is False


def test_geometry_quality_modules_do_not_import_p2_or_p3() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for relative in (
        Path("core/p4_geometry_quality.py"),
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
