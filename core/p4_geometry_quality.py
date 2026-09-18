from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite
from numbers import Integral
from typing import Any, Mapping

from core.p4_spatial_contract import SpatialFrame

P4_GEOMETRY_QUALITY_CONTRACT_VERSION = "p4.geometry_quality.v1"
P4_GEOMETRY_QUALITY_POLICY_VERSION = "p4.geometry_quality.policy.v1"

P4_GEOMETRY_QUALITY_FIELDS = frozenset(
    {
        "frame_id",
        "timestamp_ms",
        "state",
        "reason",
        "homography_source",
        "last_direct_calibration_ms",
        "homography_age_frames",
        "anchor_count",
        "inlier_count",
        "reprojection_rmse_px",
        "projection_uncertainty_m",
        "policy",
        "provenance",
        "contract_version",
    }
)
P4_GEOMETRY_QUALITY_POLICY_FIELDS = frozenset(
    {"fresh_hold_max_age_frames", "policy_version"}
)


class GeometryQualityState(str, Enum):
    DIRECT_CALIBRATION = "direct_calibration"
    OPTICAL_FLOW_HOLD_FRESH = "optical_flow_hold_fresh"
    OPTICAL_FLOW_HOLD_AGING = "optical_flow_hold_aging"
    INVALID = "invalid"


class P4GeometryQualityError(ValueError):
    """Raised when geometry-quality inputs violate the frozen P4 v1 contract."""


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise P4GeometryQualityError(f"{field_name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise P4GeometryQualityError(f"{field_name} must be a non-negative integer")
    return normalized


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field_name)


def _optional_nonnegative_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise P4GeometryQualityError(f"{field_name} must be a non-negative finite number") from exc
    if normalized < 0.0 or not isfinite(normalized):
        raise P4GeometryQualityError(f"{field_name} must be a non-negative finite number")
    return normalized


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise P4GeometryQualityError(f"{field_name} must be non-empty when provided")
    return text


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P4GeometryQualityError(f"{field_name} must be an object")
    normalized = dict(value)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise P4GeometryQualityError(f"{field_name} must be JSON serializable") from exc
    return normalized


def _exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], object_name: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise P4GeometryQualityError(
            f"{object_name} fields must match frozen v1 schema "
            f"(missing={missing}, extra={extra})"
        )


@dataclass(frozen=True, slots=True)
class GeometryQualityPolicy:
    """Versioned classification policy without inventing a numeric confidence score."""

    fresh_hold_max_age_frames: int
    policy_version: str = P4_GEOMETRY_QUALITY_POLICY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fresh_hold_max_age_frames",
            _nonnegative_int(
                self.fresh_hold_max_age_frames,
                "fresh_hold_max_age_frames",
            ),
        )
        if self.policy_version != P4_GEOMETRY_QUALITY_POLICY_VERSION:
            raise P4GeometryQualityError(
                f"policy_version must be {P4_GEOMETRY_QUALITY_POLICY_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fresh_hold_max_age_frames": self.fresh_hold_max_age_frames,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometryQualityPolicy":
        _exact_fields(payload, P4_GEOMETRY_QUALITY_POLICY_FIELDS, "GeometryQualityPolicy")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class GeometryQualityObservation:
    """Per-frame geometry evidence quality, separate from spatial eligibility."""

    frame_id: int
    timestamp_ms: int
    state: GeometryQualityState
    reason: str
    homography_source: str | None
    last_direct_calibration_ms: int | None
    homography_age_frames: int | None
    anchor_count: int | None
    inlier_count: int | None
    reprojection_rmse_px: float | None
    projection_uncertainty_m: float | None
    policy: GeometryQualityPolicy
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_GEOMETRY_QUALITY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _nonnegative_int(self.frame_id, "frame_id"))
        object.__setattr__(
            self, "timestamp_ms", _nonnegative_int(self.timestamp_ms, "timestamp_ms")
        )
        if not isinstance(self.state, GeometryQualityState):
            try:
                object.__setattr__(self, "state", GeometryQualityState(str(self.state)))
            except ValueError as exc:
                raise P4GeometryQualityError("state is not supported") from exc
        reason = str(self.reason).strip()
        if not reason:
            raise P4GeometryQualityError("reason must be non-empty")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "homography_source",
            _optional_text(self.homography_source, "homography_source"),
        )
        object.__setattr__(
            self,
            "last_direct_calibration_ms",
            _optional_nonnegative_int(
                self.last_direct_calibration_ms,
                "last_direct_calibration_ms",
            ),
        )
        object.__setattr__(
            self,
            "homography_age_frames",
            _optional_nonnegative_int(self.homography_age_frames, "homography_age_frames"),
        )
        object.__setattr__(
            self,
            "anchor_count",
            _optional_nonnegative_int(self.anchor_count, "anchor_count"),
        )
        object.__setattr__(
            self,
            "inlier_count",
            _optional_nonnegative_int(self.inlier_count, "inlier_count"),
        )
        object.__setattr__(
            self,
            "reprojection_rmse_px",
            _optional_nonnegative_number(self.reprojection_rmse_px, "reprojection_rmse_px"),
        )
        object.__setattr__(
            self,
            "projection_uncertainty_m",
            _optional_nonnegative_number(
                self.projection_uncertainty_m,
                "projection_uncertainty_m",
            ),
        )
        if not isinstance(self.policy, GeometryQualityPolicy):
            raise P4GeometryQualityError("policy must be GeometryQualityPolicy")
        object.__setattr__(
            self, "provenance", _json_object(self.provenance, "provenance")
        )
        if self.contract_version != P4_GEOMETRY_QUALITY_CONTRACT_VERSION:
            raise P4GeometryQualityError(
                f"contract_version must be {P4_GEOMETRY_QUALITY_CONTRACT_VERSION!r}"
            )

    @property
    def valid(self) -> bool:
        return self.state is not GeometryQualityState.INVALID

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "state": self.state.value,
            "reason": self.reason,
            "homography_source": self.homography_source,
            "last_direct_calibration_ms": self.last_direct_calibration_ms,
            "homography_age_frames": self.homography_age_frames,
            "anchor_count": self.anchor_count,
            "inlier_count": self.inlier_count,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "projection_uncertainty_m": self.projection_uncertainty_m,
            "policy": self.policy.as_dict(),
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometryQualityObservation":
        _exact_fields(
            payload, P4_GEOMETRY_QUALITY_FIELDS, "GeometryQualityObservation"
        )
        normalized = dict(payload)
        policy = normalized["policy"]
        if not isinstance(policy, Mapping):
            raise P4GeometryQualityError("policy must be an object")
        normalized["policy"] = GeometryQualityPolicy.from_dict(policy)
        return cls(**normalized)


def p4_geometry_quality_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_GEOMETRY_QUALITY_CONTRACT_VERSION,
        "policy_version": P4_GEOMETRY_QUALITY_POLICY_VERSION,
        "states": [state.value for state in GeometryQualityState],
        "fields": sorted(P4_GEOMETRY_QUALITY_FIELDS),
        "policy_fields": sorted(P4_GEOMETRY_QUALITY_POLICY_FIELDS),
        "classification_inputs": [
            "SpatialFrame.geometry.geometry_valid",
            "SpatialFrame.geometry.homography_source",
            "SpatialFrame.geometry.homography_matrix",
            "P4 sequence history",
        ],
        "diagnostics_only_inputs": [
            "anchor_count",
            "inlier_count",
            "reprojection_rmse_px",
            "projection_uncertainty_m",
        ],
        "numeric_confidence_model": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "spatial_eligibility_decision": False,
    }


class GeometryQualityTracker:
    """Stateful P4 classifier over ordered SpatialFrames.

    P1 remains untouched. The tracker reconstructs homography age from the P4 frame
    sequence and only classifies optical-flow holds when a direct calibration has
    previously been observed in the same tracker instance.
    """

    DIRECT_SOURCE = "direct_keypoints"
    FLOW_SOURCE = "optical_flow_hold"

    def __init__(self, *, policy: GeometryQualityPolicy) -> None:
        if not isinstance(policy, GeometryQualityPolicy):
            raise P4GeometryQualityError("policy must be GeometryQualityPolicy")
        self.policy = policy
        self._last_frame_id: int | None = None
        self._last_timestamp_ms: int | None = None
        self._last_direct_frame_id: int | None = None
        self._last_direct_timestamp_ms: int | None = None

    @staticmethod
    def _camera_diagnostics(frame: SpatialFrame) -> dict[str, Any]:
        raw = frame.geometry.provenance.get("p1_camera_diagnostics", {})
        return dict(raw) if isinstance(raw, Mapping) else {}

    @staticmethod
    def _optional_diag_int(diagnostics: Mapping[str, Any], key: str) -> int | None:
        value = diagnostics.get(key)
        if value is None:
            return None
        try:
            return _nonnegative_int(value, key)
        except P4GeometryQualityError:
            return None

    def _validate_order(self, frame: SpatialFrame) -> None:
        if not isinstance(frame, SpatialFrame):
            raise P4GeometryQualityError("frame must be SpatialFrame")
        if self._last_frame_id is not None and frame.frame_id <= self._last_frame_id:
            raise P4GeometryQualityError("SpatialFrames must be strictly increasing by frame_id")
        if (
            self._last_timestamp_ms is not None
            and frame.timestamp_ms < self._last_timestamp_ms
        ):
            raise P4GeometryQualityError(
                "SpatialFrame timestamps must be non-decreasing"
            )

    def update(self, frame: SpatialFrame) -> GeometryQualityObservation:
        self._validate_order(frame)
        geometry = frame.geometry
        source = (geometry.homography_source or "").strip().lower() or None
        diagnostics = self._camera_diagnostics(frame)
        anchor_count = self._optional_diag_int(diagnostics, "anchor_count")
        inlier_count = self._optional_diag_int(diagnostics, "inlier_count")

        if self._last_direct_frame_id is None:
            age_frames = None
        else:
            age_frames = frame.frame_id - self._last_direct_frame_id

        state = GeometryQualityState.INVALID
        reason = "geometry_marked_invalid"

        has_matrix = geometry.homography_matrix is not None
        if geometry.geometry_valid and not has_matrix:
            reason = "missing_homography_matrix"
        elif geometry.geometry_valid and source == self.DIRECT_SOURCE:
            self._last_direct_frame_id = frame.frame_id
            self._last_direct_timestamp_ms = frame.timestamp_ms
            age_frames = 0
            state = GeometryQualityState.DIRECT_CALIBRATION
            reason = "direct_keypoint_calibration"
        elif geometry.geometry_valid and source == self.FLOW_SOURCE:
            if self._last_direct_frame_id is None:
                reason = "optical_flow_without_direct_history"
            else:
                age_frames = frame.frame_id - self._last_direct_frame_id
                if age_frames <= self.policy.fresh_hold_max_age_frames:
                    state = GeometryQualityState.OPTICAL_FLOW_HOLD_FRESH
                    reason = "optical_flow_hold_within_fresh_window"
                else:
                    state = GeometryQualityState.OPTICAL_FLOW_HOLD_AGING
                    reason = "optical_flow_hold_beyond_fresh_window"
        elif geometry.geometry_valid and source is None:
            reason = "missing_homography_source"
        elif geometry.geometry_valid:
            reason = "unsupported_homography_source"

        observation = GeometryQualityObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            state=state,
            reason=reason,
            homography_source=source,
            last_direct_calibration_ms=self._last_direct_timestamp_ms,
            homography_age_frames=age_frames,
            anchor_count=anchor_count,
            inlier_count=inlier_count,
            reprojection_rmse_px=geometry.reprojection_rmse_px,
            projection_uncertainty_m=geometry.projection_uncertainty_m,
            policy=self.policy,
            provenance={
                "classifier": "core.p4_geometry_quality.GeometryQualityTracker",
                "source_contract_version": frame.contract_version,
                "confidence_model": "none",
                "spatial_eligibility_decision": False,
                "p2_consumed": False,
                "p3_consumed": False,
            },
        )
        self._last_frame_id = frame.frame_id
        self._last_timestamp_ms = frame.timestamp_ms
        return observation
