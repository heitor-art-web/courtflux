from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite
from numbers import Integral, Real
from typing import Any, Sequence

from core.p4_geometry_quality import GeometryQualityState
from core.p4_spatial_eligibility import SpatialEligibilityState
from core.p4_spatial_snapshot import SpatialSnapshot
from core.p4_trajectory import TrajectoryPointState

P4_SPATIAL_DIAGNOSTICS_CONTRACT_VERSION = "p4.spatial_diagnostics.v1"

_GAP_STATES = {
    TrajectoryPointState.GAP,
    TrajectoryPointState.OUTLIER_REJECTED,
    TrajectoryPointState.INVALID_INPUT,
}


class P4SpatialDiagnosticsError(ValueError):
    """Raised when spatial diagnostics inputs violate the v1 contract."""


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise P4SpatialDiagnosticsError(f"{name} must be a non-negative integer")
    return int(value)


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4SpatialDiagnosticsError(f"{name} must be a non-negative finite number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise P4SpatialDiagnosticsError(f"{name} must be a non-negative finite number")
    return result


def _optional_nonnegative_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_number(value, name)


def _fraction(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class PlayerSpatialDiagnostics:
    track_id: int
    frame_presence_count: int
    raw_metric_count: int
    smoothed_metric_count: int
    velocity_count: int
    gap_state_count: int
    raw_metric_fraction: float
    smoothed_metric_fraction: float
    velocity_fraction: float
    mean_smoothing_displacement_m: float | None
    max_smoothing_displacement_m: float | None
    mean_raw_step_m: float | None
    mean_smoothed_step_m: float | None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonnegative_int(self.track_id, "track_id"))
        for name in (
            "frame_presence_count",
            "raw_metric_count",
            "smoothed_metric_count",
            "velocity_count",
            "gap_state_count",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        for name in ("raw_metric_fraction", "smoothed_metric_fraction", "velocity_fraction"):
            value = _nonnegative_number(getattr(self, name), name)
            if value > 1.0:
                raise P4SpatialDiagnosticsError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "mean_smoothing_displacement_m",
            "max_smoothing_displacement_m",
            "mean_raw_step_m",
            "mean_smoothed_step_m",
        ):
            object.__setattr__(
                self, name, _optional_nonnegative_number(getattr(self, name), name)
            )
        object.__setattr__(self, "provenance", dict(self.provenance))

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_presence_count": self.frame_presence_count,
            "raw_metric_count": self.raw_metric_count,
            "smoothed_metric_count": self.smoothed_metric_count,
            "velocity_count": self.velocity_count,
            "gap_state_count": self.gap_state_count,
            "raw_metric_fraction": self.raw_metric_fraction,
            "smoothed_metric_fraction": self.smoothed_metric_fraction,
            "velocity_fraction": self.velocity_fraction,
            "mean_smoothing_displacement_m": self.mean_smoothing_displacement_m,
            "max_smoothing_displacement_m": self.max_smoothing_displacement_m,
            "mean_raw_step_m": self.mean_raw_step_m,
            "mean_smoothed_step_m": self.mean_smoothed_step_m,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class SpatialDiagnosticsReport:
    total_frames: int
    geometry_state_counts: dict[str, int]
    geometry_state_fractions: dict[str, float]
    eligibility_state_counts: dict[str, int]
    eligibility_state_fractions: dict[str, float]
    court_visible_frames: int
    geometry_valid_frames: int
    reprojection_rmse_sample_count: int
    mean_reprojection_rmse_px: float | None
    max_reprojection_rmse_px: float | None
    projection_uncertainty_sample_count: int
    mean_projection_uncertainty_m: float | None
    max_projection_uncertainty_m: float | None
    player_diagnostics: tuple[PlayerSpatialDiagnostics, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_SPATIAL_DIAGNOSTICS_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_frames", _nonnegative_int(self.total_frames, "total_frames"))
        for name in (
            "court_visible_frames",
            "geometry_valid_frames",
            "reprojection_rmse_sample_count",
            "projection_uncertainty_sample_count",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        for mapping_name in ("geometry_state_counts", "eligibility_state_counts"):
            mapping = dict(getattr(self, mapping_name))
            if any(not isinstance(key, str) or not key for key in mapping):
                raise P4SpatialDiagnosticsError(f"{mapping_name} keys must be non-empty strings")
            mapping = {key: _nonnegative_int(value, mapping_name) for key, value in mapping.items()}
            object.__setattr__(self, mapping_name, mapping)
        for mapping_name in ("geometry_state_fractions", "eligibility_state_fractions"):
            mapping = dict(getattr(self, mapping_name))
            normalized = {}
            for key, value in mapping.items():
                fraction = _nonnegative_number(value, mapping_name)
                if fraction > 1.0:
                    raise P4SpatialDiagnosticsError(f"{mapping_name} values must be within [0, 1]")
                normalized[str(key)] = fraction
            object.__setattr__(self, mapping_name, normalized)
        for name in (
            "mean_reprojection_rmse_px",
            "max_reprojection_rmse_px",
            "mean_projection_uncertainty_m",
            "max_projection_uncertainty_m",
        ):
            object.__setattr__(self, name, _optional_nonnegative_number(getattr(self, name), name))
        players = tuple(self.player_diagnostics)
        if not all(isinstance(item, PlayerSpatialDiagnostics) for item in players):
            raise P4SpatialDiagnosticsError("player_diagnostics contain invalid values")
        if len({item.track_id for item in players}) != len(players):
            raise P4SpatialDiagnosticsError("player_diagnostics track ids must be unique")
        object.__setattr__(self, "player_diagnostics", players)
        object.__setattr__(self, "provenance", dict(self.provenance))
        if self.contract_version != P4_SPATIAL_DIAGNOSTICS_CONTRACT_VERSION:
            raise P4SpatialDiagnosticsError(
                f"contract_version must be {P4_SPATIAL_DIAGNOSTICS_CONTRACT_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "geometry_state_counts": dict(self.geometry_state_counts),
            "geometry_state_fractions": dict(self.geometry_state_fractions),
            "eligibility_state_counts": dict(self.eligibility_state_counts),
            "eligibility_state_fractions": dict(self.eligibility_state_fractions),
            "court_visible_frames": self.court_visible_frames,
            "geometry_valid_frames": self.geometry_valid_frames,
            "reprojection_rmse_sample_count": self.reprojection_rmse_sample_count,
            "mean_reprojection_rmse_px": self.mean_reprojection_rmse_px,
            "max_reprojection_rmse_px": self.max_reprojection_rmse_px,
            "projection_uncertainty_sample_count": self.projection_uncertainty_sample_count,
            "mean_projection_uncertainty_m": self.mean_projection_uncertainty_m,
            "max_projection_uncertainty_m": self.max_projection_uncertainty_m,
            "player_diagnostics": [item.as_dict() for item in self.player_diagnostics],
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }


def _step_distances(points: Sequence[tuple[float, float]]) -> list[float]:
    return [
        hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(points, points[1:])
    ]


def build_spatial_diagnostics(
    snapshots: Sequence[SpatialSnapshot],
) -> SpatialDiagnosticsReport:
    values = tuple(snapshots)
    if not all(isinstance(item, SpatialSnapshot) for item in values):
        raise P4SpatialDiagnosticsError("snapshots must contain SpatialSnapshot values")
    ordered = tuple(sorted(values, key=lambda item: (item.timestamp_ms, item.frame_id)))
    keys = [(item.timestamp_ms, item.frame_id) for item in ordered]
    if len(keys) != len(set(keys)):
        raise P4SpatialDiagnosticsError("snapshots must have unique frame/timestamp pairs")

    geometry_counts = {state.value: 0 for state in GeometryQualityState}
    eligibility_counts = {state.value: 0 for state in SpatialEligibilityState}
    reprojection: list[float] = []
    uncertainty: list[float] = []
    court_visible = 0
    geometry_valid = 0
    per_track: dict[int, list[Any]] = {}

    for snapshot in ordered:
        geometry_counts[snapshot.geometry_quality.state.value] += 1
        eligibility_counts[snapshot.spatial_eligibility.state.value] += 1
        court_visible += int(snapshot.geometry.court_visible)
        geometry_valid += int(snapshot.geometry.geometry_valid)
        if snapshot.geometry_quality.reprojection_rmse_px is not None:
            reprojection.append(snapshot.geometry_quality.reprojection_rmse_px)
        if snapshot.geometry_quality.projection_uncertainty_m is not None:
            uncertainty.append(snapshot.geometry_quality.projection_uncertainty_m)
        for player in snapshot.players:
            per_track.setdefault(player.track_id, []).append(player)

    player_reports: list[PlayerSpatialDiagnostics] = []
    for track_id in sorted(per_track):
        players = per_track[track_id]
        raw = [item.court_position_m for item in players if item.court_position_m is not None]
        smoothed = [
            item.smoothed_position_m for item in players if item.smoothed_position_m is not None
        ]
        displacement = [
            hypot(
                item.court_position_m[0] - item.smoothed_position_m[0],
                item.court_position_m[1] - item.smoothed_position_m[1],
            )
            for item in players
            if item.court_position_m is not None and item.smoothed_position_m is not None
        ]
        velocity_count = sum(item.velocity_mps is not None for item in players)
        gap_count = sum(item.trajectory_state in _GAP_STATES for item in players)
        presence = len(players)
        player_reports.append(
            PlayerSpatialDiagnostics(
                track_id=track_id,
                frame_presence_count=presence,
                raw_metric_count=len(raw),
                smoothed_metric_count=len(smoothed),
                velocity_count=velocity_count,
                gap_state_count=gap_count,
                raw_metric_fraction=_fraction(len(raw), presence),
                smoothed_metric_fraction=_fraction(len(smoothed), presence),
                velocity_fraction=_fraction(velocity_count, presence),
                mean_smoothing_displacement_m=_mean(displacement),
                max_smoothing_displacement_m=max(displacement) if displacement else None,
                mean_raw_step_m=_mean(_step_distances(raw)),
                mean_smoothed_step_m=_mean(_step_distances(smoothed)),
                provenance={
                    "position_scope": "track_presence_in_materialized_snapshots",
                    "thresholds_applied": False,
                    "p2_consumed": False,
                    "p3_consumed": False,
                    "tactical_semantics": False,
                },
            )
        )

    total = len(ordered)
    return SpatialDiagnosticsReport(
        total_frames=total,
        geometry_state_counts=geometry_counts,
        geometry_state_fractions={
            key: _fraction(value, total) for key, value in geometry_counts.items()
        },
        eligibility_state_counts=eligibility_counts,
        eligibility_state_fractions={
            key: _fraction(value, total) for key, value in eligibility_counts.items()
        },
        court_visible_frames=court_visible,
        geometry_valid_frames=geometry_valid,
        reprojection_rmse_sample_count=len(reprojection),
        mean_reprojection_rmse_px=_mean(reprojection),
        max_reprojection_rmse_px=max(reprojection) if reprojection else None,
        projection_uncertainty_sample_count=len(uncertainty),
        mean_projection_uncertainty_m=_mean(uncertainty),
        max_projection_uncertainty_m=max(uncertainty) if uncertainty else None,
        player_diagnostics=tuple(player_reports),
        provenance={
            "builder": "core.p4_spatial_diagnostics.build_spatial_diagnostics",
            "thresholds_applied": False,
            "p2_consumed": False,
            "p3_consumed": False,
            "tactical_semantics": False,
        },
    )


def p4_spatial_diagnostics_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_SPATIAL_DIAGNOSTICS_CONTRACT_VERSION,
        "source_contract": "p4.spatial_snapshot.v1",
        "geometry_states": [state.value for state in GeometryQualityState],
        "eligibility_states": [state.value for state in SpatialEligibilityState],
        "thresholds_applied": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_semantics": False,
    }
