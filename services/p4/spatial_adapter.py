from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.p4_spatial_contract import (
    P4_UPSTREAM_COURT_COORDINATE_SYSTEM,
    SpatialFrame,
    SpatialGeometryEvidence,
    SpatialPlayerObservation,
)


def _optional_local_team(
    track_id: int,
    local_team_by_track_id: Mapping[int, str | tuple[str, float] | None] | None,
) -> tuple[str | None, float | None]:
    if not local_team_by_track_id or track_id not in local_team_by_track_id:
        return None, None
    value = local_team_by_track_id[track_id]
    if value is None:
        return None, None
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("local team tuple must be (team_id, confidence)")
        return str(value[0]), float(value[1])
    return str(value), None


def spatial_frame_from_p1(
    frame_state: Any,
    *,
    source_runtime_version: str | None = None,
    local_team_by_track_id: Mapping[int, str | tuple[str, float] | None] | None = None,
) -> SpatialFrame:
    """Adapt canonical P1 state into the P4 spatial v1 contract.

    The adapter reads only frame timing/geometry, ``court_state`` and
    ``player_states``. P2/P3/tactical fields on the input object are intentionally
    ignored so the spatial foundation can be reproduced without basketball truth.
    """

    frame_id = int(getattr(frame_state, "frame_id"))
    timestamp_sec = float(getattr(frame_state, "timestamp"))
    if timestamp_sec < 0.0:
        raise ValueError("frame timestamp must be non-negative")
    timestamp_ms = int(round(timestamp_sec * 1000.0))

    court_state = getattr(frame_state, "court_state", None)
    homography_matrix = getattr(frame_state, "homography_matrix", None)
    court_polygon = (
        getattr(court_state, "court_polygon_px", None)
        if court_state is not None
        else getattr(frame_state, "court_polygon", None)
    )
    geometry = SpatialGeometryEvidence(
        geometry_valid=bool(getattr(court_state, "geometry_valid", False)),
        court_visible=bool(getattr(court_state, "court_visible", False)),
        homography_source=getattr(court_state, "homography_source", None),
        reprojection_rmse_px=getattr(court_state, "reprojection_rmse_px", None),
        homography_matrix=homography_matrix,
        court_polygon_px=court_polygon,
        court_coordinate_system=P4_UPSTREAM_COURT_COORDINATE_SYSTEM,
        projection_uncertainty_m=None,
        homography_age_frames=None,
        provenance=(
            dict(getattr(court_state, "provenance", {}) or {})
            if court_state is not None
            else {}
        ),
    )

    players: list[SpatialPlayerObservation] = []
    for player in list(getattr(frame_state, "player_states", []) or []):
        player_frame_id = int(getattr(player, "frame_id"))
        if player_frame_id != frame_id:
            raise ValueError(
                f"PlayerState frame_id {player_frame_id} does not match FrameState {frame_id}"
            )
        track_id = int(getattr(player, "track_id"))
        local_team_id, local_team_confidence = _optional_local_team(
            track_id, local_team_by_track_id
        )
        players.append(
            SpatialPlayerObservation(
                track_id=track_id,
                local_team_id=local_team_id,
                local_team_confidence=local_team_confidence,
                image_center_px=getattr(player, "center_px"),
                court_position_m=getattr(player, "court_position_m", None),
                observation_confidence=getattr(player, "confidence"),
                observation_uncertainty=getattr(player, "uncertainty"),
                position_confidence=None,
                visible=bool(getattr(player, "visible")),
                source=str(getattr(player, "source")),
                provenance=dict(getattr(player, "provenance", {}) or {}),
            )
        )

    return SpatialFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        geometry=geometry,
        players=tuple(players),
        source_runtime_version=source_runtime_version,
        provenance={
            "adapter": "services.p4.spatial_adapter.spatial_frame_from_p1",
            "source_layer": "P1",
            "p2_consumed": False,
            "p3_consumed": False,
            "tactical_semantics": False,
        },
    )
