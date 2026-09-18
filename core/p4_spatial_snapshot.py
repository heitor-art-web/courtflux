from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
import json
from typing import Any, Mapping, Sequence

from core.p4_geometry_quality import (
    GeometryQualityObservation,
    P4_GEOMETRY_QUALITY_CONTRACT_VERSION,
)
from core.p4_kinematics import (
    KinematicsPointState,
    PlayerKinematics,
    P4_KINEMATICS_CONTRACT_VERSION,
)
from core.p4_motion_primitives import (
    MotionPrimitiveObservation,
    P4_MOTION_PRIMITIVE_CONTRACT_VERSION,
)
from core.p4_spatial_contract import (
    P4_SPATIAL_CONTRACT_VERSION,
    SpatialFrame,
    SpatialGeometryEvidence,
)
from core.p4_spatial_eligibility import (
    P4_SPATIAL_ELIGIBILITY_CONTRACT_VERSION,
    SpatialEligibilityObservation,
    SpatialEligibilityState,
)
from core.p4_spatial_relationships import (
    P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION,
    PairwiseSpatialRelationship,
    SpatialActorState,
    pairwise_relationships,
)
from core.p4_team_shape import (
    P4_TEAM_SHAPE_CONTRACT_VERSION,
    TeamShapeMetrics,
    compute_all_team_shapes,
)
from core.p4_trajectory import (
    P4_TRAJECTORY_CONTRACT_VERSION,
    PlayerTrajectory,
    TrajectoryPointState,
)

P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION = "p4.spatial_snapshot.v1"

_PLAYER_FIELDS = frozenset({
    "track_id", "local_team_id", "visible", "image_position_px",
    "court_position_m", "smoothed_position_m", "trajectory_state",
    "kinematics_state", "velocity_mps", "speed_mps",
    "observation_confidence", "position_confidence", "provenance",
})
_SNAPSHOT_FIELDS = frozenset({
    "frame_id", "timestamp_ms", "source_runtime_version",
    "source_spatial_contract_version", "geometry", "geometry_quality",
    "spatial_eligibility", "players", "relationships", "team_shapes",
    "motion_primitives", "provenance", "contract_version",
})


class P4SpatialSnapshotError(ValueError):
    pass


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise P4SpatialSnapshotError(f"{name} must be a non-negative integer")
    return int(value)


def _text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    if not result:
        raise P4SpatialSnapshotError(f"{name} must be non-empty when provided")
    return result


def _number(value: Any, name: str, *, probability: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise P4SpatialSnapshotError(f"{name} must be finite")
    result = float(value)
    if not isfinite(result) or (probability and not 0.0 <= result <= 1.0):
        raise P4SpatialSnapshotError(f"{name} is invalid")
    return result


def _point(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise P4SpatialSnapshotError(f"{name} must contain two coordinates")
    x = _number(value[0], name)
    y = _number(value[1], name)
    assert x is not None and y is not None
    return x, y


def _json(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise P4SpatialSnapshotError(f"{name} must be a JSON object")
    result = dict(value)
    try:
        json.dumps(result, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise P4SpatialSnapshotError(f"{name} must be JSON serializable") from exc
    return result


def _fields(payload: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if frozenset(payload) != expected:
        raise P4SpatialSnapshotError(f"{name} fields must match frozen v1 schema")


@dataclass(frozen=True, slots=True)
class SpatialSnapshotPlayer:
    track_id: int
    local_team_id: str | None
    visible: bool
    image_position_px: tuple[float, float] | None
    court_position_m: tuple[float, float] | None
    smoothed_position_m: tuple[float, float] | None
    trajectory_state: TrajectoryPointState | None
    kinematics_state: KinematicsPointState | None
    velocity_mps: tuple[float, float] | None
    speed_mps: float | None
    observation_confidence: float | None
    position_confidence: float | None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _int(self.track_id, "track_id"))
        object.__setattr__(self, "local_team_id", _text(self.local_team_id, "local_team_id"))
        if not isinstance(self.visible, bool):
            raise P4SpatialSnapshotError("visible must be boolean")
        for name in ("image_position_px", "court_position_m", "smoothed_position_m", "velocity_mps"):
            object.__setattr__(self, name, _point(getattr(self, name), name))
        if self.trajectory_state is not None and not isinstance(self.trajectory_state, TrajectoryPointState):
            try:
                object.__setattr__(self, "trajectory_state", TrajectoryPointState(str(self.trajectory_state)))
            except ValueError as exc:
                raise P4SpatialSnapshotError("unsupported trajectory_state") from exc
        if self.kinematics_state is not None and not isinstance(self.kinematics_state, KinematicsPointState):
            try:
                object.__setattr__(self, "kinematics_state", KinematicsPointState(str(self.kinematics_state)))
            except ValueError as exc:
                raise P4SpatialSnapshotError("unsupported kinematics_state") from exc
        speed = _number(self.speed_mps, "speed_mps")
        if speed is not None and speed < 0.0:
            raise P4SpatialSnapshotError("speed_mps must be >= 0")
        object.__setattr__(self, "speed_mps", speed)
        object.__setattr__(self, "observation_confidence", _number(
            self.observation_confidence, "observation_confidence", probability=True
        ))
        object.__setattr__(self, "position_confidence", _number(
            self.position_confidence, "position_confidence", probability=True
        ))
        object.__setattr__(self, "provenance", _json(self.provenance, "provenance"))
        if self.kinematics_state is KinematicsPointState.UNAVAILABLE and (
            self.velocity_mps is not None or self.speed_mps is not None
        ):
            raise P4SpatialSnapshotError("UNAVAILABLE kinematics cannot expose velocity/speed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "local_team_id": self.local_team_id,
            "visible": self.visible,
            "image_position_px": list(self.image_position_px) if self.image_position_px else None,
            "court_position_m": list(self.court_position_m) if self.court_position_m else None,
            "smoothed_position_m": list(self.smoothed_position_m) if self.smoothed_position_m else None,
            "trajectory_state": self.trajectory_state.value if self.trajectory_state else None,
            "kinematics_state": self.kinematics_state.value if self.kinematics_state else None,
            "velocity_mps": list(self.velocity_mps) if self.velocity_mps else None,
            "speed_mps": self.speed_mps,
            "observation_confidence": self.observation_confidence,
            "position_confidence": self.position_confidence,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialSnapshotPlayer":
        _fields(payload, _PLAYER_FIELDS, "SpatialSnapshotPlayer")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class SpatialSnapshot:
    frame_id: int
    timestamp_ms: int
    source_runtime_version: str | None
    source_spatial_contract_version: str
    geometry: SpatialGeometryEvidence
    geometry_quality: GeometryQualityObservation
    spatial_eligibility: SpatialEligibilityObservation
    players: tuple[SpatialSnapshotPlayer, ...]
    relationships: tuple[PairwiseSpatialRelationship, ...]
    team_shapes: tuple[TeamShapeMetrics, ...]
    motion_primitives: tuple[MotionPrimitiveObservation, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _int(self.frame_id, "frame_id"))
        object.__setattr__(self, "timestamp_ms", _int(self.timestamp_ms, "timestamp_ms"))
        object.__setattr__(self, "source_runtime_version", _text(
            self.source_runtime_version, "source_runtime_version"
        ))
        if self.source_spatial_contract_version != P4_SPATIAL_CONTRACT_VERSION:
            raise P4SpatialSnapshotError("unsupported source spatial contract")
        if not isinstance(self.geometry, SpatialGeometryEvidence):
            raise P4SpatialSnapshotError("geometry must be SpatialGeometryEvidence")
        if not isinstance(self.geometry_quality, GeometryQualityObservation):
            raise P4SpatialSnapshotError("geometry_quality must be GeometryQualityObservation")
        if not isinstance(self.spatial_eligibility, SpatialEligibilityObservation):
            raise P4SpatialSnapshotError("spatial_eligibility must be SpatialEligibilityObservation")
        if (
            self.geometry_quality.frame_id, self.geometry_quality.timestamp_ms
        ) != (self.frame_id, self.timestamp_ms) or (
            self.spatial_eligibility.frame_id, self.spatial_eligibility.timestamp_ms
        ) != (self.frame_id, self.timestamp_ms):
            raise P4SpatialSnapshotError("nested frame/time must match snapshot")
        if self.spatial_eligibility.geometry_quality_state is not self.geometry_quality.state:
            raise P4SpatialSnapshotError("eligibility geometry state must match geometry quality")

        players = tuple(self.players)
        if not all(isinstance(item, SpatialSnapshotPlayer) for item in players):
            raise P4SpatialSnapshotError("players contain invalid values")
        ids = [item.track_id for item in players]
        if len(ids) != len(set(ids)):
            raise P4SpatialSnapshotError("players must have unique track ids")
        object.__setattr__(self, "players", players)
        known = set(ids)

        relationships = tuple(self.relationships)
        for item in relationships:
            if not isinstance(item, PairwiseSpatialRelationship) or item.timestamp_ms != self.timestamp_ms:
                raise P4SpatialSnapshotError("relationship must match snapshot timestamp")
            if not {item.track_a_id, item.track_b_id}.issubset(known):
                raise P4SpatialSnapshotError("relationship references missing track")
        object.__setattr__(self, "relationships", relationships)

        shapes = tuple(self.team_shapes)
        for item in shapes:
            if not isinstance(item, TeamShapeMetrics) or item.timestamp_ms != self.timestamp_ms:
                raise P4SpatialSnapshotError("team shape must match snapshot timestamp")
            if not set(item.track_ids).issubset(known):
                raise P4SpatialSnapshotError("team shape references missing track")
        object.__setattr__(self, "team_shapes", shapes)

        primitives = tuple(self.motion_primitives)
        for item in primitives:
            if not isinstance(item, MotionPrimitiveObservation):
                raise P4SpatialSnapshotError("motion primitive has invalid type")
            if not item.start_ms <= self.timestamp_ms <= item.end_ms:
                raise P4SpatialSnapshotError("motion primitive is not active at snapshot time")
            if not set(item.track_ids).issubset(known):
                raise P4SpatialSnapshotError("motion primitive references missing track")
        object.__setattr__(self, "motion_primitives", primitives)

        if self.spatial_eligibility.state is SpatialEligibilityState.INVALID:
            if relationships or shapes or primitives:
                raise P4SpatialSnapshotError("INVALID snapshot cannot expose derived collections")
            if any(
                player.smoothed_position_m is not None
                or player.velocity_mps is not None
                or player.speed_mps is not None
                for player in players
            ):
                raise P4SpatialSnapshotError("INVALID snapshot cannot expose derived player truth")

        object.__setattr__(self, "provenance", _json(self.provenance, "provenance"))
        if self.contract_version != P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION:
            raise P4SpatialSnapshotError("unsupported snapshot contract")

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "source_runtime_version": self.source_runtime_version,
            "source_spatial_contract_version": self.source_spatial_contract_version,
            "geometry": self.geometry.as_dict(),
            "geometry_quality": self.geometry_quality.as_dict(),
            "spatial_eligibility": self.spatial_eligibility.as_dict(),
            "players": [item.as_dict() for item in self.players],
            "relationships": [item.as_dict() for item in self.relationships],
            "team_shapes": [item.as_dict() for item in self.team_shapes],
            "motion_primitives": [item.as_dict() for item in self.motion_primitives],
            "provenance": dict(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpatialSnapshot":
        _fields(payload, _SNAPSHOT_FIELDS, "SpatialSnapshot")
        data = dict(payload)
        data["geometry"] = SpatialGeometryEvidence.from_dict(data["geometry"])
        data["geometry_quality"] = GeometryQualityObservation.from_dict(data["geometry_quality"])
        data["spatial_eligibility"] = SpatialEligibilityObservation.from_dict(data["spatial_eligibility"])
        data["players"] = tuple(SpatialSnapshotPlayer.from_dict(item) for item in data["players"])
        data["relationships"] = tuple(PairwiseSpatialRelationship.from_dict(item) for item in data["relationships"])
        data["team_shapes"] = tuple(TeamShapeMetrics.from_dict(item) for item in data["team_shapes"])
        data["motion_primitives"] = tuple(MotionPrimitiveObservation.from_dict(item) for item in data["motion_primitives"])
        return cls(**data)


def p4_spatial_snapshot_manifest() -> dict[str, Any]:
    return {
        "contract_version": P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION,
        "source_contracts": {
            "spatial_frame": P4_SPATIAL_CONTRACT_VERSION,
            "geometry_quality": P4_GEOMETRY_QUALITY_CONTRACT_VERSION,
            "spatial_eligibility": P4_SPATIAL_ELIGIBILITY_CONTRACT_VERSION,
            "trajectory": P4_TRAJECTORY_CONTRACT_VERSION,
            "kinematics": P4_KINEMATICS_CONTRACT_VERSION,
            "relationships": P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION,
            "team_shape": P4_TEAM_SHAPE_CONTRACT_VERSION,
            "motion_primitives": P4_MOTION_PRIMITIVE_CONTRACT_VERSION,
        },
        "relationship_position_source": "smoothed_position_m",
        "invalid_emits_metric_truth": False,
        "attack_direction_inference": False,
        "p2_dependency": False,
        "p3_dependency": False,
        "tactical_semantics": False,
    }


def _point_at(points: Sequence[Any], frame_id: int, timestamp_ms: int) -> Any | None:
    matches = [p for p in points if p.frame_id == frame_id and p.timestamp_ms == timestamp_ms]
    if len(matches) > 1:
        raise P4SpatialSnapshotError("duplicate source point for frame/timestamp")
    return matches[0] if matches else None


def materialize_spatial_snapshot(
    frame: SpatialFrame,
    geometry_quality: GeometryQualityObservation,
    spatial_eligibility: SpatialEligibilityObservation,
    *,
    trajectories: Sequence[PlayerTrajectory] = (),
    kinematics: Sequence[PlayerKinematics] = (),
    motion_primitives: Sequence[MotionPrimitiveObservation] = (),
) -> SpatialSnapshot:
    if not isinstance(frame, SpatialFrame):
        raise P4SpatialSnapshotError("frame must be SpatialFrame")
    if (
        geometry_quality.frame_id, geometry_quality.timestamp_ms,
        spatial_eligibility.frame_id, spatial_eligibility.timestamp_ms,
    ) != (frame.frame_id, frame.timestamp_ms, frame.frame_id, frame.timestamp_ms):
        raise P4SpatialSnapshotError("frame, geometry quality and eligibility must match")

    trajectory_values = tuple(trajectories)
    kinematics_values = tuple(kinematics)
    trajectory_ids = [item.track_id for item in trajectory_values]
    kinematics_ids = [item.track_id for item in kinematics_values]
    if len(trajectory_ids) != len(set(trajectory_ids)) or len(kinematics_ids) != len(set(kinematics_ids)):
        raise P4SpatialSnapshotError("trajectory/kinematics track ids must be unique")
    trajectory_map = {}
    for item in trajectory_values:
        point = _point_at(item.points, frame.frame_id, frame.timestamp_ms)
        if point is not None:
            trajectory_map[item.track_id] = point
    kinematics_map = {}
    for item in kinematics_values:
        point = _point_at(item.points, frame.frame_id, frame.timestamp_ms)
        if point is not None:
            kinematics_map[item.track_id] = point
    if not set(kinematics_map).issubset(set(trajectory_map)):
        raise P4SpatialSnapshotError("kinematics requires trajectory lineage")

    observations = {item.track_id: item for item in frame.players}
    track_ids = sorted(set(observations) | set(trajectory_map) | set(kinematics_map))
    players: list[SpatialSnapshotPlayer] = []
    for track_id in track_ids:
        observation = observations.get(track_id)
        trajectory = trajectory_map.get(track_id)
        kine = kinematics_map.get(track_id)
        if kine is not None and trajectory is None:
            raise P4SpatialSnapshotError("kinematics point requires trajectory point")
        if kine is not None and trajectory is not None and kine.position_m != trajectory.smoothed_position_m:
            raise P4SpatialSnapshotError("kinematics position must match smoothed trajectory")
        players.append(SpatialSnapshotPlayer(
            track_id=track_id,
            local_team_id=observation.local_team_id if observation else None,
            visible=observation.visible if observation else False,
            image_position_px=observation.image_center_px if observation else None,
            court_position_m=observation.court_position_m if observation else None,
            smoothed_position_m=trajectory.smoothed_position_m if trajectory else None,
            trajectory_state=trajectory.state if trajectory else None,
            kinematics_state=kine.state if kine else None,
            velocity_mps=kine.velocity_mps if kine else None,
            speed_mps=kine.speed_mps if kine else None,
            observation_confidence=observation.observation_confidence if observation else None,
            position_confidence=observation.position_confidence if observation else None,
            provenance={
                "frame_observation_present": observation is not None,
                "trajectory_point_present": trajectory is not None,
                "kinematics_point_present": kine is not None,
                "p2_consumed": False, "p3_consumed": False, "tactical_semantics": False,
            },
        ))

    primitive_values = tuple(motion_primitives)
    if not all(isinstance(item, MotionPrimitiveObservation) for item in primitive_values):
        raise P4SpatialSnapshotError("motion_primitives contain invalid values")

    if spatial_eligibility.state is SpatialEligibilityState.INVALID:
        if any(p.smoothed_position_m is not None or p.velocity_mps is not None or p.speed_mps is not None for p in players):
            raise P4SpatialSnapshotError("INVALID eligibility cannot materialize derived player truth")
        relationships = ()
        shapes = ()
        active = ()
    else:
        actors = tuple(SpatialActorState(
            track_id=p.track_id, timestamp_ms=frame.timestamp_ms,
            position_m=p.smoothed_position_m, velocity_mps=p.velocity_mps,
            local_team_id=p.local_team_id,
            provenance={"position_source": "smoothed_position_m", "p2_consumed": False, "p3_consumed": False},
        ) for p in players if p.smoothed_position_m is not None)
        relationships = pairwise_relationships(actors)
        shapes = compute_all_team_shapes(actors)
        known = {p.track_id for p in players}
        active_list = []
        for primitive in primitive_values:
            if primitive.start_ms <= frame.timestamp_ms <= primitive.end_ms:
                if not set(primitive.track_ids).issubset(known):
                    raise P4SpatialSnapshotError("active primitive references missing track")
                active_list.append(primitive)
        active = tuple(active_list)

    return SpatialSnapshot(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        source_runtime_version=frame.source_runtime_version,
        source_spatial_contract_version=frame.contract_version,
        geometry=frame.geometry,
        geometry_quality=geometry_quality,
        spatial_eligibility=spatial_eligibility,
        players=tuple(players),
        relationships=relationships,
        team_shapes=shapes,
        motion_primitives=active,
        provenance={
            "materializer": "core.p4_spatial_snapshot.materialize_spatial_snapshot",
            "relationship_position_source": "smoothed_position_m",
            "p2_consumed": False, "p3_consumed": False, "tactical_semantics": False,
        },
    )


def materialize_spatial_snapshots(
    frames: Sequence[SpatialFrame],
    geometry_quality: Sequence[GeometryQualityObservation],
    spatial_eligibility: Sequence[SpatialEligibilityObservation],
    **kwargs: Any,
) -> tuple[SpatialSnapshot, ...]:
    if not (len(frames) == len(geometry_quality) == len(spatial_eligibility)):
        raise P4SpatialSnapshotError("parallel snapshot inputs must have identical length")
    return tuple(
        materialize_spatial_snapshot(frame, quality, eligibility, **kwargs)
        for frame, quality, eligibility in zip(frames, geometry_quality, spatial_eligibility)
    )
