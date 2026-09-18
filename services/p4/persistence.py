from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping
from uuid import UUID, uuid5

from core.p4_spatial_contract import P4_SPATIAL_CONTRACT_VERSION, SpatialFrame


_P4_SPATIAL_ID_NAMESPACE = UUID("0c31cbaa-2118-4b89-87b3-a6df26779c9a")


def _nonempty(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be non-empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return number


def _json_object(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    normalized = dict(value)
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc
    return normalized


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class P4SpatialRunSpec:
    analysis_run_id: str
    game_id: str | None
    spatial_runtime_version: str
    configuration_version: str
    source_versions: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    contract_version: str = P4_SPATIAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_run_id", _nonempty(self.analysis_run_id, "analysis_run_id"))
        object.__setattr__(
            self,
            "game_id",
            None if self.game_id is None else _nonempty(self.game_id, "game_id"),
        )
        object.__setattr__(
            self,
            "spatial_runtime_version",
            _nonempty(self.spatial_runtime_version, "spatial_runtime_version"),
        )
        object.__setattr__(
            self,
            "configuration_version",
            _nonempty(self.configuration_version, "configuration_version"),
        )
        object.__setattr__(
            self, "source_versions", _json_object(self.source_versions, "source_versions")
        )
        object.__setattr__(
            self, "configuration", _json_object(self.configuration, "configuration")
        )
        if self.contract_version != P4_SPATIAL_CONTRACT_VERSION:
            raise ValueError(
                f"contract_version must be {P4_SPATIAL_CONTRACT_VERSION!r}"
            )

    def fingerprint(self) -> str:
        payload = {
            "analysis_run_id": self.analysis_run_id,
            "game_id": self.game_id,
            "contract_version": self.contract_version,
            "spatial_runtime_version": self.spatial_runtime_version,
            "configuration_version": self.configuration_version,
            "source_versions": self.source_versions,
            "configuration": self.configuration,
        }
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _stable_uuid(kind: str, *parts: object) -> str:
    identity = ":".join(str(part) for part in parts)
    return str(uuid5(_P4_SPATIAL_ID_NAMESPACE, f"{kind}:{identity}"))


def spatial_run_id(spec: P4SpatialRunSpec) -> str:
    return _stable_uuid("run", spec.analysis_run_id, spec.fingerprint())


def spatial_run_row(
    spec: P4SpatialRunSpec,
    *,
    workspace_id: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": spatial_run_id(spec),
        "workspace_id": _nonempty(workspace_id, "workspace_id"),
        "game_id": spec.game_id,
        "analysis_run_id": spec.analysis_run_id,
        "contract_version": spec.contract_version,
        "spatial_runtime_version": spec.spatial_runtime_version,
        "configuration_version": spec.configuration_version,
        "fingerprint": spec.fingerprint(),
        "source_versions": dict(spec.source_versions),
        "configuration": dict(spec.configuration),
        "provenance": _json_object(provenance, "provenance"),
    }


def spatial_snapshot_id(spatial_run_id_value: str, frame_id: int) -> str:
    return _stable_uuid(
        "snapshot",
        _nonempty(spatial_run_id_value, "spatial_run_id"),
        _nonnegative_int(frame_id, "frame_id"),
    )


def spatial_snapshot_row(
    frame: SpatialFrame,
    *,
    workspace_id: str,
    spatial_run_id_value: str,
    analysis_run_id: str,
    game_id: str | None,
    snapshot_kind: str = "materialized",
) -> dict[str, Any]:
    payload = frame.as_dict()
    return {
        "id": spatial_snapshot_id(spatial_run_id_value, frame.frame_id),
        "workspace_id": _nonempty(workspace_id, "workspace_id"),
        "game_id": None if game_id is None else _nonempty(game_id, "game_id"),
        "analysis_run_id": _nonempty(analysis_run_id, "analysis_run_id"),
        "spatial_run_id": _nonempty(spatial_run_id_value, "spatial_run_id"),
        "frame_id": frame.frame_id,
        "timestamp_ms": frame.timestamp_ms,
        "snapshot_kind": _nonempty(snapshot_kind, "snapshot_kind"),
        "contract_version": frame.contract_version,
        "source_runtime_version": frame.source_runtime_version,
        "geometry_valid": frame.geometry.geometry_valid,
        "court_visible": frame.geometry.court_visible,
        "player_count": len(frame.players),
        "payload": payload,
    }


def spatial_artifact_link_id(
    spatial_run_id_value: str,
    artifact_id: str,
    artifact_kind: str,
) -> str:
    return _stable_uuid(
        "artifact",
        _nonempty(spatial_run_id_value, "spatial_run_id"),
        _nonempty(artifact_id, "artifact_id"),
        _nonempty(artifact_kind, "artifact_kind"),
    )


def spatial_artifact_link_row(
    *,
    workspace_id: str,
    spatial_run_id_value: str,
    analysis_run_id: str,
    game_id: str | None,
    artifact_id: str,
    artifact_kind: str,
    content_format: str | None = None,
    content_contract_version: str = P4_SPATIAL_CONTRACT_VERSION,
    frame_start: int | None = None,
    frame_end: int | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    record_count: int | None = None,
    player_observation_count: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if content_contract_version != P4_SPATIAL_CONTRACT_VERSION:
        raise ValueError(
            f"content_contract_version must be {P4_SPATIAL_CONTRACT_VERSION!r}"
        )
    normalized_counts: dict[str, int | None] = {}
    for name, value in {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "record_count": record_count,
        "player_observation_count": player_observation_count,
    }.items():
        normalized_counts[name] = (
            None if value is None else _nonnegative_int(value, name)
        )
    if (
        normalized_counts["frame_start"] is not None
        and normalized_counts["frame_end"] is not None
        and normalized_counts["frame_end"] < normalized_counts["frame_start"]
    ):
        raise ValueError("frame_end must be >= frame_start")
    if (
        normalized_counts["start_ms"] is not None
        and normalized_counts["end_ms"] is not None
        and normalized_counts["end_ms"] < normalized_counts["start_ms"]
    ):
        raise ValueError("end_ms must be >= start_ms")
    return {
        "id": spatial_artifact_link_id(spatial_run_id_value, artifact_id, artifact_kind),
        "workspace_id": _nonempty(workspace_id, "workspace_id"),
        "game_id": None if game_id is None else _nonempty(game_id, "game_id"),
        "analysis_run_id": _nonempty(analysis_run_id, "analysis_run_id"),
        "spatial_run_id": _nonempty(spatial_run_id_value, "spatial_run_id"),
        "artifact_id": _nonempty(artifact_id, "artifact_id"),
        "artifact_kind": _nonempty(artifact_kind, "artifact_kind"),
        "content_contract_version": content_contract_version,
        "content_format": None if content_format is None else _nonempty(content_format, "content_format"),
        "frame_start": normalized_counts["frame_start"],
        "frame_end": normalized_counts["frame_end"],
        "start_ms": normalized_counts["start_ms"],
        "end_ms": normalized_counts["end_ms"],
        "record_count": normalized_counts["record_count"],
        "player_observation_count": normalized_counts["player_observation_count"],
        "metadata": _json_object(metadata, "metadata"),
    }


def _rpc_value(data: Any, *keys: str) -> Any:
    if isinstance(data, list):
        return _rpc_value(data[0], *keys) if data else None
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        if len(data) == 1:
            return next(iter(data.values()))
    return data


class P4SpatialPersistenceStore:
    """Supabase adapter for P1-only P4 Spatial Foundation persistence."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def start_run(
        self,
        *,
        workspace_id: str,
        spec: P4SpatialRunSpec,
        provenance: Mapping[str, Any] | None = None,
    ) -> str:
        row = spatial_run_row(spec, workspace_id=workspace_id, provenance=provenance)
        result = self.client.rpc(
            "start_p4_spatial_run",
            {
                "p_id": row["id"],
                "p_workspace_id": row["workspace_id"],
                "p_game_id": row["game_id"],
                "p_analysis_run_id": row["analysis_run_id"],
                "p_contract_version": row["contract_version"],
                "p_spatial_runtime_version": row["spatial_runtime_version"],
                "p_configuration_version": row["configuration_version"],
                "p_fingerprint": row["fingerprint"],
                "p_source_versions": row["source_versions"],
                "p_configuration": row["configuration"],
                "p_provenance": row["provenance"],
            },
        ).execute()
        run_id = _rpc_value(getattr(result, "data", None), "start_p4_spatial_run", "id")
        if not run_id:
            raise RuntimeError("start_p4_spatial_run returned no run id")
        return str(run_id)

    def persist_snapshots(
        self,
        *,
        workspace_id: str,
        spatial_run_id_value: str,
        analysis_run_id: str,
        game_id: str | None,
        frames: Iterable[SpatialFrame],
        snapshot_kind: str = "materialized",
    ) -> int:
        rows = [
            spatial_snapshot_row(
                frame,
                workspace_id=workspace_id,
                spatial_run_id_value=spatial_run_id_value,
                analysis_run_id=analysis_run_id,
                game_id=game_id,
                snapshot_kind=snapshot_kind,
            )
            for frame in frames
        ]
        if not rows:
            return 0
        self.client.table("p4_spatial_snapshots").upsert(
            rows,
            on_conflict="id",
        ).execute()
        return len(rows)

    def link_artifact(self, **kwargs: Any) -> str:
        row = spatial_artifact_link_row(**kwargs)
        self.client.table("p4_spatial_artifacts").upsert(
            row,
            on_conflict="id",
        ).execute()
        return str(row["id"])

    def complete_run(
        self,
        spatial_run_id_value: str,
        *,
        output_summary: Mapping[str, Any] | None = None,
    ) -> bool:
        result = self.client.rpc(
            "complete_p4_spatial_run",
            {
                "p_spatial_run_id": _nonempty(spatial_run_id_value, "spatial_run_id"),
                "p_output_summary": _json_object(output_summary, "output_summary"),
            },
        ).execute()
        return bool(_rpc_value(getattr(result, "data", None), "complete_p4_spatial_run"))

    def fail_run(self, spatial_run_id_value: str, *, error: str) -> bool:
        message = _nonempty(error, "error")[:2000]
        result = self.client.rpc(
            "fail_p4_spatial_run",
            {
                "p_spatial_run_id": _nonempty(spatial_run_id_value, "spatial_run_id"),
                "p_error": message,
            },
        ).execute()
        return bool(_rpc_value(getattr(result, "data", None), "fail_p4_spatial_run"))
