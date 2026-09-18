from __future__ import annotations

from pathlib import Path
from typing import Any

from core.p4_geometry_quality import (
    GeometryQualityObservation,
    GeometryQualityPolicy,
    GeometryQualityState,
)
from core.p4_spatial_contract import (
    P4_SPATIAL_CONTRACT_VERSION,
    SpatialFrame,
    SpatialGeometryEvidence,
    SpatialPlayerObservation,
)
from core.p4_spatial_eligibility import (
    SpatialEligibilityObservation,
    SpatialEligibilityPolicy,
    SpatialEligibilityReason,
    SpatialEligibilityState,
)
from core.p4_spatial_snapshot import (
    P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION,
    SpatialSnapshot,
    materialize_spatial_snapshot,
)
from services.p4.persistence import (
    P4SpatialPersistenceStore,
    P4SpatialRunSpec,
    spatial_artifact_link_row,
    spatial_run_id,
    spatial_run_row,
    spatial_snapshot_row,
)


MIGRATION = (
    Path(__file__).parents[1]
    / "supabase"
    / "migrations"
    / "20260918133500_p4_spatial_persistence.sql"
)


class _Result:
    def __init__(self, data: Any = None) -> None:
        self.data = data


class _Query:
    def __init__(self, client: "_Client", operation: str, payload: Any = None) -> None:
        self.client = client
        self.operation = operation
        self.payload = payload
        self.on_conflict: str | None = None

    def execute(self) -> _Result:
        self.client.calls.append(
            {
                "operation": self.operation,
                "payload": self.payload,
                "on_conflict": self.on_conflict,
            }
        )
        if self.operation == "rpc:start_p4_spatial_run":
            return _Result(self.client.start_run_id)
        if self.operation in {
            "rpc:complete_p4_spatial_run",
            "rpc:fail_p4_spatial_run",
        }:
            return _Result(True)
        return _Result([])


class _Table:
    def __init__(self, client: "_Client", name: str) -> None:
        self.client = client
        self.name = name

    def upsert(self, payload: Any, *, on_conflict: str) -> _Query:
        query = _Query(self.client, f"upsert:{self.name}", payload)
        query.on_conflict = on_conflict
        return query


class _Client:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.start_run_id = "00000000-0000-0000-0000-000000000111"

    def rpc(self, name: str, payload: Any) -> _Query:
        return _Query(self, f"rpc:{name}", payload)

    def table(self, name: str) -> _Table:
        return _Table(self, name)


def _spec(**overrides: Any) -> P4SpatialRunSpec:
    payload = {
        "analysis_run_id": "00000000-0000-0000-0000-000000000010",
        "game_id": "00000000-0000-0000-0000-000000000020",
        "spatial_runtime_version": "p4-spatial-runtime-v1",
        "configuration_version": "spatial-config-v1",
        "source_versions": {
            "player_state": "player_state_v1",
            "court_geometry": "court_geometry_v1",
        },
        "configuration": {
            "snapshot_stride_frames": 30,
            "dense_artifact_format": "jsonl",
        },
    }
    payload.update(overrides)
    return P4SpatialRunSpec(**payload)


def _frame() -> SpatialFrame:
    return SpatialFrame(
        frame_id=42,
        timestamp_ms=1400,
        geometry=SpatialGeometryEvidence(
            geometry_valid=True,
            court_visible=True,
            homography_source="direct_keypoints",
            reprojection_rmse_px=2.5,
            homography_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            court_coordinate_system="p4.fiba_28x15_xy.v1",
            provenance={"source": "test"},
        ),
        players=(
            SpatialPlayerObservation(
                track_id=7,
                local_team_id="team_local_a",
                local_team_confidence=0.9,
                image_center_px=(100.0, 200.0),
                court_position_m=(8.0, 4.0),
                observation_confidence=0.8,
                observation_uncertainty=0.2,
                position_confidence=0.8,
                visible=True,
                source="tracked_player_detection",
                provenance={"fixture": True},
            ),
        ),
        source_runtime_version="p1-spatial-source-v1",
        provenance={"fixture": "p4"},
    )


def _snapshot() -> SpatialSnapshot:
    frame = _frame()
    quality = GeometryQualityObservation(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        state=GeometryQualityState.DIRECT_CALIBRATION,
        reason="direct calibration",
        homography_source="direct_keypoints",
        last_direct_calibration_ms=frame.timestamp_ms,
        homography_age_frames=0,
        anchor_count=8,
        inlier_count=8,
        reprojection_rmse_px=2.5,
        projection_uncertainty_m=None,
        policy=GeometryQualityPolicy(fresh_hold_max_age_frames=3),
        provenance={},
    )
    eligibility = SpatialEligibilityObservation(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        state=SpatialEligibilityState.ELIGIBLE,
        reasons=(SpatialEligibilityReason.ELIGIBLE,),
        geometry_quality_state=quality.state,
        visible_player_count=1,
        metric_player_count=1,
        missing_metric_player_count=0,
        out_of_bounds_metric_player_count=0,
        metric_player_fraction=1.0,
        policy=SpatialEligibilityPolicy(
            min_metric_player_count=1,
            min_metric_player_fraction=1.0,
            court_extent_tolerance_m=0.1,
            allow_aging_geometry=True,
            require_court_visible=True,
        ),
        provenance={},
    )
    return materialize_spatial_snapshot(frame, quality, eligibility)


def test_run_fingerprint_and_ids_are_deterministic() -> None:
    spec_a = _spec(configuration={"b": 2, "a": 1})
    spec_b = _spec(configuration={"a": 1, "b": 2})
    assert spec_a.fingerprint() == spec_b.fingerprint()
    assert spatial_run_id(spec_a) == spatial_run_id(spec_b)

    changed = _spec(configuration={"a": 1, "b": 3})
    assert changed.fingerprint() != spec_a.fingerprint()
    assert spatial_run_id(changed) != spatial_run_id(spec_a)


def test_rows_preserve_source_and_snapshot_contract_lineage() -> None:
    spec = _spec()
    row = spatial_run_row(
        spec,
        workspace_id="00000000-0000-0000-0000-000000000001",
        provenance={"worker": "test"},
    )
    assert row["contract_version"] == P4_SPATIAL_CONTRACT_VERSION
    assert row["analysis_run_id"] == spec.analysis_run_id

    materialized = _snapshot()
    snapshot = spatial_snapshot_row(
        materialized,
        workspace_id=row["workspace_id"],
        spatial_run_id_value=row["id"],
        analysis_run_id=spec.analysis_run_id,
        game_id=spec.game_id,
        snapshot_kind="diagnostic",
    )
    assert snapshot["payload"] == materialized.as_dict()
    assert snapshot["contract_version"] == P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION
    assert (
        snapshot["payload"]["source_spatial_contract_version"]
        == P4_SPATIAL_CONTRACT_VERSION
    )
    assert snapshot["player_count"] == 1
    assert snapshot["geometry_valid"] is True

    forbidden = {
        "possession_id",
        "game_player_id",
        "p3_derivation_id",
        "review_item_id",
        "clip_id",
        "tactical_action",
    }
    assert forbidden.isdisjoint(snapshot["payload"])

    artifact = spatial_artifact_link_row(
        workspace_id=row["workspace_id"],
        spatial_run_id_value=row["id"],
        analysis_run_id=spec.analysis_run_id,
        game_id=spec.game_id,
        artifact_id="00000000-0000-0000-0000-000000000099",
        artifact_kind="dense_materialized_timeline",
        content_format="jsonl",
        content_contract_version=P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION,
        frame_start=0,
        frame_end=42,
        start_ms=0,
        end_ms=1400,
        record_count=43,
        player_observation_count=380,
        metadata={"compression": "none"},
    )
    assert (
        artifact["content_contract_version"]
        == P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION
    )
    assert artifact["record_count"] == 43


def test_store_uses_service_owned_rpcs_and_idempotent_upserts() -> None:
    client = _Client()
    store = P4SpatialPersistenceStore(client)
    spec = _spec()

    run_id = store.start_run(
        workspace_id="00000000-0000-0000-0000-000000000001",
        spec=spec,
        provenance={"worker": "fixture"},
    )
    assert run_id == client.start_run_id
    start = client.calls[0]
    assert start["operation"] == "rpc:start_p4_spatial_run"
    assert start["payload"]["p_fingerprint"] == spec.fingerprint()

    count = store.persist_snapshots(
        workspace_id="00000000-0000-0000-0000-000000000001",
        spatial_run_id_value=run_id,
        analysis_run_id=spec.analysis_run_id,
        game_id=spec.game_id,
        snapshots=[_snapshot()],
        snapshot_kind="diagnostic",
    )
    assert count == 1
    assert client.calls[1]["operation"] == "upsert:p4_spatial_snapshots"
    assert client.calls[1]["on_conflict"] == "id"
    assert (
        client.calls[1]["payload"][0]["contract_version"]
        == P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION
    )

    link_id = store.link_artifact(
        workspace_id="00000000-0000-0000-0000-000000000001",
        spatial_run_id_value=run_id,
        analysis_run_id=spec.analysis_run_id,
        game_id=spec.game_id,
        artifact_id="00000000-0000-0000-0000-000000000099",
        artifact_kind="dense_timeline",
        content_format="jsonl",
    )
    assert link_id
    assert client.calls[2]["operation"] == "upsert:p4_spatial_artifacts"

    assert store.complete_run(run_id, output_summary={"frames": 1}) is True
    assert client.calls[3]["operation"] == "rpc:complete_p4_spatial_run"
    assert store.fail_run(run_id, error="fixture failure") is True
    assert client.calls[4]["operation"] == "rpc:fail_p4_spatial_run"


def test_migration_is_p1_platform_only_and_snapshot_fail_closed() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    for required in (
        "create table if not exists public.p4_spatial_runs",
        "create table if not exists public.p4_spatial_snapshots",
        "create table if not exists public.p4_spatial_artifacts",
        "references public.analysis_runs",
        "references public.analysis_artifacts",
        "private.is_workspace_member(workspace_id)",
        "enable row level security",
        "grant select on public.p4_spatial_runs to authenticated",
        "grant select, insert, update, delete on public.p4_spatial_runs to service_role",
        "start_p4_spatial_run",
        "complete_p4_spatial_run",
        "fail_p4_spatial_run",
        "p4_spatial_runs_one_current_idx",
        "check (contract_version = 'p4.spatial_snapshot.v1')",
        "source_spatial_contract_version",
        "snapshot geometry_quality must match indexed frame/time",
        "snapshot spatial_eligibility must match indexed frame/time",
        "invalid spatial snapshot cannot persist derived metric truth",
        "'p4.spatial_frame.v1'",
        "'p4.spatial_snapshot.v1'",
    ):
        assert required in sql

    for forbidden in (
        "analysis_derivations",
        "p3_",
        "p2_",
        "possession_id",
        "game_player_id",
        "review_item_id",
        "clip_id",
    ):
        assert forbidden not in sql

    assert sql.count("enable row level security") == 3
    assert "grant insert on public.p4_spatial_runs to authenticated" not in sql
    assert "grant update on public.p4_spatial_runs to authenticated" not in sql
    assert "security definer" not in sql


def test_invalid_artifact_ranges_and_contracts_are_rejected_before_write() -> None:
    spec = _spec()
    run_id = spatial_run_id(spec)
    try:
        spatial_artifact_link_row(
            workspace_id="00000000-0000-0000-0000-000000000001",
            spatial_run_id_value=run_id,
            analysis_run_id=spec.analysis_run_id,
            game_id=spec.game_id,
            artifact_id="00000000-0000-0000-0000-000000000099",
            artifact_kind="dense_timeline",
            frame_start=20,
            frame_end=10,
        )
    except ValueError as exc:
        assert "frame_end" in str(exc)
    else:
        raise AssertionError("reversed artifact frame ranges must fail")

    try:
        spatial_artifact_link_row(
            workspace_id="00000000-0000-0000-0000-000000000001",
            spatial_run_id_value=run_id,
            analysis_run_id=spec.analysis_run_id,
            game_id=spec.game_id,
            artifact_id="00000000-0000-0000-0000-000000000099",
            artifact_kind="dense_timeline",
            content_contract_version="p4.unknown.v1",
        )
    except ValueError as exc:
        assert "content_contract_version" in str(exc)
    else:
        raise AssertionError("unsupported artifact contract must fail")
