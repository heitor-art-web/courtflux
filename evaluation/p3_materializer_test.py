from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from core.p3_contract import P3SourceEvent
from services.p3.artifact_store import StoredClipArtifact
from services.p3.clip_generator import PhysicalClipArtifact
from services.p3.materializer import P3Materializer
from services.p3.persistence import clip_plan_id


class _Persistence:
    def __init__(
        self,
        clip_states: dict[str, dict[str, Any]] | None = None,
        *,
        acquire: bool = True,
    ) -> None:
        self.clip_states = dict(clip_states or {})
        self.acquire = acquire
        self.persisted_events: list[P3SourceEvent] = []
        self.snapshot = None
        self.generating: list[str] = []
        self.ready: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []
        self.summary: dict[str, Any] | None = None
        self.spec = None

    def start_derivation(self, *, workspace_id: str, spec: Any, provenance: dict[str, Any]) -> str:
        assert workspace_id == "workspace-1"
        self.spec = spec
        assert provenance["materializer_version"] == "p3.things_to_review.v1"
        return "derivation-1"

    def persist_source_events(
        self,
        *,
        workspace_id: str,
        derivation_id: str,
        events: list[P3SourceEvent],
    ) -> int:
        assert workspace_id == "workspace-1"
        assert derivation_id == "derivation-1"
        self.persisted_events = list(events)
        return len(self.persisted_events)

    def persist_review_snapshot(
        self,
        *,
        workspace_id: str,
        derivation_id: str,
        snapshot: Any,
    ) -> dict[str, int]:
        assert workspace_id == "workspace-1"
        assert derivation_id == "derivation-1"
        self.snapshot = snapshot
        return {"clips": snapshot.clip_plannable_count, "review_items": snapshot.total_items}

    def get_clip_states(self, clip_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {
            clip_id: self.clip_states[clip_id]
            for clip_id in clip_ids
            if clip_id in self.clip_states
        }

    def mark_clip_generating(self, clip_id: str) -> bool:
        self.generating.append(clip_id)
        return self.acquire

    def mark_clip_ready(self, clip_id: str, *, artifact_id: str) -> bool:
        self.ready.append((clip_id, artifact_id))
        return True

    def mark_clip_failed(self, clip_id: str, *, error: str) -> bool:
        self.failed.append((clip_id, error))
        return True

    def set_output_summary(self, derivation_id: str, summary: dict[str, Any]) -> None:
        assert derivation_id == "derivation-1"
        self.summary = dict(summary)


class _ArtifactStore:
    bucket = "analysis-artifacts"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def persist_clip(
        self,
        *,
        workspace_id: str,
        analysis_run_id: str,
        clip_id: str,
        artifact: PhysicalClipArtifact,
    ) -> StoredClipArtifact:
        self.calls.append(
            {
                "workspace_id": workspace_id,
                "analysis_run_id": analysis_run_id,
                "clip_id": clip_id,
                "artifact": artifact,
            }
        )
        return StoredClipArtifact(
            artifact_id=f"artifact-{clip_id}",
            bucket=self.bucket,
            object_path=f"{workspace_id}/{analysis_run_id}/p3/clips/{clip_id}/hash.mp4",
            content_type="video/mp4",
            size_bytes=artifact.size_bytes,
            checksum_sha256=artifact.checksum_sha256,
        )


def _event() -> P3SourceEvent:
    return P3SourceEvent(
        source_event_id="shot-1",
        game_id="game-1",
        analysis_run_id="run-1",
        event_type="shot_attempt",
        decision="candidate",
        confidence=0.72,
        uncertainty=0.28,
        frame_start=300,
        frame_end=305,
        start_ms=10_000,
        end_ms=10_500,
        actor_track_id=7,
        evidence=("shot_flight",),
        provenance={"version": "shot_tracker_v1"},
        attributes={"possession_id": "possession-7"},
        source_runtime_version="p2_engine_v1",
    )


def _generator_calls(calls: list[dict[str, Any]]):
    def _generate(request: Any, *, source_path: Path, output_path: Path, config: Any):
        calls.append(
            {
                "request": request,
                "source_path": Path(source_path),
                "output_path": Path(output_path),
                "config": config,
            }
        )
        Path(output_path).write_bytes(b"physical-clip")
        checksum = sha256(Path(output_path).read_bytes()).hexdigest()
        return PhysicalClipArtifact(
            path=str(output_path),
            size_bytes=Path(output_path).stat().st_size,
            checksum_sha256=checksum,
            content_type="video/mp4",
            game_id=request.game_id,
            analysis_run_id=request.analysis_run_id,
            source_video_ref=request.source_video_ref,
            review_type=request.review_type,
            source_start_ms=request.source_start_ms,
            source_end_ms=request.source_end_ms,
            clip_start_ms=request.clip_start_ms,
            clip_end_ms=request.clip_end_ms,
            source_event_ids=tuple(request.source_event_ids),
            source_possession_id=request.source_possession_id,
            game_player_ids=tuple(request.game_player_ids),
            context_policy_version=request.context_policy_version,
            request_contract_version=request.contract_version,
            request_fingerprint=request.fingerprint(),
            generation_version=config.version,
        )

    return _generate


def test_materializer_persists_durable_clip_lineage_and_metrics() -> None:
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "game.mp4"
        source.write_bytes(b"source")
        persistence = _Persistence()
        artifacts = _ArtifactStore()
        generator_calls: list[dict[str, Any]] = []
        materializer = P3Materializer(
            persistence=persistence,
            artifact_store=artifacts,
            clip_generator=_generator_calls(generator_calls),
            duration_probe=lambda _path: 60_000,
        )

        result = materializer.materialize(
            [_event()],
            workspace_id="workspace-1",
            game_id="game-1",
            analysis_run_id="run-1",
            source_video_path=source,
            output_dir=Path(tmp) / "clips",
        )

        assert result.review_item_count == 1
        assert result.clip_planned_count == 1
        assert result.clip_ready_count == 1
        assert result.clip_failed_count == 0
        assert result.clip_reused_count == 0
        assert result.clip_generation_attempted_count == 1
        assert result.clip_generation_succeeded_count == 1
        assert result.clip_generation_deferred_count == 0
        assert result.clip_ready_rate == 1.0
        request = persistence.snapshot.items[0].clip_request
        assert request.source_video_ref == "analysis-run:run-1"
        assert request.source_possession_id == "possession-7"
        assert request.clip_end_ms <= 60_000
        assert persistence.spec.source_versions["source_video_ref"] == "analysis-run:run-1"
        assert persistence.spec.configuration["source_duration_ms"] == 60_000
        assert persistence.summary["clip_request_contract_version"] == "p3.clip_request.v1"
        assert persistence.summary["clip_storage_contract_version"] == "p3.clip_storage.v1"
        assert not generator_calls[0]["output_path"].exists()


def test_materializer_reuses_already_ready_durable_clip() -> None:
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "game.mp4"
        source.write_bytes(b"source")
        first = _Persistence()
        first_materializer = P3Materializer(
            persistence=first,
            artifact_store=_ArtifactStore(),
            clip_generator=_generator_calls([]),
            duration_probe=lambda _path: 60_000,
        )
        first_materializer.materialize(
            [_event()],
            workspace_id="workspace-1",
            game_id="game-1",
            analysis_run_id="run-1",
            source_video_path=source,
            output_dir=Path(tmp) / "first",
        )
        request = first.snapshot.items[0].clip_request
        clip_id = clip_plan_id("derivation-1", request)

        persistence = _Persistence(
            {
                clip_id: {
                    "status": "ready",
                    "artifact_id": "artifact-existing",
                    "error": None,
                }
            }
        )
        artifacts = _ArtifactStore()
        generator_calls: list[dict[str, Any]] = []
        result = P3Materializer(
            persistence=persistence,
            artifact_store=artifacts,
            clip_generator=_generator_calls(generator_calls),
            duration_probe=lambda _path: 60_000,
        ).materialize(
            [_event()],
            workspace_id="workspace-1",
            game_id="game-1",
            analysis_run_id="run-1",
            source_video_path=source,
            output_dir=Path(tmp) / "second",
        )

        assert result.clip_ready_count == 1
        assert result.clip_reused_count == 1
        assert result.clip_generation_attempted_count == 0
        assert generator_calls == []
        assert artifacts.calls == []
        assert persistence.generating == []


def test_materializer_defers_when_generation_lease_is_held_elsewhere() -> None:
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "game.mp4"
        source.write_bytes(b"source")
        persistence = _Persistence(acquire=False)
        calls: list[dict[str, Any]] = []
        result = P3Materializer(
            persistence=persistence,
            artifact_store=_ArtifactStore(),
            clip_generator=_generator_calls(calls),
            duration_probe=lambda _path: 60_000,
        ).materialize(
            [_event()],
            workspace_id="workspace-1",
            game_id="game-1",
            analysis_run_id="run-1",
            source_video_path=source,
            output_dir=Path(tmp) / "clips",
        )
        assert result.clip_generation_deferred_count == 1
        assert result.clip_generation_attempted_count == 0
        assert result.clip_ready_count == 0
        assert calls == []


def test_materializer_records_clip_failure_without_losing_review_item() -> None:
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "game.mp4"
        source.write_bytes(b"source")
        persistence = _Persistence()

        def fail_generator(*_args: Any, **_kwargs: Any):
            raise RuntimeError("ffmpeg unavailable")

        result = P3Materializer(
            persistence=persistence,
            artifact_store=_ArtifactStore(),
            clip_generator=fail_generator,
            duration_probe=lambda _path: 60_000,
        ).materialize(
            [_event()],
            workspace_id="workspace-1",
            game_id="game-1",
            analysis_run_id="run-1",
            source_video_path=source,
            output_dir=Path(tmp) / "clips",
        )

        assert result.review_item_count == 1
        assert result.clip_failed_count == 1
        assert result.clip_ready_count == 0
        assert result.clip_generation_attempted_count == 1
        assert result.clip_generation_succeeded_count == 0
        assert "ffmpeg unavailable" in result.errors[0]
        assert len(persistence.failed) == 1
        assert persistence.snapshot.total_items == 1


if __name__ == "__main__":
    test_materializer_persists_durable_clip_lineage_and_metrics()
    test_materializer_reuses_already_ready_durable_clip()
    test_materializer_defers_when_generation_lease_is_held_elsewhere()
    test_materializer_records_clip_failure_without_losing_review_item()
    print("p3_materializer_test: PASS")
