from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from core.p3_contract import P3DerivationSpec, P3SourceEvent
from services.p3.artifact_store import P3ArtifactStore, StoredClipArtifact
from services.p3.clip_generator import (
    PhysicalClipArtifact,
    PhysicalClipConfig,
    generate_physical_clip,
)
from services.p3.p2_adapter import adapt_p2_event_rows
from services.p3.persistence import P3PersistenceStore, clip_plan_id
from services.p3.review_builder import ReviewBuildPolicy, build_things_to_review


P3_THINGS_TO_REVIEW_DERIVATION_VERSION = "p3.things_to_review.v1"


@dataclass(slots=True)
class P3MaterializationResult:
    derivation_id: str
    source_event_count: int
    review_item_count: int
    clip_planned_count: int
    clip_ready_count: int
    clip_failed_count: int
    clip_reused_count: int
    counts_by_type: dict[str, int]
    counts_by_support_state: dict[str, int]
    excluded_by_policy: dict[str, int]
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "derivation_id": self.derivation_id,
            "source_event_count": self.source_event_count,
            "review_item_count": self.review_item_count,
            "clip_planned_count": self.clip_planned_count,
            "clip_ready_count": self.clip_ready_count,
            "clip_failed_count": self.clip_failed_count,
            "clip_reused_count": self.clip_reused_count,
            "counts_by_type": dict(self.counts_by_type),
            "counts_by_support_state": dict(self.counts_by_support_state),
            "excluded_by_policy": dict(self.excluded_by_policy),
            "errors": list(self.errors),
        }


def _canonical_source_fingerprint(events: Iterable[P3SourceEvent]) -> str:
    payload = [
        event.as_dict()
        for event in sorted(
            events,
            key=lambda item: (
                item.frame_start,
                item.frame_end,
                item.source_event_id,
            ),
        )
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_p2_event_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load the canonical P2 JSONL export without changing event semantics."""

    source = Path(path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"P2 event JSONL does not exist: {source}")

    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid P2 event JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"P2 event JSONL line {line_number} must contain an object"
                )
            rows.append(row)
    return rows


class P3Materializer:
    """Materialize Things to Review and durable physical evidence from P3 source events.

    The source-event truth is immutable. Re-running with the same source events and
    versioned policy reuses the same derivation and already-ready clip artifacts.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        persistence: P3PersistenceStore | Any | None = None,
        artifact_store: P3ArtifactStore | Any | None = None,
        clip_generator: Callable[..., PhysicalClipArtifact] = generate_physical_clip,
    ) -> None:
        if persistence is None:
            if client is None:
                raise ValueError("client or persistence is required")
            persistence = P3PersistenceStore(client)
        if artifact_store is None:
            if client is None:
                raise ValueError("client or artifact_store is required")
            artifact_store = P3ArtifactStore(client)
        self.persistence = persistence
        self.artifact_store = artifact_store
        self.clip_generator = clip_generator

    def materialize(
        self,
        events: Iterable[P3SourceEvent],
        *,
        workspace_id: str,
        game_id: str,
        analysis_run_id: str,
        source_video_path: str | Path,
        output_dir: str | Path,
        policy: ReviewBuildPolicy | None = None,
        clip_config: PhysicalClipConfig | None = None,
        cleanup_local_after_upload: bool = True,
        provenance: dict[str, Any] | None = None,
    ) -> P3MaterializationResult:
        rows = list(events)
        for event in rows:
            if event.game_id != str(game_id):
                raise ValueError("P3 source event game_id does not match materialization game")
            if event.analysis_run_id != str(analysis_run_id):
                raise ValueError(
                    "P3 source event analysis_run_id does not match materialization run"
                )

        policy = policy or ReviewBuildPolicy()
        clip_config = clip_config or PhysicalClipConfig()
        source_fingerprint = _canonical_source_fingerprint(rows)
        runtime_versions = sorted(
            {
                str(event.source_runtime_version)
                for event in rows
                if event.source_runtime_version
            }
        )
        contract_versions = sorted({event.source_contract_version for event in rows})

        spec = P3DerivationSpec(
            game_id=str(game_id),
            analysis_run_id=str(analysis_run_id),
            derivation_key="things_to_review",
            derivation_version=P3_THINGS_TO_REVIEW_DERIVATION_VERSION,
            configuration_version=(
                f"{policy.version}|{policy.clip_context_policy.version}|{clip_config.version}"
            ),
            source_contract_version=(
                ",".join(contract_versions)
                if contract_versions
                else "core.basketball_event.v1"
            ),
            source_versions={
                "p2_events_sha256": source_fingerprint,
                "p2_runtime_versions": ",".join(runtime_versions) or "unknown",
            },
            configuration={
                "review_policy": policy.as_dict(),
                "clip_generation": clip_config.as_dict(),
                "artifact_bucket": getattr(
                    self.artifact_store, "bucket", "analysis-artifacts"
                ),
            },
        )
        derivation_id = self.persistence.start_derivation(
            workspace_id=str(workspace_id),
            spec=spec,
            provenance={
                "materializer_version": P3_THINGS_TO_REVIEW_DERIVATION_VERSION,
                **dict(provenance or {}),
            },
        )

        self.persistence.persist_source_events(
            workspace_id=str(workspace_id),
            derivation_id=derivation_id,
            events=rows,
        )
        snapshot = build_things_to_review(
            rows,
            policy=policy,
            game_id=str(game_id),
            analysis_run_id=str(analysis_run_id),
        )
        self.persistence.persist_review_snapshot(
            workspace_id=str(workspace_id),
            derivation_id=derivation_id,
            snapshot=snapshot,
        )

        clip_ids = [
            clip_plan_id(derivation_id, result.clip_request)
            for result in snapshot.items
            if result.clip_request is not None
        ]
        existing = self.persistence.get_clip_states(clip_ids)

        source_video = Path(source_video_path)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        ready = 0
        failed = 0
        reused = 0
        errors: list[str] = []

        for result in snapshot.items:
            request = result.clip_request
            if request is None:
                continue

            clip_id = clip_plan_id(derivation_id, request)
            state = existing.get(clip_id, {})
            if state.get("status") == "ready" and state.get("artifact_id"):
                ready += 1
                reused += 1
                continue

            self.persistence.mark_clip_generating(clip_id)
            output_path = output_root / f"{clip_id}.mp4"
            try:
                physical = self.clip_generator(
                    request,
                    source_path=source_video,
                    output_path=output_path,
                    config=clip_config,
                )
                stored: StoredClipArtifact = self.artifact_store.persist_clip(
                    workspace_id=str(workspace_id),
                    analysis_run_id=str(analysis_run_id),
                    clip_id=clip_id,
                    artifact=physical,
                )
                self.persistence.mark_clip_ready(
                    clip_id,
                    artifact_id=stored.artifact_id,
                )
                ready += 1
                if cleanup_local_after_upload:
                    output_path.unlink(missing_ok=True)
            except Exception as exc:
                failed += 1
                message = f"{clip_id}: {type(exc).__name__}: {exc}"
                errors.append(message[:2000])
                self.persistence.mark_clip_failed(clip_id, error=message)

        materialized = P3MaterializationResult(
            derivation_id=derivation_id,
            source_event_count=len(rows),
            review_item_count=snapshot.total_items,
            clip_planned_count=snapshot.clip_plannable_count,
            clip_ready_count=ready,
            clip_failed_count=failed,
            clip_reused_count=reused,
            counts_by_type=snapshot.counts_by_type(),
            counts_by_support_state=snapshot.counts_by_support_state(),
            excluded_by_policy=dict(snapshot.excluded_by_policy),
            errors=tuple(errors),
        )
        self.persistence.set_output_summary(
            derivation_id,
            {
                **materialized.as_dict(),
                "review_policy_version": policy.version,
                "clip_context_policy_version": policy.clip_context_policy.version,
                "clip_generation_version": clip_config.version,
                "source_events_sha256": source_fingerprint,
                "ranking_applied": False,
                "accuracy_thresholds_applied": bool(
                    policy.minimum_confidence_by_type
                ),
            },
        )
        return materialized

    def materialize_p2_jsonl(
        self,
        p2_events_path: str | Path,
        *,
        workspace_id: str,
        game_id: str,
        analysis_run_id: str,
        source_video_path: str | Path,
        output_dir: str | Path,
        p2_contract_version: str = "core.basketball_event.v1",
        p2_runtime_version: str | None = None,
        policy: ReviewBuildPolicy | None = None,
        clip_config: PhysicalClipConfig | None = None,
        cleanup_local_after_upload: bool = True,
        provenance: dict[str, Any] | None = None,
    ) -> P3MaterializationResult:
        rows = load_p2_event_rows(p2_events_path)
        events = adapt_p2_event_rows(
            rows,
            game_id=str(game_id),
            analysis_run_id=str(analysis_run_id),
            p2_contract_version=p2_contract_version,
            p2_runtime_version=p2_runtime_version,
        )
        return self.materialize(
            events,
            workspace_id=workspace_id,
            game_id=game_id,
            analysis_run_id=analysis_run_id,
            source_video_path=source_video_path,
            output_dir=output_dir,
            policy=policy,
            clip_config=clip_config,
            cleanup_local_after_upload=cleanup_local_after_upload,
            provenance={
                "p2_events_path": str(Path(p2_events_path)),
                **dict(provenance or {}),
            },
        )