from __future__ import annotations

import json
from typing import Any, Iterable
from uuid import UUID, uuid5

from core.p3_clip import P3ClipRequest
from core.p3_contract import P3DerivationSpec, P3SourceEvent
from services.p3.review_builder import ReviewItemDraft, ThingsToReviewSnapshot


_P3_ID_NAMESPACE = UUID("d92d9a8d-1b86-4fa8-9b10-398bc971dc4e")


def source_event_row(
    event: P3SourceEvent,
    *,
    workspace_id: str,
    derivation_id: str,
) -> dict[str, Any]:
    """Convert the stable P3 source contract into a durable database row."""

    return {
        "workspace_id": str(workspace_id),
        "game_id": event.game_id,
        "analysis_run_id": event.analysis_run_id,
        "derivation_id": str(derivation_id),
        "source_event_id": event.source_event_id,
        "event_type": event.event_type,
        "decision": event.decision,
        "confidence": event.confidence,
        "uncertainty": event.uncertainty,
        "frame_start": event.frame_start,
        "frame_end": event.frame_end,
        "start_ms": event.start_ms,
        "end_ms": event.end_ms,
        "actor_track_id": event.actor_track_id,
        "secondary_track_id": event.secondary_track_id,
        "actor_game_player_id": event.actor_game_player_id,
        "secondary_game_player_id": event.secondary_game_player_id,
        "team_id": event.team_id,
        "location_court_m": (
            list(event.location_court_m) if event.location_court_m is not None else None
        ),
        "evidence": list(event.evidence),
        "attributes": dict(event.attributes),
        "provenance": dict(event.provenance),
        "contract_version": event.contract_version,
        "source_contract_version": event.source_contract_version,
        "source_runtime_version": event.source_runtime_version,
    }


def _stable_uuid(kind: str, derivation_id: str, identity: str) -> str:
    return str(uuid5(_P3_ID_NAMESPACE, f"{kind}:{derivation_id}:{identity}"))


def clip_plan_id(derivation_id: str, request: P3ClipRequest) -> str:
    identity = json.dumps(
        request.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return _stable_uuid("clip", str(derivation_id), identity)


def review_item_id(derivation_id: str, item: ReviewItemDraft) -> str:
    return _stable_uuid("review", str(derivation_id), item.review_key())


def clip_plan_row(
    request: P3ClipRequest,
    *,
    workspace_id: str,
    derivation_id: str,
) -> dict[str, Any]:
    return {
        "id": clip_plan_id(derivation_id, request),
        "workspace_id": str(workspace_id),
        "game_id": request.game_id,
        "analysis_run_id": request.analysis_run_id,
        "derivation_id": str(derivation_id),
        "review_type": request.review_type,
        "source_start_ms": request.source_start_ms,
        "source_end_ms": request.source_end_ms,
        "clip_start_ms": request.clip_start_ms,
        "clip_end_ms": request.clip_end_ms,
        "source_event_ids": list(request.source_event_ids),
        "game_player_ids": list(request.game_player_ids),
        "context_policy_version": request.context_policy_version,
        "generation_version": request.generation_version,
    }


def review_item_row(
    item: ReviewItemDraft,
    *,
    workspace_id: str,
    derivation_id: str,
    clip_id: str | None,
) -> dict[str, Any]:
    return {
        "id": review_item_id(derivation_id, item),
        "workspace_id": str(workspace_id),
        "game_id": item.game_id,
        "analysis_run_id": item.analysis_run_id,
        "derivation_id": str(derivation_id),
        "clip_id": clip_id,
        "review_type": item.review_type,
        "source_event_ids": list(item.source_event_ids),
        "source_possession_id": item.source_possession_id,
        "game_player_ids": list(item.game_player_ids),
        "team_id": item.team_id,
        "confidence": item.confidence,
        "support_state": item.support_state,
        "completeness": item.completeness,
        "reason_shown": dict(item.reason_shown),
    }


class P3PersistenceStore:
    """Small Supabase adapter for P3 foundation persistence.

    The service-role client remains the authority for model/derived writes. Starting a
    derivation is delegated to the database RPC so current/stale/superseded transitions
    happen transactionally. Human reviews remain a separate authenticated-user surface.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def start_derivation(
        self,
        *,
        workspace_id: str,
        spec: P3DerivationSpec,
        layer: str = "p3",
        provenance: dict[str, Any] | None = None,
    ) -> str:
        result = self.client.rpc(
            "start_analysis_derivation",
            {
                "p_workspace_id": str(workspace_id),
                "p_game_id": spec.game_id,
                "p_analysis_run_id": spec.analysis_run_id,
                "p_layer": str(layer),
                "p_derivation_key": spec.derivation_key,
                "p_contract_version": spec.contract_version,
                "p_source_contract_version": spec.source_contract_version,
                "p_derivation_version": spec.derivation_version,
                "p_configuration_version": spec.configuration_version,
                "p_fingerprint": spec.fingerprint(),
                "p_source_versions": dict(spec.source_versions),
                "p_configuration": dict(spec.configuration),
                "p_provenance": dict(provenance or {}),
            },
        ).execute()
        derivation_id = result.data
        if isinstance(derivation_id, list):
            derivation_id = derivation_id[0] if derivation_id else None
        if isinstance(derivation_id, dict):
            derivation_id = derivation_id.get("start_analysis_derivation") or derivation_id.get("id")
        if not derivation_id:
            raise RuntimeError("start_analysis_derivation returned no derivation id")
        return str(derivation_id)

    def persist_source_events(
        self,
        *,
        workspace_id: str,
        derivation_id: str,
        events: Iterable[P3SourceEvent],
    ) -> int:
        rows = [
            source_event_row(
                event,
                workspace_id=workspace_id,
                derivation_id=derivation_id,
            )
            for event in events
        ]
        if not rows:
            return 0
        self.client.table("p3_source_events").upsert(
            rows,
            on_conflict="derivation_id,source_event_id",
        ).execute()
        return len(rows)

    def persist_review_snapshot(
        self,
        *,
        workspace_id: str,
        derivation_id: str,
        snapshot: ThingsToReviewSnapshot,
    ) -> dict[str, int]:
        clip_rows: list[dict[str, Any]] = []
        item_rows: list[dict[str, Any]] = []

        for result in snapshot.items:
            clip_id: str | None = None
            if result.clip_request is not None:
                clip_row = clip_plan_row(
                    result.clip_request,
                    workspace_id=workspace_id,
                    derivation_id=derivation_id,
                )
                clip_id = str(clip_row["id"])
                clip_rows.append(clip_row)
            item_rows.append(
                review_item_row(
                    result.item,
                    workspace_id=workspace_id,
                    derivation_id=derivation_id,
                    clip_id=clip_id,
                )
            )

        if clip_rows:
            # Do not include status/artifact/error in the upsert payload. Re-running the
            # same derivation must never turn an already-ready clip back into planned.
            self.client.table("p3_clips").upsert(clip_rows, on_conflict="id").execute()
        if item_rows:
            self.client.table("p3_review_items").upsert(
                item_rows,
                on_conflict="id",
            ).execute()
        return {"clips": len(clip_rows), "review_items": len(item_rows)}

    def get_clip_states(self, clip_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = sorted({str(clip_id) for clip_id in clip_ids if str(clip_id)})
        if not ids:
            return {}
        result = (
            self.client.table("p3_clips")
            .select("id,status,artifact_id,error")
            .in_("id", ids)
            .execute()
        )
        rows = getattr(result, "data", None) or []
        return {
            str(row["id"]): {
                "status": row.get("status"),
                "artifact_id": row.get("artifact_id"),
                "error": row.get("error"),
            }
            for row in rows
            if isinstance(row, dict) and row.get("id")
        }

    def mark_clip_generating(self, clip_id: str) -> None:
        self.client.table("p3_clips").update(
            {"status": "generating", "error": None}
        ).eq("id", str(clip_id)).execute()

    def mark_clip_ready(self, clip_id: str, *, artifact_id: str) -> None:
        self.client.table("p3_clips").update(
            {"status": "ready", "artifact_id": str(artifact_id), "error": None}
        ).eq("id", str(clip_id)).execute()

    def mark_clip_failed(self, clip_id: str, *, error: str) -> None:
        self.client.table("p3_clips").update(
            {"status": "failed", "error": str(error)[:2000]}
        ).eq("id", str(clip_id)).execute()

    def set_output_summary(
        self,
        derivation_id: str,
        summary: dict[str, Any],
    ) -> None:
        self.client.table("analysis_derivations").update(
            {"output_summary": dict(summary)}
        ).eq("id", str(derivation_id)).execute()

    def mark_failed(self, derivation_id: str, *, reason: str) -> None:
        self.client.table("analysis_derivations").update(
            {
                "status": "failed",
                "stale_reason": str(reason)[:1000],
            }
        ).eq("id", str(derivation_id)).execute()