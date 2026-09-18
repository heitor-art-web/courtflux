from __future__ import annotations

from typing import Any

from core.p3_contract import P3SourceEvent
from services.p3.persistence import P3PersistenceStore, clip_plan_id, review_item_id
from services.p3.review_builder import build_things_to_review


class _Result:
    def __init__(self, data: Any = None) -> None:
        self.data = data


class _Query:
    def __init__(self, client: "_Client", operation: str, payload: Any = None) -> None:
        self.client = client
        self.operation = operation
        self.payload = payload
        self.filters: list[tuple[str, str]] = []
        self.on_conflict: str | None = None

    def eq(self, key: str, value: str) -> "_Query":
        self.filters.append((key, value))
        return self

    def execute(self) -> _Result:
        self.client.calls.append(
            {
                "operation": self.operation,
                "payload": self.payload,
                "filters": list(self.filters),
                "on_conflict": self.on_conflict,
            }
        )
        return _Result([])


class _Table:
    def __init__(self, client: "_Client", name: str) -> None:
        self.client = client
        self.name = name

    def upsert(self, payload: Any, *, on_conflict: str) -> _Query:
        query = _Query(self.client, f"upsert:{self.name}", payload)
        query.on_conflict = on_conflict
        return query

    def update(self, payload: Any) -> _Query:
        return _Query(self.client, f"update:{self.name}", payload)


class _Client:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def table(self, name: str) -> _Table:
        return _Table(self, name)

    def rpc(self, name: str, payload: Any) -> _Query:
        return _Query(self, f"rpc:{name}", payload)


def _event() -> P3SourceEvent:
    return P3SourceEvent(
        source_event_id="shot-1",
        game_id="game-1",
        analysis_run_id="run-1",
        event_type="shot_attempt",
        decision="candidate",
        confidence=0.7,
        uncertainty=0.3,
        frame_start=10,
        frame_end=12,
        start_ms=10_000,
        end_ms=10_500,
        actor_game_player_id="game_player_03",
    )


def test_review_snapshot_persistence_is_idempotent_by_deterministic_ids() -> None:
    snapshot = build_things_to_review([_event()])
    client = _Client()
    store = P3PersistenceStore(client)
    derivation_id = "11111111-1111-1111-1111-111111111111"

    counts = store.persist_review_snapshot(
        workspace_id="workspace-1",
        derivation_id=derivation_id,
        snapshot=snapshot,
    )

    assert counts == {"clips": 1, "review_items": 1}
    clip_call, review_call = client.calls
    assert clip_call["operation"] == "upsert:p3_clips"
    assert clip_call["on_conflict"] == "id"
    assert review_call["operation"] == "upsert:p3_review_items"
    assert review_call["on_conflict"] == "id"

    clip = clip_call["payload"][0]
    item = review_call["payload"][0]
    assert item["clip_id"] == clip["id"]
    assert "status" not in clip
    assert "artifact_id" not in clip
    assert "error" not in clip
    assert clip["id"] == clip_plan_id(derivation_id, snapshot.items[0].clip_request)
    assert item["id"] == review_item_id(derivation_id, snapshot.items[0].item)


def test_clip_status_transitions_are_separate_from_generation() -> None:
    client = _Client()
    store = P3PersistenceStore(client)

    store.mark_clip_generating("clip-1")
    store.mark_clip_ready("clip-1", artifact_id="artifact-1")
    store.mark_clip_failed("clip-2", error="upload failed")

    assert client.calls[0]["payload"] == {"status": "generating", "error": None}
    assert client.calls[1]["payload"] == {
        "status": "ready",
        "artifact_id": "artifact-1",
        "error": None,
    }
    assert client.calls[2]["payload"]["status"] == "failed"
    assert client.calls[2]["filters"] == [("id", "clip-2")]


def main() -> None:
    test_review_snapshot_persistence_is_idempotent_by_deterministic_ids()
    test_clip_status_transitions_are_separate_from_generation()
    print("p3_review_persistence_test: PASS")


if __name__ == "__main__":
    main()
