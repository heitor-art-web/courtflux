from __future__ import annotations

from typing import Any

from core.p3_contract import P3DerivationSpec, P3SourceEvent
from services.p3.persistence import P3PersistenceStore, source_event_row


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
        if self.operation == "rpc:start_analysis_derivation":
            return _Result("derivation-1")
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

    def rpc(self, name: str, payload: Any) -> _Query:
        return _Query(self, f"rpc:{name}", payload)

    def table(self, name: str) -> _Table:
        return _Table(self, name)


def _source_event() -> P3SourceEvent:
    return P3SourceEvent(
        source_event_id="event-1",
        game_id="game-1",
        analysis_run_id="run-1",
        event_type="pass_candidate",
        decision="candidate",
        confidence=0.68,
        uncertainty=0.32,
        frame_start=30,
        frame_end=34,
        start_ms=1000,
        end_ms=1133,
        actor_track_id=7,
        actor_game_player_id="game_player_02",
        evidence=("handler_release",),
        provenance={"version": "possession_tracker_v1"},
    )


def test_source_event_row_preserves_lineage_fields() -> None:
    row = source_event_row(
        _source_event(),
        workspace_id="workspace-1",
        derivation_id="derivation-1",
    )
    assert row["workspace_id"] == "workspace-1"
    assert row["derivation_id"] == "derivation-1"
    assert row["source_event_id"] == "event-1"
    assert row["decision"] == "candidate"
    assert row["actor_track_id"] == 7
    assert row["actor_game_player_id"] == "game_player_02"
    assert row["evidence"] == ["handler_release"]
    assert row["provenance"]["version"] == "possession_tracker_v1"


def test_store_uses_atomic_derivation_rpc_and_upserts_source_events() -> None:
    client = _Client()
    store = P3PersistenceStore(client)
    spec = P3DerivationSpec(
        game_id="game-1",
        analysis_run_id="run-1",
        derivation_key="source_events",
        derivation_version="source-events-v1",
        configuration_version="config-v1",
        source_versions={"p2": "p2-v1"},
        configuration={"preserve_candidates": True},
    )

    derivation_id = store.start_derivation(
        workspace_id="workspace-1",
        spec=spec,
        provenance={"worker": "test"},
    )
    assert derivation_id == "derivation-1"
    assert client.calls[0]["operation"] == "rpc:start_analysis_derivation"
    assert client.calls[0]["payload"]["p_fingerprint"] == spec.fingerprint()
    assert client.calls[0]["payload"]["p_source_versions"] == {"p2": "p2-v1"}

    written = store.persist_source_events(
        workspace_id="workspace-1",
        derivation_id=derivation_id,
        events=[_source_event()],
    )
    assert written == 1
    assert client.calls[1]["operation"] == "upsert:p3_source_events"
    assert client.calls[1]["on_conflict"] == "derivation_id,source_event_id"
    assert client.calls[1]["payload"][0]["analysis_run_id"] == "run-1"


def test_store_records_output_summary_and_failure_without_touching_source_events() -> None:
    client = _Client()
    store = P3PersistenceStore(client)

    store.set_output_summary("derivation-1", {"event_count": 3})
    store.mark_failed("derivation-2", reason="fixture failure")

    assert client.calls[0]["operation"] == "update:analysis_derivations"
    assert client.calls[0]["payload"] == {"output_summary": {"event_count": 3}}
    assert client.calls[0]["filters"] == [("id", "derivation-1")]
    assert client.calls[1]["payload"]["status"] == "failed"
    assert client.calls[1]["filters"] == [("id", "derivation-2")]


def main() -> None:
    test_source_event_row_preserves_lineage_fields()
    test_store_uses_atomic_derivation_rpc_and_upserts_source_events()
    test_store_records_output_summary_and_failure_without_touching_source_events()
    print("p3_persistence_test: PASS")


if __name__ == "__main__":
    main()
