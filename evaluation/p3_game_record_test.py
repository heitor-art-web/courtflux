from __future__ import annotations

from core.p3_contract import P3SourceEvent
from services.p3.game_record import build_game_record_snapshot


def _event(
    event_id: str,
    *,
    event_type: str,
    decision: str,
    start_ms: int | None,
    end_ms: int | None,
    evidence: tuple[str, ...] = (),
    actor_game_player_id: str | None = None,
) -> P3SourceEvent:
    return P3SourceEvent(
        source_event_id=event_id,
        game_id="game-1",
        analysis_run_id="run-1",
        event_type=event_type,
        decision=decision,
        confidence=0.5,
        uncertainty=0.5,
        frame_start=0,
        frame_end=1,
        start_ms=start_ms,
        end_ms=end_ms,
        evidence=evidence,
        actor_game_player_id=actor_game_player_id,
    )


def test_snapshot_keeps_candidates_separate_from_confirmed() -> None:
    snapshot = build_game_record_snapshot(
        [
            _event(
                "e1",
                event_type="turnover_candidate",
                decision="candidate",
                start_ms=1000,
                end_ms=1200,
                evidence=("possession_change",),
            ),
            _event(
                "e2",
                event_type="shot_attempt",
                decision="confirmed",
                start_ms=2000,
                end_ms=2300,
                evidence=("shot_flight",),
                actor_game_player_id="game_player_03",
            ),
            _event(
                "e3",
                event_type="shot_attempt",
                decision="candidate",
                start_ms=None,
                end_ms=None,
            ),
        ]
    )

    assert snapshot.event_count == 3
    assert snapshot.event_counts == {"shot_attempt": 2, "turnover_candidate": 1}
    assert snapshot.decision_counts == {"candidate": 2, "confirmed": 1}
    assert snapshot.evidence_event_count == 2
    assert snapshot.attributed_event_count == 1
    assert snapshot.timestamped_event_count == 2
    assert snapshot.first_start_ms == 1000
    assert snapshot.last_end_ms == 2300
    assert snapshot.evidence_coverage == 2 / 3
    assert snapshot.attribution_coverage == 1 / 3
    assert snapshot.timestamp_coverage == 2 / 3


def test_snapshot_rejects_mixed_analysis_runs() -> None:
    first = _event(
        "e1",
        event_type="pass_candidate",
        decision="candidate",
        start_ms=1000,
        end_ms=1100,
    )
    second = _event(
        "e2",
        event_type="shot_attempt",
        decision="candidate",
        start_ms=1200,
        end_ms=1300,
    )
    second.analysis_run_id = "run-2"

    try:
        build_game_record_snapshot([first, second])
    except ValueError as exc:
        assert "cannot mix games or analysis runs" in str(exc)
    else:
        raise AssertionError("mixed analysis runs must be rejected")


def main() -> None:
    test_snapshot_keeps_candidates_separate_from_confirmed()
    test_snapshot_rejects_mixed_analysis_runs()
    print("p3_game_record_test: PASS")


if __name__ == "__main__":
    main()
