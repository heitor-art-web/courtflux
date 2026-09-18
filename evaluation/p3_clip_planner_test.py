from __future__ import annotations

from core.p3_clip import ClipContextPolicy, plan_evidence_clip
from core.p3_contract import P3SourceEvent


def _event(
    event_id: str,
    start_ms: int | None,
    end_ms: int | None,
    *,
    actor: str | None = None,
    secondary: str | None = None,
) -> P3SourceEvent:
    return P3SourceEvent(
        source_event_id=event_id,
        game_id="game-1",
        analysis_run_id="run-1",
        event_type="shot_attempt",
        decision="candidate",
        confidence=0.7,
        uncertainty=0.3,
        frame_start=10,
        frame_end=12,
        start_ms=start_ms,
        end_ms=end_ms,
        actor_game_player_id=actor,
        secondary_game_player_id=secondary,
    )


def test_default_shot_context_is_regenerable_from_source_time() -> None:
    request = plan_evidence_clip(
        [_event("shot-1", 10_000, 10_500, actor="game_player_03")],
        review_type="shot",
        source_duration_ms=60_000,
    )

    assert request.source_start_ms == 10_000
    assert request.source_end_ms == 10_500
    assert request.clip_start_ms == 4_000
    assert request.clip_end_ms == 14_500
    assert request.source_event_ids == ("shot-1",)
    assert request.game_player_ids == ("game_player_03",)
    assert request.context_policy_version == "p3.clip_context.v1"


def test_policy_change_changes_context_without_changing_source_interval() -> None:
    policy = ClipContextPolicy(
        version="coach-review-v2",
        windows_ms={"shot": (2_000, 1_000)},
    )
    request = plan_evidence_clip(
        [_event("shot-1", 10_000, 10_500)],
        review_type="shot",
        policy=policy,
    )

    assert request.source_start_ms == 10_000
    assert request.source_end_ms == 10_500
    assert request.clip_start_ms == 8_000
    assert request.clip_end_ms == 11_500
    assert request.context_policy_version == "coach-review-v2"


def test_clip_bounds_are_clamped_to_source_video() -> None:
    request = plan_evidence_clip(
        [_event("event-1", 1_000, 59_000)],
        review_type="turnover",
        source_duration_ms=60_000,
    )
    assert request.clip_start_ms == 0
    assert request.clip_end_ms == 60_000


def test_multiple_events_preserve_all_evidence_ids_and_players() -> None:
    request = plan_evidence_clip(
        [
            _event("pass-1", 20_000, 20_300, actor="game_player_05"),
            _event(
                "shot-1",
                21_000,
                21_400,
                actor="game_player_03",
                secondary="game_player_05",
            ),
        ],
        review_type="alley_oop",
    )
    assert request.source_start_ms == 20_000
    assert request.source_end_ms == 21_400
    assert request.source_event_ids == ("pass-1", "shot-1")
    assert request.game_player_ids == ("game_player_03", "game_player_05")


def test_clip_requires_timestamped_evidence() -> None:
    try:
        plan_evidence_clip(
            [_event("event-1", None, None)],
            review_type="shot",
        )
    except ValueError as exc:
        assert "requires source-video timestamps" in str(exc)
    else:
        raise AssertionError("untimestamped evidence must not produce a physical clip")


def main() -> None:
    test_default_shot_context_is_regenerable_from_source_time()
    test_policy_change_changes_context_without_changing_source_interval()
    test_clip_bounds_are_clamped_to_source_video()
    test_multiple_events_preserve_all_evidence_ids_and_players()
    test_clip_requires_timestamped_evidence()
    print("p3_clip_planner_test: PASS")


if __name__ == "__main__":
    main()
