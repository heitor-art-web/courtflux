from __future__ import annotations

from core.p3_clip import (
    P3_CLIP_REQUEST_CONTRACT_VERSION,
    ClipContextPolicy,
    plan_evidence_clip,
)
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

    assert request.source_video_ref == "analysis-run:run-1"
    assert request.source_start_ms == 10_000
    assert request.source_end_ms == 10_500
    assert request.clip_start_ms == 4_000
    assert request.clip_end_ms == 14_500
    assert request.source_event_ids == ("shot-1",)
    assert request.game_player_ids == ("game_player_03",)
    assert request.context_policy_version == "p3.clip_context.v1"
    assert request.contract_version == P3_CLIP_REQUEST_CONTRACT_VERSION
    assert len(request.fingerprint()) == 64


def test_policy_change_changes_context_without_changing_source_interval() -> None:
    policy = ClipContextPolicy(
        version="coach-review-v2",
        windows_ms={"shot": (2_000, 1_000)},
    )
    request = plan_evidence_clip(
        [_event("shot-1", 10_000, 10_500)],
        review_type="shot",
        policy=policy,
        source_video_ref="upload:game-1",
    )

    assert request.source_video_ref == "upload:game-1"
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


def test_source_interval_outside_video_is_rejected() -> None:
    try:
        plan_evidence_clip(
            [_event("event-1", 59_500, 60_500)],
            review_type="shot",
            source_duration_ms=60_000,
        )
    except ValueError as exc:
        assert "outside the source-video duration" in str(exc)
    else:
        raise AssertionError("invalid source interval must not produce a clip plan")


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
        source_possession_id="possession-12",
    )
    assert request.source_start_ms == 20_000
    assert request.source_end_ms == 21_400
    assert request.source_event_ids == ("pass-1", "shot-1")
    assert request.source_possession_id == "possession-12"
    assert request.game_player_ids == ("game_player_03", "game_player_05")


def test_clip_requires_timestamped_evidence_for_every_source_event() -> None:
    try:
        plan_evidence_clip(
            [
                _event("event-1", 10_000, 10_500),
                _event("event-2", None, None),
            ],
            review_type="shot",
        )
    except ValueError as exc:
        assert "every source event" in str(exc)
    else:
        raise AssertionError("partial timestamps must not produce a physical clip")


if __name__ == "__main__":
    test_default_shot_context_is_regenerable_from_source_time()
    test_policy_change_changes_context_without_changing_source_interval()
    test_clip_bounds_are_clamped_to_source_video()
    test_source_interval_outside_video_is_rejected()
    test_multiple_events_preserve_all_evidence_ids_and_players()
    test_clip_requires_timestamped_evidence_for_every_source_event()
    print("p3_clip_planner_test: PASS")
