from __future__ import annotations

from core.p3_contract import P3SourceEvent
from services.p3.review_builder import ReviewBuildPolicy, build_things_to_review


def _event(
    event_id: str,
    event_type: str,
    *,
    decision: str = "candidate",
    confidence: float = 0.7,
    frame_start: int = 100,
    frame_end: int = 102,
    start_ms: int | None = 10_000,
    end_ms: int | None = 10_200,
    actor: str | None = "game_player_03",
    secondary: str | None = None,
    team_id: str | None = "team-a",
    attributes: dict | None = None,
    evidence: tuple[str, ...] = ("fixture",),
) -> P3SourceEvent:
    return P3SourceEvent(
        source_event_id=event_id,
        game_id="game-1",
        analysis_run_id="run-1",
        event_type=event_type,
        decision=decision,
        confidence=confidence,
        uncertainty=1.0 - confidence,
        frame_start=frame_start,
        frame_end=frame_end,
        start_ms=start_ms,
        end_ms=end_ms,
        actor_game_player_id=actor,
        secondary_game_player_id=secondary,
        team_id=team_id,
        attributes=attributes or {},
        evidence=evidence,
    )


def test_default_builder_materializes_reviewable_taxonomy_without_confidence_filter() -> None:
    snapshot = build_things_to_review(
        [
            _event("pass-1", "pass_candidate", confidence=0.12),
            _event("turnover-1", "turnover_candidate", frame_start=120, start_ms=12_000),
            _event("shot-1", "shot_attempt", frame_start=140, start_ms=14_000),
            _event("rebound-1", "rebound_candidate", frame_start=160, start_ms=16_000),
            _event("ignored-1", "possession_acquired", frame_start=180, start_ms=18_000),
        ]
    )

    assert snapshot.total_items == 4
    assert snapshot.counts_by_type() == {
        "pass": 1,
        "turnover": 1,
        "shot": 1,
        "rebound": 1,
    }
    assert snapshot.excluded_by_policy == {}
    pass_item = next(result.item for result in snapshot.items if result.item.review_type == "pass")
    assert pass_item.confidence == 0.12
    assert "configured_minimum_confidence" not in pass_item.reason_shown


def test_dunk_remains_a_shot_classification_source() -> None:
    snapshot = build_things_to_review(
        [_event("shot-dunk-1", "shot_attempt", attributes={"shot_type": "DUNK"})]
    )
    assert snapshot.counts_by_type() == {"dunk": 1}
    assert snapshot.items[0].item.source_event_ids == ("shot-dunk-1",)


def test_loose_ball_lifecycle_collapses_into_one_review_item() -> None:
    started = _event(
        "loose-ball-started:110:7",
        "loose_ball_started",
        frame_start=108,
        frame_end=110,
        start_ms=10_900,
        end_ms=11_000,
        confidence=0.55,
    )
    recovered = _event(
        "loose-ball-recovered:110:125:9",
        "loose_ball_recovered",
        decision="confirmed",
        frame_start=110,
        frame_end=125,
        start_ms=11_000,
        end_ms=12_500,
        confidence=0.88,
        actor="game_player_05",
        secondary="game_player_03",
    )

    snapshot = build_things_to_review([started, recovered])
    assert snapshot.total_items == 1
    result = snapshot.items[0]
    assert result.item.review_type == "loose_ball"
    assert result.item.support_state == "mixed"
    assert result.item.completeness == "complete"
    assert result.item.confidence == 0.55
    assert set(result.item.source_event_ids) == {
        "loose-ball-started:110:7",
        "loose-ball-recovered:110:125:9",
    }
    assert result.clip_request is not None
    assert result.clip_request.source_start_ms == 10_900
    assert result.clip_request.source_end_ms == 12_500


def test_unmatched_loose_start_is_fragmented_but_reviewable() -> None:
    snapshot = build_things_to_review(
        [
            _event(
                "loose-ball-started:110:7",
                "loose_ball_started",
                frame_start=108,
                frame_end=110,
                start_ms=10_900,
                end_ms=11_000,
            )
        ]
    )
    assert snapshot.total_items == 1
    assert snapshot.items[0].item.completeness == "fragmented"


def test_configured_threshold_is_explicit_and_measurable() -> None:
    policy = ReviewBuildPolicy(minimum_confidence_by_type={"turnover": 0.75})
    snapshot = build_things_to_review(
        [_event("turnover-low", "turnover_candidate", confidence=0.6)],
        policy=policy,
    )
    assert snapshot.total_items == 0
    assert snapshot.excluded_by_policy == {"minimum_confidence:turnover": 1}
    assert policy.as_dict()["minimum_confidence_by_type"] == {"turnover": 0.75}


def test_missing_timestamps_keeps_review_item_without_fake_clip() -> None:
    snapshot = build_things_to_review(
        [_event("pass-no-time", "pass_candidate", start_ms=None, end_ms=None)]
    )
    assert snapshot.total_items == 1
    assert snapshot.items[0].clip_request is None
    assert snapshot.items[0].item.completeness == "unknown"
    assert snapshot.items[0].item.reason_shown["clip_plannable"] is False


def test_empty_game_is_a_valid_snapshot_when_identity_is_supplied() -> None:
    snapshot = build_things_to_review(
        [], game_id="game-1", analysis_run_id="run-1"
    )
    assert snapshot.total_items == 0
    assert snapshot.game_id == "game-1"
    assert snapshot.analysis_run_id == "run-1"


def main() -> None:
    test_default_builder_materializes_reviewable_taxonomy_without_confidence_filter()
    test_dunk_remains_a_shot_classification_source()
    test_loose_ball_lifecycle_collapses_into_one_review_item()
    test_unmatched_loose_start_is_fragmented_but_reviewable()
    test_configured_threshold_is_explicit_and_measurable()
    test_missing_timestamps_keeps_review_item_without_fake_clip()
    test_empty_game_is_a_valid_snapshot_when_identity_is_supplied()
    print("p3_review_builder_test: PASS")


if __name__ == "__main__":
    main()
