from services.p2.possession_tracker_v1 import PossessionTrackerV1


def _tracker() -> PossessionTrackerV1:
    tracker = PossessionTrackerV1()
    tracker.current_track_id = 13
    tracker.current_team_id = "team_a"
    tracker.current_confidence = 0.8
    return tracker


def test_supported_owner_keeps_team_attribution() -> None:
    tracker = _tracker()
    tracker._lost_frames = 0
    tracker._candidate_track_id = None
    tracker._candidate_frames = 0

    diagnostics = tracker.diagnostics()

    assert diagnostics["current_team_id"] == "team_a"
    assert diagnostics["team_attribution_abstained"] is False
    assert diagnostics["team_attribution_abstention_reason"] is None
    assert tracker.current_team_id == "team_a"


def test_stale_owner_abstains_without_mutating_internal_team_memory() -> None:
    tracker = _tracker()
    tracker._lost_frames = 1
    tracker._raw_best_track_id = 14
    tracker._raw_best_score = 0.55

    diagnostics = tracker.diagnostics()

    assert diagnostics["current_team_id"] is None
    assert diagnostics["internal_current_team_id"] == "team_a"
    assert diagnostics["team_attribution_abstained"] is True
    assert diagnostics["team_attribution_abstention_reason"] == "owner_not_currently_supported"
    assert diagnostics["raw_best_matches_current_owner"] is False
    assert tracker.current_team_id == "team_a"


def test_stale_owner_keeps_team_when_same_owner_is_still_raw_best() -> None:
    tracker = _tracker()
    tracker._lost_frames = 2
    tracker._raw_best_track_id = 13
    tracker._raw_best_score = 0.56

    diagnostics = tracker.diagnostics()

    assert diagnostics["current_team_id"] == "team_a"
    assert diagnostics["internal_current_team_id"] == "team_a"
    assert diagnostics["team_attribution_abstained"] is False
    assert diagnostics["team_attribution_abstention_reason"] is None
    assert diagnostics["raw_best_matches_current_owner"] is True
    assert tracker.current_team_id == "team_a"


def test_distinct_pending_challenger_abstains_until_switch_is_confirmed() -> None:
    tracker = _tracker()
    tracker._lost_frames = 1
    tracker._raw_best_track_id = 13
    tracker._raw_best_score = 0.7
    tracker._candidate_track_id = 14
    tracker._candidate_frames = 1

    diagnostics = tracker.diagnostics()

    assert diagnostics["current_team_id"] is None
    assert diagnostics["internal_current_team_id"] == "team_a"
    assert diagnostics["team_attribution_abstained"] is True
    assert diagnostics["team_attribution_abstention_reason"] == "distinct_challenger_pending"
    assert diagnostics["raw_best_matches_current_owner"] is True
    assert tracker.current_team_id == "team_a"


def test_same_owner_candidate_does_not_abstain() -> None:
    tracker = _tracker()
    tracker._lost_frames = 0
    tracker._candidate_track_id = 13
    tracker._candidate_frames = 1

    diagnostics = tracker.diagnostics()

    assert diagnostics["current_team_id"] == "team_a"
    assert diagnostics["team_attribution_abstained"] is False
    assert tracker.current_team_id == "team_a"


def main() -> None:
    test_supported_owner_keeps_team_attribution()
    test_stale_owner_abstains_without_mutating_internal_team_memory()
    test_stale_owner_keeps_team_when_same_owner_is_still_raw_best()
    test_distinct_pending_challenger_abstains_until_switch_is_confirmed()
    test_same_owner_candidate_does_not_abstain()
    print("p2_team_attribution_abstention_test: PASS")


if __name__ == "__main__":
    main()
