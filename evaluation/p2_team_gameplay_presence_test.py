from types import SimpleNamespace

from services.vision.identity_v2_detector import PerformanceDetector


def _frame(*, gameplay_present: bool | None, analysis_eligible: bool):
    court_state = SimpleNamespace(analysis_eligible=analysis_eligible)
    if gameplay_present is not None:
        court_state.gameplay_present = gameplay_present
    return SimpleNamespace(
        court_state=court_state,
        players=[
            {"track_id": 1, "active_on_court": True},
            {"track_id": 2, "active_on_court": None},
            {"track_id": 3, "active_on_court": False},
        ],
    )


def test_team_evidence_uses_only_positive_on_court_players_during_limited_gameplay() -> None:
    frame = _frame(gameplay_present=True, analysis_eligible=False)

    eligible = PerformanceDetector._team_eligible_players(frame)

    assert [row["track_id"] for row in eligible] == [1]
    assert PerformanceDetector._identity_eligible_players(frame) == []


def test_analysis_eligible_gameplay_keeps_unknown_on_court_players() -> None:
    frame = _frame(gameplay_present=True, analysis_eligible=True)

    eligible = PerformanceDetector._team_eligible_players(frame)

    assert [row["track_id"] for row in eligible] == [1, 2]


def test_team_evidence_stays_off_when_gameplay_is_absent() -> None:
    frame = _frame(gameplay_present=False, analysis_eligible=True)

    assert PerformanceDetector._team_eligible_players(frame) == []


def test_team_evidence_falls_back_to_analysis_eligible_for_legacy_court_state() -> None:
    eligible_frame = _frame(gameplay_present=None, analysis_eligible=True)
    ineligible_frame = _frame(gameplay_present=None, analysis_eligible=False)

    assert [
        row["track_id"]
        for row in PerformanceDetector._team_eligible_players(eligible_frame)
    ] == [1, 2]
    assert PerformanceDetector._team_eligible_players(ineligible_frame) == []


def main() -> None:
    test_team_evidence_uses_only_positive_on_court_players_during_limited_gameplay()
    test_analysis_eligible_gameplay_keeps_unknown_on_court_players()
    test_team_evidence_stays_off_when_gameplay_is_absent()
    test_team_evidence_falls_back_to_analysis_eligible_for_legacy_court_state()
    print("p2_team_gameplay_presence_test: PASS")


if __name__ == "__main__":
    main()
