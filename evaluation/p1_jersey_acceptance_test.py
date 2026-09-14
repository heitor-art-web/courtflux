from __future__ import annotations

from services.vision.p1_jersey_analyzer import PerformanceJerseyAnalyzer
from services.vision.performance_jersey_analyzer import (
    PerformanceJerseyAnalyzer as BasePerformanceJerseyAnalyzer,
)


def _prediction(raw: str | None, candidate: str | None, accepted: bool = True) -> dict:
    return {
        "accepted": accepted,
        "rejection_reason": "accepted" if accepted else "low_roster_margin",
        "raw_canonical": raw,
        "candidate_number": candidate,
    }


def test_raw_roster_disagreement_abstains() -> None:
    original = BasePerformanceJerseyAnalyzer.predict_jersey
    BasePerformanceJerseyAnalyzer.predict_jersey = lambda self, track_id, team_name=None: _prediction("6", "4")
    try:
        analyzer = PerformanceJerseyAnalyzer.__new__(PerformanceJerseyAnalyzer)
        result = analyzer.predict_jersey(12, "Ovarense")
    finally:
        BasePerformanceJerseyAnalyzer.predict_jersey = original

    assert result["accepted"] is False
    assert result["rejection_reason"] == "raw_roster_disagreement"
    assert result["rejected_candidate_number"] == "4"


def test_matching_raw_and_roster_candidate_remains_accepted() -> None:
    original = BasePerformanceJerseyAnalyzer.predict_jersey
    BasePerformanceJerseyAnalyzer.predict_jersey = lambda self, track_id, team_name=None: _prediction("4", "4")
    try:
        analyzer = PerformanceJerseyAnalyzer.__new__(PerformanceJerseyAnalyzer)
        result = analyzer.predict_jersey(12, "Ovarense")
    finally:
        BasePerformanceJerseyAnalyzer.predict_jersey = original

    assert result["accepted"] is True
    assert result["rejection_reason"] == "accepted"


def test_existing_rejection_is_preserved() -> None:
    original = BasePerformanceJerseyAnalyzer.predict_jersey
    BasePerformanceJerseyAnalyzer.predict_jersey = lambda self, track_id, team_name=None: _prediction("6", "4", accepted=False)
    try:
        analyzer = PerformanceJerseyAnalyzer.__new__(PerformanceJerseyAnalyzer)
        result = analyzer.predict_jersey(12, "Ovarense")
    finally:
        BasePerformanceJerseyAnalyzer.predict_jersey = original

    assert result["accepted"] is False
    assert result["rejection_reason"] == "low_roster_margin"


def main() -> None:
    test_raw_roster_disagreement_abstains()
    test_matching_raw_and_roster_candidate_remains_accepted()
    test_existing_rejection_is_preserved()
    print("p1_jersey_acceptance_test: PASS")


if __name__ == "__main__":
    main()
