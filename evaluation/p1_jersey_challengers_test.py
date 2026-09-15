from __future__ import annotations

from services.vision.p1_jersey_challengers import (
    ExternalJerseyBase,
    RoboflowClassificationJerseyAnalyzer,
)


class StubExternalJerseyAnalyzer(ExternalJerseyBase):
    BACKEND_NAME = "stub_external"



def _observation() -> dict:
    return {
        "crop": None,
        "quality": 0.9,
        "ordinal": 1,
        "frame_id": 100,
    }


def test_strict_number_parser() -> None:
    assert ExternalJerseyBase.parse_number("23") == ("23", "parsed")
    assert ExternalJerseyBase.parse_number("Jersey number: 13") == ("13", "parsed")
    assert ExternalJerseyBase.parse_number("00") == ("00", "parsed")
    assert ExternalJerseyBase.parse_number("UNKNOWN") == (None, "model_abstained")
    assert ExternalJerseyBase.parse_number("13 or 23") == (
        None,
        "ambiguous_numeric_output",
    )
    assert ExternalJerseyBase.parse_number("123") == (
        None,
        "out_of_range_or_malformed",
    )


def test_roboflow_classification_parser() -> None:
    response = {
        "predictions": [
            {"class": "23", "confidence": 0.81},
            {"class": "13", "confidence": 0.12},
            {"class": "8", "confidence": 0.04},
        ]
    }
    candidates = RoboflowClassificationJerseyAnalyzer._classification_candidates(response)
    assert candidates[0] == {"number": "23", "posterior": 0.81}
    assert candidates[1] == {"number": "13", "posterior": 0.12}


def test_roster_is_validator_only() -> None:
    analyzer = StubExternalJerseyAnalyzer(
        profiling_path="data/match_config.json",
        crop_mode="torso",
    )
    rejected = analyzer._finalize_prediction(
        track_id=1,
        team_name="Ovarense",
        selected=[_observation()],
        number="66",
        parse_reason="parsed",
        confidence=0.95,
        margin=0.8,
    )
    assert rejected["accepted"] is False
    assert rejected["rejection_reason"] == "raw_not_in_roster"
    assert rejected["candidate_number"] == "66"
    assert rejected["raw_canonical"] == "66"
    assert "23" in rejected["valid_roster_numbers"]
    assert rejected["candidate_number"] != "23"

    accepted = analyzer._finalize_prediction(
        track_id=1,
        team_name="Ovarense",
        selected=[_observation()],
        number="23",
        parse_reason="parsed",
        confidence=0.85,
        margin=0.6,
    )
    assert accepted["accepted"] is True
    assert accepted["candidate_number"] == "23"
    assert accepted["roster_policy"] == "validator_only"


def main() -> int:
    test_strict_number_parser()
    test_roboflow_classification_parser()
    test_roster_is_validator_only()
    print("p1_jersey_challengers_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
