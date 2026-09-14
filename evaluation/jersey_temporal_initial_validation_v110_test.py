from __future__ import annotations

from evaluation.jersey_temporal_initial_validation_v110 import (
    _aggregate_track_predictions,
    _sequence_plan,
)


def main() -> int:
    assert _sequence_plan(0) == []
    assert _sequence_plan(1) == []

    plan2 = _sequence_plan(2)
    assert len(plan2) == 1
    assert plan2[0]["support"] == "partial_repeated"
    assert len(plan2[0]["indices"]) == 8
    assert set(plan2[0]["indices"]) == {0, 1}

    plan7 = _sequence_plan(7)
    assert len(plan7) == 1
    assert plan7[0]["support"] == "partial_repeated"
    assert len(plan7[0]["indices"]) == 8

    plan8 = _sequence_plan(8)
    assert len(plan8) == 1
    assert plan8[0]["support"] == "full_8_distinct"
    assert plan8[0]["indices"] == list(range(8))

    plan16 = _sequence_plan(16)
    assert len(plan16) == 3
    assert plan16[0]["indices"] == list(range(0, 8))
    assert plan16[1]["indices"] == list(range(4, 12))
    assert plan16[2]["indices"] == list(range(8, 16))

    empty = _aggregate_track_predictions([])
    assert empty["prediction"] is None

    one = _aggregate_track_predictions(
        [{"prediction": "8", "raw_probability": 0.4}]
    )
    assert one["prediction"] == "8"
    assert one["stable"] is None
    assert one["modal_fraction"] == 1.0

    stable = _aggregate_track_predictions(
        [
            {"prediction": "8", "raw_probability": 0.4},
            {"prediction": "8", "raw_probability": 0.5},
            {"prediction": "8", "raw_probability": 0.3},
        ]
    )
    assert stable["prediction"] == "8"
    assert stable["stable"] is True
    assert stable["modal_fraction"] == 1.0
    assert stable["best_probability"] == 0.5

    mixed = _aggregate_track_predictions(
        [
            {"prediction": "8", "raw_probability": 0.4},
            {"prediction": "9", "raw_probability": 0.8},
            {"prediction": "8", "raw_probability": 0.5},
        ]
    )
    assert mixed["prediction"] == "8"
    assert mixed["stable"] is False
    assert abs(mixed["modal_fraction"] - (2.0 / 3.0)) < 1e-12

    print("jersey_temporal_initial_validation_v110_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
