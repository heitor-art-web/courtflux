from __future__ import annotations

import unittest

from evaluation.p2_airborne_classify_diagnose import diagnose


def frame(frame_id: int, x: float, y: float) -> dict:
    return {
        "frame_id": frame_id,
        "selected_ball": True,
        "selected_ball_center": [x, y],
        "court_state": {
            "basket_evidence": [
                {
                    "kind": "rim",
                    "bbox_px": [90.0, 90.0, 110.0, 100.0],
                }
            ]
        },
    }


def state(
    frame_id: int,
    *,
    flight_active: bool = True,
    flight_supported: bool = False,
    shot_active: bool = False,
) -> dict:
    return {
        "frame_id": frame_id,
        "flight_active": flight_active,
        "flight_supported": flight_supported,
        "shot_active": shot_active,
    }


class P2AirborneClassifyDiagnoseTest(unittest.TestCase):
    def test_rising_toward_rim_signal_is_diagnostic_only(self) -> None:
        states = [
            state(0),
            state(1, flight_supported=True),
            state(2, flight_supported=True),
        ]
        frames = [
            frame(0, 100.0, 150.0),
            frame(1, 100.0, 125.0),
            frame(2, 100.0, 110.0),
        ]
        ground_truth = {
            "control_intervals": [
                {"state": "shot_flight", "frame_start": 0, "frame_end": 2},
            ]
        }

        report = diagnose(states, frames, ground_truth)
        interval = report["intervals"][0]
        self.assertEqual(interval["early_shot_signal_frames"], 2)
        self.assertEqual(interval["supported_early_shot_signal_frames"], 2)
        self.assertEqual(interval["first_early_shot_signal_frame"], 1)
        self.assertEqual(report["contract"], "diagnostic_only_no_runtime_decision_changes")

    def test_moving_away_from_rim_does_not_signal_shot(self) -> None:
        states = [state(10), state(11, flight_supported=True)]
        frames = [
            frame(10, 100.0, 110.0),
            frame(11, 100.0, 130.0),
        ]
        ground_truth = {
            "control_intervals": [
                {"state": "pass_flight", "frame_start": 10, "frame_end": 11},
            ]
        }

        report = diagnose(states, frames, ground_truth)
        interval = report["intervals"][0]
        self.assertEqual(interval["early_shot_signal_frames"], 0)
        self.assertEqual(interval["supported_early_shot_signal_frames"], 0)


if __name__ == "__main__":
    unittest.main()
