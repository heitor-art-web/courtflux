from __future__ import annotations

import unittest

from evaluation.p2_airborne_build_dataset import FEATURES, _features_from_existing_row


class AirborneDatasetBuilderTest(unittest.TestCase):
    def test_existing_state_row_maps_complete_runtime_feature_contract(self) -> None:
        row = {
            "flight_supported": True,
            "flight_frames": 3,
            "flight_direct_observations": 2,
            "flight_displacement_player_heights": 0.8,
            "pass_flight_receiver_evidence": True,
            "raw_best_distinct_from_flight_origin": True,
            "raw_best_score": 0.4,
            "candidate_score": 0.2,
            "shot_active": False,
            "shot_preflight_available": True,
            "shot_preflight_direct_observed": True,
            "shot_preflight_near_x": False,
            "shot_preflight_above": True,
            "shot_preflight_ball_to_rim_distance_norm": 4.2,
            "shot_preflight_horizontal_distance_rim_widths": 3.1,
            "shot_preflight_relative_y_rim_heights": -1.2,
            "shot_preflight_delta_distance_norm": -0.4,
            "shot_preflight_delta_relative_y": -0.3,
            "shot_preflight_approach_streak": 2,
            "shot_preflight_rising_streak": 3,
        }
        features = _features_from_existing_row(row)
        self.assertEqual(tuple(features.keys()), tuple(FEATURES))
        self.assertEqual(features["shot_preflight_available"], 1.0)
        self.assertEqual(features["shot_preflight_above"], 1.0)
        self.assertAlmostEqual(features["shot_preflight_delta_distance_norm"], -0.4)

    def test_missing_historical_fields_are_neutral_not_key_errors(self) -> None:
        features = _features_from_existing_row({"flight_supported": True})
        self.assertEqual(tuple(features.keys()), tuple(FEATURES))
        self.assertEqual(features["flight_supported"], 1.0)
        self.assertEqual(features["shot_preflight_available"], 0.0)
        self.assertEqual(features["shot_preflight_ball_to_rim_distance_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
