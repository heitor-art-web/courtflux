from __future__ import annotations

import unittest

from evaluation.p2_airborne_train_two_stage import FEATURES, train
from services.p2.airborne_classifier_two_stage_v1 import AirborneClassifierV1


class AirborneTwoStageTrainTest(unittest.TestCase):
    @staticmethod
    def _row(video: str, interval: int, label: str, signal: float) -> dict:
        features = {name: 0.0 for name in FEATURES}
        features["flight_supported"] = 1.0
        features["raw_best_score"] = signal
        features["shot_preflight_above"] = 1.0 if label == "shot_flight" else 0.0
        return {
            "video_id": video,
            "truth_interval_index": interval,
            "label": label,
            "features": features,
        }

    def test_train_emits_two_stage_artifact_and_reports(self) -> None:
        rows = []
        for video_index, video in enumerate(("a", "b", "c")):
            rows.extend(
                [
                    self._row(video, 0, "unresolved", 0.1 + video_index * 0.01),
                    self._row(video, 1, "pass_flight", 0.6 + video_index * 0.01),
                    self._row(video, 2, "shot_flight", 0.9 + video_index * 0.01),
                ]
            )

        artifact, report = train(
            rows,
            learning_rate=0.05,
            epochs=20,
            l2=0.001,
            minimum_probability=0.55,
        )

        self.assertEqual(artifact["version"], AirborneClassifierV1.VERSION)
        self.assertEqual(set(artifact["stages"]), {"airborne", "type"})
        self.assertEqual(
            tuple(artifact["stages"]["airborne"]["labels"]),
            AirborneClassifierV1.STAGE1_LABELS,
        )
        self.assertEqual(
            tuple(artifact["stages"]["type"]["labels"]),
            AirborneClassifierV1.STAGE2_LABELS,
        )
        aggregate = report["grouped_cross_validation"]["aggregate"]
        self.assertIn("stage1_airborne_vs_continuity", aggregate)
        self.assertIn("stage2_pass_vs_shot", aggregate)
        self.assertIn("end_to_end", aggregate)
        self.assertEqual(len(report["grouped_cross_validation"]["folds"]), 3)


if __name__ == "__main__":
    unittest.main()
