from __future__ import annotations

import unittest

from evaluation.p2_airborne_train import FEATURES, train
from services.p2.airborne_classifier_v1_1 import AirborneClassifierV1


class AirborneTwoStageTrainTest(unittest.TestCase):
    @staticmethod
    def _row(video: str, kind: str, signal: float) -> dict:
        features = {name: 0.0 for name in FEATURES}
        features["episode_duration_sec"] = 0.4
        features["flight_supported_any"] = 1.0
        features["flight_supported_fraction"] = 1.0
        features["raw_best_score_mean"] = signal
        features["raw_best_score_max"] = signal
        if kind == "continuity":
            stage1_label = "continuity"
            stage2_label = None
            features["receiver_evidence_fraction"] = 0.0
        elif kind == "pass_flight":
            stage1_label = "airborne"
            stage2_label = "pass_flight"
            features["receiver_evidence_any"] = 1.0
            features["receiver_evidence_fraction"] = 1.0
        else:
            stage1_label = "airborne"
            stage2_label = "shot_flight"
            features["above_fraction"] = 1.0
            features["shot_active_any"] = 1.0
        return {
            "video_id": video,
            "stage1_label": stage1_label,
            "stage2_label": stage2_label,
            "features": features,
            "prefixes": [
                {"frame_id": 0, "elapsed_sec": 0.2, "features": dict(features)},
                {"frame_id": 1, "elapsed_sec": 0.4, "features": dict(features)},
            ],
        }

    def test_train_emits_episode_two_stage_artifact_and_reports(self) -> None:
        rows = []
        for video_index, video in enumerate(("a", "b", "c")):
            rows.extend(
                [
                    self._row(video, "continuity", 0.1 + video_index * 0.01),
                    self._row(video, "pass_flight", 0.6 + video_index * 0.01),
                    self._row(video, "shot_flight", 0.9 + video_index * 0.01),
                ]
            )

        artifact, report = train(
            rows,
            learning_rate=0.05,
            epochs=30,
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
        self.assertIn("terminal_episode_end_to_end", aggregate)
        self.assertIn("causal_episode_end_to_end", aggregate)
        self.assertEqual(len(report["grouped_cross_validation"]["folds"]), 3)
        self.assertEqual(report["episodes"], 9)


if __name__ == "__main__":
    unittest.main()
