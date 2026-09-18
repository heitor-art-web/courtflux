from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.p2_event_engine_test import ball, frame_state
from services.p2 import P2EngineV1
from services.p2.airborne_classifier_two_stage_v1 import AirborneClassifierV1


class AirborneClassifierV1Test(unittest.TestCase):
    @staticmethod
    def _p2(*, flight_active: bool = True) -> dict:
        return {
            "possession": {
                "candidate_score": 0.2,
                "raw_best_score": 0.3,
                "raw_best_distinct_from_flight_origin": True,
                "pass_flight_receiver_evidence": False,
                "flight": {
                    "active": flight_active,
                    "flight_supported": flight_active,
                    "flight_frames": 3,
                    "flight_direct_observations": 2,
                    "flight_displacement_player_heights": 0.8,
                },
            },
            "shot": {"active": False},
        }

    def _model_file(
        self,
        *,
        airborne_bias: float = 8.0,
        type_bias: float = -8.0,
    ) -> tempfile.NamedTemporaryFile:
        classifier = AirborneClassifierV1()
        n = len(classifier.FEATURE_NAMES)

        def stage(labels: tuple[str, str], bias: float) -> dict:
            return {
                "labels": list(labels),
                "feature_mean": [0.0] * n,
                "feature_scale": [1.0] * n,
                "weights": [0.0] * n,
                "bias": bias,
                "minimum_probability": 0.55,
            }

        payload = {
            "version": classifier.VERSION,
            "classes": list(classifier.CLASSES),
            "features": list(classifier.FEATURE_NAMES),
            "stages": {
                "airborne": stage(classifier.STAGE1_LABELS, airborne_bias),
                "type": stage(classifier.STAGE2_LABELS, type_bias),
            },
        }
        handle = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
        json.dump(payload, handle)
        handle.flush()
        return handle

    def test_without_model_emits_features_and_abstains(self) -> None:
        classifier = AirborneClassifierV1()
        state = frame_state(0, ball_state=ball(0, 150, 150))
        result = classifier.classify(frame_state=state, p2=self._p2())
        self.assertFalse(result["model_loaded"])
        self.assertEqual(result["label"], "unresolved")
        self.assertEqual(set(result["features"]), set(classifier.FEATURE_NAMES))

    def test_stage1_continuity_blocks_stage2(self) -> None:
        handle = self._model_file(airborne_bias=-8.0, type_bias=8.0)
        try:
            classifier = AirborneClassifierV1(handle.name)
            state = frame_state(0, ball_state=ball(0, 150, 150))
            result = classifier.classify(frame_state=state, p2=self._p2())
            self.assertEqual(result["label"], "unresolved")
            self.assertEqual(result["decision_stage"], "stage1_continuity")
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def test_stage2_pass_after_airborne(self) -> None:
        handle = self._model_file(airborne_bias=8.0, type_bias=-8.0)
        try:
            classifier = AirborneClassifierV1(handle.name)
            state = frame_state(0, ball_state=ball(0, 150, 150))
            result = classifier.classify(frame_state=state, p2=self._p2())
            self.assertEqual(result["label"], "pass_flight")
            self.assertEqual(result["decision_stage"], "stage2_pass")
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def test_stage2_shot_after_airborne(self) -> None:
        handle = self._model_file(airborne_bias=8.0, type_bias=8.0)
        try:
            classifier = AirborneClassifierV1(handle.name)
            state = frame_state(0, ball_state=ball(0, 150, 150))
            result = classifier.classify(frame_state=state, p2=self._p2())
            self.assertEqual(result["label"], "shot_flight")
            self.assertEqual(result["decision_stage"], "stage2_shot")
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def test_stage2_can_abstain_near_half_probability(self) -> None:
        handle = self._model_file(airborne_bias=8.0, type_bias=0.0)
        try:
            classifier = AirborneClassifierV1(handle.name)
            state = frame_state(0, ball_state=ball(0, 150, 150))
            result = classifier.classify(frame_state=state, p2=self._p2())
            self.assertEqual(result["label"], "unresolved")
            self.assertEqual(result["decision_stage"], "stage2_abstain")
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def test_loaded_model_classifies_only_when_flight_is_active(self) -> None:
        handle = self._model_file(airborne_bias=8.0, type_bias=-8.0)
        try:
            classifier = AirborneClassifierV1(handle.name)
            state = frame_state(0, ball_state=ball(0, 150, 150))
            active = classifier.classify(frame_state=state, p2=self._p2(flight_active=True))
            self.assertEqual(active["label"], "pass_flight")
            inactive = classifier.classify(frame_state=state, p2=self._p2(flight_active=False))
            self.assertEqual(inactive["label"], "unresolved")
            self.assertEqual(inactive["decision_stage"], "not_active_flight")
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def test_canonical_engine_accepts_two_stage_model_path(self) -> None:
        handle = self._model_file(airborne_bias=8.0, type_bias=8.0)
        try:
            engine = P2EngineV1(fps=10.0, airborne_model_path=handle.name)
            self.assertTrue(engine.airborne_classifier.model_loaded)
            self.assertEqual(engine.airborne_classifier.VERSION, AirborneClassifierV1.VERSION)
            self.assertEqual(engine.__class__.__name__, "LiveControlP2EngineV1")
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def test_invalid_stage_contract_is_rejected(self) -> None:
        classifier = AirborneClassifierV1()
        payload = {
            "version": classifier.VERSION,
            "classes": list(classifier.CLASSES),
            "features": list(classifier.FEATURE_NAMES),
            "stages": {},
        }
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                AirborneClassifierV1(path)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
