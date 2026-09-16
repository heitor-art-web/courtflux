from __future__ import annotations

from types import SimpleNamespace
import unittest

from services.p2.metrics import P2MetricsCollector


class P2RuntimeMetricsTest(unittest.TestCase):
    def _frame(self, *, frame_id: int, p2: dict, valid_gameplay: bool = True):
        return SimpleNamespace(
            frame_id=frame_id,
            timestamp=frame_id / 25.0,
            fps=25.0,
            valid_gameplay=valid_gameplay,
            diagnostics={"p2": p2},
        )

    def test_exposes_lab_metrics_and_loose_closure(self):
        collector = P2MetricsCollector()
        collector.observe(
            self._frame(
                frame_id=1,
                p2={
                    "engine_eligible": True,
                    "p1_state_is_not_mutated": True,
                    "possession": {
                        "control_state": "possessed",
                        "current_track_id": 7,
                        "current_confidence": 0.8,
                        "flight": {},
                    },
                    "shot": {},
                    "temporal_reset": {"count": 0},
                    "events": [
                        {
                            "event_type": "loose_ball_started",
                            "decision": "confirmed",
                            "confidence": 0.7,
                            "actor_track_id": 7,
                            "team_id": "home",
                            "attributes": {},
                        }
                    ],
                },
            )
        )
        collector.observe(
            self._frame(
                frame_id=2,
                p2={
                    "engine_eligible": True,
                    "p1_state_is_not_mutated": True,
                    "possession": {
                        "control_state": "possessed",
                        "current_track_id": 7,
                        "current_confidence": 0.9,
                        "flight": {},
                    },
                    "shot": {},
                    "temporal_reset": {"count": 1, "last_reason": "analysis_ineligible_gap"},
                    "events": [
                        {
                            "event_type": "loose_ball_recovered",
                            "decision": "confirmed",
                            "confidence": 0.9,
                            "actor_track_id": 7,
                            "secondary_track_id": 7,
                            "team_id": "home",
                            "attributes": {
                                "terminal_reason": "same_handler_recovered",
                                "recovered": True,
                            },
                        },
                        {
                            "event_type": "shot_attempt",
                            "decision": "confirmed",
                            "confidence": 0.85,
                            "actor_track_id": 7,
                            "team_id": "home",
                            "attributes": {"outcome": "made"},
                        },
                    ],
                },
            )
        )

        summary = collector.summary()
        self.assertEqual(summary["version"], "p2_metrics_v3")
        self.assertEqual(summary["possession"]["possessed_rate"], 100.0)
        self.assertEqual(summary["possession"]["handler_memory_rate"], 100.0)
        self.assertEqual(summary["transitions"]["loose_ball_started"], 1)
        self.assertEqual(summary["transitions"]["loose_ball_recovered"], 1)
        self.assertEqual(summary["transitions"]["loose_ball_open_intervals"], 0)
        self.assertEqual(summary["transitions"]["loose_ball_closure_rate"], 100.0)
        self.assertEqual(summary["shots"]["attempts"], 1)
        self.assertEqual(summary["shots"]["outcomes"], {"made": 1})
        self.assertEqual(summary["events"]["attribution_by_event_type"]["shot_attempt"]["actor_rate"], 100.0)
        self.assertEqual(summary["events"]["attribution_by_event_type"]["shot_attempt"]["team_rate"], 100.0)
        self.assertEqual(summary["resets"]["reasons"], {"analysis_ineligible_gap": 1})
        self.assertEqual(summary["p1_state_not_mutated_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
