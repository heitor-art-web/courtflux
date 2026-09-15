from __future__ import annotations

import unittest
from types import SimpleNamespace

from evaluation.p2_real_video_probe import P2MetricsCollector


class P2RealVideoProbeTest(unittest.TestCase):
    @staticmethod
    def frame(*, track_id, confidence=0.8, events=None, eligible=True):
        return SimpleNamespace(
            valid_gameplay=eligible,
            diagnostics={
                "p2": {
                    "p1_state_is_not_mutated": True,
                    "possession": {
                        "current_track_id": track_id,
                        "current_confidence": confidence,
                    },
                    "events": list(events or []),
                }
            },
        )

    def test_aggregates_possession_events_and_unknown_shots(self):
        collector = P2MetricsCollector()
        collector.observe(self.frame(track_id=7, confidence=0.9))
        collector.observe(
            self.frame(
                track_id=12,
                confidence=0.8,
                events=[
                    {
                        "event_type": "possession_change",
                        "decision": "confirmed",
                        "confidence": 0.84,
                        "actor_track_id": 12,
                        "team_id": "Porto",
                        "attributes": {},
                    },
                    {
                        "event_type": "pass_candidate",
                        "decision": "candidate",
                        "confidence": 0.72,
                        "actor_track_id": 12,
                        "team_id": "Porto",
                        "attributes": {},
                    },
                ],
            )
        )
        collector.observe(
            self.frame(
                track_id=None,
                events=[
                    {
                        "event_type": "shot_attempt",
                        "decision": "confirmed",
                        "confidence": 0.77,
                        "actor_track_id": 12,
                        "team_id": None,
                        "attributes": {"outcome": "unknown"},
                    }
                ],
            )
        )

        summary = collector.summary()
        self.assertEqual(summary["frames_seen"], 3)
        self.assertEqual(summary["metric_eligible_frames"], 3)
        self.assertEqual(summary["possession"]["locked_frames"], 2)
        self.assertEqual(summary["possession"]["coverage_rate"], 66.67)
        self.assertEqual(summary["possession"]["unknown_rate"], 33.33)
        self.assertEqual(summary["possession"]["changes"], 1)
        self.assertEqual(summary["passes"], 1)
        self.assertEqual(summary["shots"]["attempts"], 1)
        self.assertEqual(summary["shots"]["outcomes"]["unknown"], 1)
        self.assertEqual(summary["shots"]["unknown_outcome_rate"], 100.0)
        self.assertEqual(summary["events"]["actor_attribution_rate"], 100.0)
        self.assertEqual(summary["events"]["team_attribution_rate"], 66.67)
        self.assertEqual(summary["p1_state_not_mutated_rate"], 100.0)

    def test_ineligible_frames_do_not_reduce_possession_coverage(self):
        collector = P2MetricsCollector()
        collector.observe(self.frame(track_id=None, eligible=False))
        collector.observe(self.frame(track_id=4, confidence=0.75, eligible=True))

        summary = collector.summary()
        self.assertEqual(summary["metric_eligible_frames"], 1)
        self.assertEqual(summary["possession"]["coverage_rate"], 100.0)
        self.assertEqual(summary["possession"]["unknown_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
