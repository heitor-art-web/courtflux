from __future__ import annotations

import unittest
from types import SimpleNamespace

from services.p2.metrics import P2MetricsCollector


class P2RealVideoProbeTest(unittest.TestCase):
    @staticmethod
    def frame(
        *,
        track_id,
        frame_id=0,
        confidence=0.8,
        events=None,
        eligible=True,
        engine_eligible=None,
        include_p2=True,
        control_state="unknown",
        reset_count=0,
        reset_reason=None,
        fps=10.0,
    ):
        diagnostics = {}
        if include_p2:
            diagnostics["p2"] = {
                "p1_state_is_not_mutated": True,
                "engine_eligible": eligible if engine_eligible is None else engine_eligible,
                "possession": {
                    "control_state": control_state,
                    "current_track_id": track_id,
                    "current_team_id": None,
                    "current_confidence": confidence,
                    "lost_frames": 0,
                    "flight": {"active": False, "flight_supported": False},
                },
                "shot": {
                    "active": False,
                    "awaiting_rearm": False,
                    "rearm_clear_observed_frames": 0,
                },
                "temporal_reset": {
                    "count": reset_count,
                    "last_reason": reset_reason,
                },
                "events": list(events or []),
            }
        return SimpleNamespace(
            frame_id=frame_id,
            timestamp=frame_id / fps,
            fps=fps,
            valid_gameplay=eligible,
            court_state=None,
            diagnostics=diagnostics,
        )

    def test_aggregates_possession_events_and_unknown_shots(self):
        collector = P2MetricsCollector()
        collector.observe(self.frame(track_id=7, frame_id=1, confidence=0.9, control_state="possessed"))
        collector.observe(
            self.frame(
                track_id=12,
                frame_id=2,
                confidence=0.8,
                control_state="possessed",
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
                frame_id=3,
                control_state="shot_flight",
                events=[
                    {
                        "event_type": "shot_attempt",
                        "decision": "confirmed",
                        "confidence": 0.77,
                        "actor_track_id": 12,
                        "team_id": None,
                        "frame_start": 1,
                        "frame_end": 3,
                        "attributes": {"outcome": "unknown"},
                    }
                ],
            )
        )

        summary = collector.summary()
        self.assertEqual(summary["version"], "p2_metrics_v5")
        self.assertEqual(summary["frames_seen"], 3)
        self.assertEqual(summary["p2_processed_frames"], 3)
        self.assertEqual(summary["metric_eligible_frames"], 3)
        self.assertEqual(summary["engine_eligible_frames"], 3)
        self.assertEqual(summary["scope"]["limited_gameplay_frames"], 0)
        self.assertEqual(summary["possession"]["possessed_frames"], 2)
        self.assertEqual(summary["possession"]["possessed_rate"], 66.67)
        self.assertEqual(summary["possession"]["handler_memory_frames"], 2)
        self.assertEqual(summary["possession"]["handler_memory_rate"], 66.67)
        self.assertEqual(summary["possession"]["changes"], 1)
        self.assertEqual(summary["passes"], 1)
        self.assertEqual(summary["shots"]["attempts"], 1)
        self.assertEqual(summary["shots"]["outcomes"]["unknown"], 1)
        self.assertEqual(summary["shots"]["unknown_outcome_rate"], 100.0)
        self.assertEqual(summary["events"]["actor_attribution_rate"], 100.0)
        self.assertEqual(summary["events"]["team_attribution_rate"], 66.67)
        self.assertEqual(summary["events"]["attribution_by_event_type"]["shot_attempt"]["actor_rate"], 100.0)
        self.assertEqual(summary["events"]["attribution_by_event_type"]["shot_attempt"]["team_rate"], 0.0)
        self.assertEqual(summary["events"]["attribution_by_event_type"]["pass_candidate"]["actor_rate"], 100.0)
        self.assertEqual(summary["events"]["attribution_by_event_type"]["pass_candidate"]["team_rate"], 100.0)
        self.assertEqual(summary["control_state"]["counts"]["possessed"], 2)
        self.assertEqual(summary["control_state"]["counts"]["shot_flight"], 1)
        self.assertEqual(summary["p1_state_not_mutated_rate"], 100.0)
        self.assertEqual(len(collector.state_rows), 3)

    def test_handler_memory_is_not_counted_as_possession(self):
        collector = P2MetricsCollector()
        collector.observe(
            self.frame(
                track_id=4,
                frame_id=1,
                eligible=True,
                control_state="loose",
            )
        )
        collector.observe(
            self.frame(
                track_id=4,
                frame_id=2,
                eligible=True,
                control_state="possessed",
            )
        )

        summary = collector.summary()
        self.assertEqual(summary["possession"]["handler_memory_frames"], 2)
        self.assertEqual(summary["possession"]["handler_memory_rate"], 100.0)
        self.assertEqual(summary["possession"]["possessed_frames"], 1)
        self.assertEqual(summary["possession"]["possessed_rate"], 50.0)
        self.assertEqual(summary["control_state"]["rates"]["loose"], 50.0)
        self.assertEqual(summary["control_state"]["rates"]["possessed"], 50.0)

    def test_ineligible_frames_do_not_reduce_possession_rate(self):
        collector = P2MetricsCollector()
        collector.observe(self.frame(track_id=None, frame_id=1, eligible=False))
        collector.observe(
            self.frame(
                track_id=4,
                frame_id=2,
                confidence=0.75,
                eligible=True,
                control_state="possessed",
            )
        )

        summary = collector.summary()
        self.assertEqual(summary["metric_eligible_frames"], 1)
        self.assertEqual(summary["possession"]["possessed_rate"], 100.0)
        self.assertEqual(summary["possession"]["handler_memory_rate"], 100.0)

    def test_skipped_frame_does_not_reduce_p1_immutability_rate(self):
        collector = P2MetricsCollector()
        collector.observe(
            self.frame(
                track_id=None,
                frame_id=1,
                eligible=False,
                include_p2=False,
            )
        )
        collector.observe(
            self.frame(
                track_id=4,
                frame_id=2,
                confidence=0.75,
                eligible=True,
                control_state="possessed",
            )
        )

        summary = collector.summary()
        self.assertEqual(summary["frames_seen"], 2)
        self.assertEqual(summary["p2_processed_frames"], 1)
        self.assertEqual(summary["p1_state_not_mutated_rate"], 100.0)

    def test_transition_reset_and_rapid_unknown_retrigger_metrics(self):
        collector = P2MetricsCollector()
        collector.observe(
            self.frame(
                track_id=None,
                frame_id=20,
                control_state="pass_flight",
                reset_count=1,
                reset_reason="gameplay_absent_gap",
                events=[
                    {
                        "event_type": "pass_candidate",
                        "decision": "candidate",
                        "confidence": 0.8,
                        "actor_track_id": 1,
                        "secondary_track_id": 2,
                        "team_id": "Porto",
                        "attributes": {"flight_supported": True},
                    },
                    {
                        "event_type": "shot_attempt",
                        "decision": "candidate",
                        "confidence": 0.5,
                        "actor_track_id": 1,
                        "team_id": "Porto",
                        "frame_start": 1,
                        "frame_end": 10,
                        "attributes": {"outcome": "unknown"},
                    },
                    {
                        "event_type": "shot_attempt",
                        "decision": "confirmed",
                        "confidence": 0.8,
                        "actor_track_id": 2,
                        "team_id": "Porto",
                        "frame_start": 12,
                        "frame_end": 14,
                        "attributes": {"outcome": "made"},
                    },
                ],
            )
        )

        summary = collector.summary()
        self.assertEqual(summary["transitions"]["passes_with_flight_evidence"], 1)
        self.assertEqual(summary["resets"]["count"], 1)
        self.assertEqual(summary["resets"]["reasons"]["gameplay_absent_gap"], 1)
        self.assertEqual(summary["shots"]["rapid_unknown_retrigger_count"], 1)
        self.assertEqual(summary["shots"]["minimum_inter_shot_gap_frames"], 2)


if __name__ == "__main__":
    unittest.main()
