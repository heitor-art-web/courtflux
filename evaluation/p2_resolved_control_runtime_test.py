from __future__ import annotations

import unittest
from types import SimpleNamespace

from services.p2.metrics import P2MetricsCollector


class P2ResolvedControlRuntimeTest(unittest.TestCase):
    @staticmethod
    def frame(
        frame_id: int,
        *,
        control_state: str,
        shot_active: bool = False,
        flight_active: bool = False,
        flight_frames: int = 0,
        receiver_evidence: bool = False,
        reset_count: int = 0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            frame_id=frame_id,
            timestamp=frame_id / 25.0,
            fps=25.0,
            valid_gameplay=True,
            court_state=None,
            diagnostics={
                "p2": {
                    "p1_state_is_not_mutated": True,
                    "engine_eligible": True,
                    "possession": {
                        "control_state": control_state,
                        "current_track_id": None,
                        "current_team_id": None,
                        "current_confidence": 0.0,
                        "lost_frames": 0,
                        "candidate_track_id": None,
                        "candidate_frames": 0,
                        "candidate_score": 0.0,
                        "candidate_peak_score": 0.0,
                        "candidate_threshold": 0.58,
                        "candidate_required_frames": 3,
                        "pass_flight_receiver_evidence": receiver_evidence,
                        "flight_classification_policy": "test",
                        "flight": {
                            "active": flight_active,
                            "origin_track_id": 7 if flight_active else None,
                            "flight_supported": flight_active,
                            "flight_frames": flight_frames,
                            "flight_direct_observations": flight_frames,
                            "flight_predicted_observations": 0,
                            "flight_displacement_px": 100.0 if flight_active else 0.0,
                            "flight_displacement_player_heights": (
                                1.0 if flight_active else 0.0
                            ),
                        },
                    },
                    "shot": {
                        "active": shot_active,
                        "awaiting_rearm": False,
                        "rearm_clear_observed_frames": 0,
                    },
                    "temporal_reset": {
                        "count": reset_count,
                        "last_reason": None,
                    },
                    "events": [],
                }
            },
        )

    def test_late_shot_activation_backfills_only_resolved_state(self) -> None:
        collector = P2MetricsCollector()
        collector.observe(self.frame(10, control_state="possessed"))
        collector.observe(
            self.frame(
                11,
                control_state="unknown",
                flight_active=True,
                flight_frames=1,
            )
        )
        collector.observe(
            self.frame(
                12,
                control_state="unknown",
                flight_active=True,
                flight_frames=2,
            )
        )
        collector.observe(
            self.frame(13, control_state="shot_flight", shot_active=True)
        )

        rows = {int(row["frame_id"]): row for row in collector.state_rows}
        self.assertEqual(rows[11]["control_state"], "unknown")
        self.assertEqual(rows[11]["live_control_state"], "unknown")
        self.assertEqual(rows[11]["resolved_control_state"], "shot_flight")
        self.assertTrue(rows[11]["resolved_control_provisional"])
        self.assertEqual(
            rows[11]["resolved_control_source"],
            "late_shot_activation_backfill",
        )
        self.assertEqual(rows[12]["resolved_control_state"], "shot_flight")
        self.assertEqual(rows[13]["resolved_control_state"], "shot_flight")
        self.assertTrue(rows[13]["resolved_shot_backfill_applied"])
        self.assertEqual(rows[13]["resolved_shot_backfill_flight_start_frame"], 11)

    def test_receiver_evidence_prevents_resolved_shot_backfill(self) -> None:
        collector = P2MetricsCollector()
        collector.observe(
            self.frame(
                20,
                control_state="unknown",
                flight_active=True,
                flight_frames=1,
                receiver_evidence=True,
            )
        )
        collector.observe(
            self.frame(21, control_state="shot_flight", shot_active=True)
        )

        rows = {int(row["frame_id"]): row for row in collector.state_rows}
        self.assertEqual(rows[20]["resolved_control_state"], "unknown")
        self.assertFalse(rows[20]["resolved_control_provisional"])
        self.assertFalse(rows[21]["resolved_shot_backfill_applied"])
        self.assertTrue(rows[21]["resolved_shot_backfill_receiver_evidence"])

    def test_backfill_does_not_cross_temporal_reset(self) -> None:
        collector = P2MetricsCollector()
        collector.observe(
            self.frame(
                30,
                control_state="unknown",
                flight_active=True,
                flight_frames=1,
                reset_count=0,
            )
        )
        collector.observe(
            self.frame(
                31,
                control_state="shot_flight",
                shot_active=True,
                reset_count=1,
            )
        )

        rows = {int(row["frame_id"]): row for row in collector.state_rows}
        self.assertEqual(rows[30]["resolved_control_state"], "unknown")
        self.assertNotIn("resolved_shot_backfill_applied", rows[31])


if __name__ == "__main__":
    unittest.main()
