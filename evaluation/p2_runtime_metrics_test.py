from __future__ import annotations

from types import SimpleNamespace
import unittest

from core.court_state import CourtState
from services.p2.engine_v1 import P2EngineV1
from services.p2.metrics import P2MetricsCollector


class P2RuntimeMetricsTest(unittest.TestCase):
    def _frame(
        self,
        *,
        frame_id: int,
        p2: dict,
        valid_gameplay: bool = True,
        court_state: CourtState | None = None,
    ):
        return SimpleNamespace(
            frame_id=frame_id,
            timestamp=frame_id / 25.0,
            fps=25.0,
            valid_gameplay=valid_gameplay,
            court_state=court_state,
            diagnostics={"p2": p2},
        )

    def test_court_state_separates_gameplay_presence_from_analysis_quality(self):
        limited = CourtState(
            frame_id=1,
            timestamp_sec=0.04,
            mode="image_space_fallback",
            court_visible=False,
            analysis_eligible=False,
            eligibility_reason="court_not_visible",
            player_count=3,
        )
        self.assertTrue(limited.gameplay_present)
        self.assertEqual(limited.gameplay_status, "gameplay_limited")
        self.assertEqual(limited.gameplay_reason, "multiple_player_evidence")
        self.assertTrue(
            P2EngineV1._analysis_eligible(
                SimpleNamespace(valid_gameplay=False, court_state=limited)
            )
        )

        analysis = CourtState(
            frame_id=2,
            timestamp_sec=0.08,
            analysis_eligible=True,
            eligibility_reason="court_and_entities_consistent",
        )
        self.assertTrue(analysis.gameplay_present)
        self.assertEqual(analysis.gameplay_status, "gameplay_analysis")

        non_game = CourtState(
            frame_id=3,
            timestamp_sec=0.12,
            analysis_eligible=False,
            player_count=0,
        )
        self.assertFalse(non_game.gameplay_present)
        self.assertEqual(non_game.gameplay_status, "non_gameplay")
        self.assertFalse(
            P2EngineV1._analysis_eligible(
                SimpleNamespace(valid_gameplay=False, court_state=non_game)
            )
        )

    def test_exposes_lab_metrics_loose_closure_and_team_breakdown(self):
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
                        "current_team_id": "home",
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
                        },
                        {
                            "event_type": "pass_candidate",
                            "decision": "candidate",
                            "confidence": 0.76,
                            "actor_track_id": 7,
                            "secondary_track_id": 8,
                            "team_id": "home",
                            "attributes": {"flight_supported": True},
                        },
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
                        "current_team_id": "home",
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
                        {
                            "event_type": "shot_attempt",
                            "decision": "confirmed",
                            "confidence": 0.8,
                            "actor_track_id": 11,
                            "team_id": "away",
                            "attributes": {"outcome": "missed"},
                        },
                        {
                            "event_type": "rebound_candidate",
                            "decision": "candidate",
                            "confidence": 0.7,
                            "actor_track_id": 12,
                            "team_id": "away",
                            "attributes": {},
                        },
                        {
                            "event_type": "turnover_candidate",
                            "decision": "candidate",
                            "confidence": 0.62,
                            "actor_track_id": 13,
                            "team_id": None,
                            "attributes": {"flight_supported": False},
                        },
                    ],
                },
            )
        )

        # GAMEPLAY_LIMITED participates in P2 control metrics without entering the strict
        # metric-eligible P1 scope.
        limited = CourtState(
            frame_id=3,
            timestamp_sec=0.12,
            analysis_eligible=False,
            player_count=3,
        )
        collector.observe(
            self._frame(
                frame_id=3,
                valid_gameplay=False,
                court_state=limited,
                p2={
                    "engine_eligible": True,
                    "p1_state_is_not_mutated": True,
                    "possession": {
                        "control_state": "possessed",
                        "current_track_id": 7,
                        "current_team_id": "home",
                        "current_confidence": 0.7,
                        "flight": {},
                    },
                    "shot": {},
                    "temporal_reset": {"count": 1, "last_reason": "analysis_ineligible_gap"},
                    "events": [],
                },
            )
        )

        summary = collector.summary()
        self.assertEqual(summary["version"], "p2_metrics_v5")
        self.assertEqual(summary["metric_eligible_frames"], 2)
        self.assertEqual(summary["engine_eligible_frames"], 3)
        self.assertEqual(summary["scope"]["limited_gameplay_frames"], 1)
        self.assertEqual(summary["possession"]["possessed_frames"], 3)
        self.assertEqual(summary["possession"]["possessed_rate"], 100.0)
        self.assertEqual(summary["possession"]["handler_memory_rate"], 100.0)
        self.assertEqual(summary["transitions"]["loose_ball_started"], 1)
        self.assertEqual(summary["transitions"]["loose_ball_recovered"], 1)
        self.assertEqual(summary["transitions"]["loose_ball_open_intervals"], 0)
        self.assertEqual(summary["transitions"]["loose_ball_closure_rate"], 100.0)
        self.assertEqual(summary["shots"]["attempts"], 2)
        self.assertEqual(summary["shots"]["outcomes"], {"made": 1, "missed": 1})
        self.assertEqual(
            summary["events"]["attribution_by_event_type"]["shot_attempt"]["actor_rate"],
            100.0,
        )
        self.assertEqual(
            summary["events"]["attribution_by_event_type"]["shot_attempt"]["team_rate"],
            100.0,
        )
        self.assertEqual(summary["resets"]["reasons"], {"analysis_ineligible_gap": 1})
        self.assertEqual(summary["p1_state_not_mutated_rate"], 100.0)

        home = summary["teams"]["home"]
        away = summary["teams"]["away"]
        unattributed = summary["teams"]["unattributed"]
        self.assertEqual(home["possessed_frames"], 3)
        self.assertEqual(home["possession_share_rate"], 100.0)
        self.assertEqual(home["passes"], 1)
        self.assertEqual(home["passes_with_flight_evidence"], 1)
        self.assertEqual(home["shots"]["attempts"], 1)
        self.assertEqual(home["shots"]["outcomes"]["made"], 1)
        self.assertEqual(away["shots"]["attempts"], 1)
        self.assertEqual(away["shots"]["outcomes"]["missed"], 1)
        self.assertEqual(away["rebounds"], 1)
        self.assertEqual(unattributed["turnovers"], 1)
        self.assertEqual(unattributed["direct_turnovers"], 1)


if __name__ == "__main__":
    unittest.main()
