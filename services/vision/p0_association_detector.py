from __future__ import annotations

from dataclasses import asdict
from typing import Any

from services.vision.performance_detector import PerformanceDetector as BasePerformanceDetector


class PerformanceDetector(BasePerformanceDetector):
    """P0.2-instrumented production detector.

    The detection and association behaviour is intentionally unchanged except for one
    diagnostics-only source-name correction: a reset after a long gap is reported as
    ``detector_reinitialized`` rather than ``detector_reacquired``. Every accepted ball
    candidate is persisted with the score breakdown that existed *before* the tracker
    consumed the frame, so production association can be evaluated exactly against GT.
    """

    @staticmethod
    def _rounded_score(score: Any) -> dict[str, Any] | None:
        if score is None:
            return None
        payload = asdict(score)
        return {
            key: round(float(value), 6) if isinstance(value, float) else value
            for key, value in payload.items()
        }

    def _tracker_config(self) -> dict[str, Any]:
        tracker = self.ball_tracker
        return {
            "max_hold_frames": int(tracker.max_hold_frames),
            "reset_after_frames": int(tracker.reset_after_frames),
            "velocity_alpha": round(float(tracker.velocity_alpha), 6),
            "score_weights": {
                "confidence": round(float(tracker.score_confidence_weight), 6),
                "proximity": round(float(tracker.score_proximity_weight), 6),
                "size": round(float(tracker.score_size_weight), 6),
            },
            "gate": {
                "base": round(float(tracker.base_gate_norm), 6),
                "growth_per_frame": round(float(tracker.gate_growth_per_frame), 6),
                "max": round(float(tracker.max_gate_norm), 6),
                "high_confidence_threshold": round(
                    float(tracker.high_confidence_gate_threshold), 6
                ),
                "high_confidence_multiplier": round(
                    float(tracker.high_confidence_gate_multiplier), 6
                ),
            },
            "reacquire": {
                "enabled": bool(tracker.reacquire_enabled),
                "min_confidence": round(float(tracker.reacquire_min_confidence), 6),
                "confidence_margin": round(float(tracker.reacquire_confidence_margin), 6),
                "confirm_frames": int(tracker.reacquire_confirm_frames),
                "radius_norm": round(float(tracker.reacquire_radius_norm), 6),
                "min_size_similarity": round(
                    float(tracker.reacquire_min_size_similarity), 6
                ),
            },
        }

    @staticmethod
    def _trajectory_payload(ball: dict[str, Any] | None) -> dict[str, Any]:
        if ball is None:
            return {
                "trajectory_center": None,
                "trajectory_box": None,
                "trajectory_confidence": None,
                "trajectory_interpolated": False,
            }

        center = ball.get("center")
        box = ball.get("box")
        return {
            "trajectory_center": (
                [round(float(center[0]), 4), round(float(center[1]), 4)]
                if center is not None and len(center) >= 2
                else None
            ),
            "trajectory_box": (
                [int(value) for value in box]
                if isinstance(box, (list, tuple)) and len(box) == 4
                else None
            ),
            "trajectory_confidence": (
                round(float(ball.get("confidence", 0.0)), 6)
                if ball.get("confidence") is not None
                else None
            ),
            "trajectory_interpolated": bool(ball.get("interpolated", False)),
        }

    def _candidate_trace(
        self,
        candidates: list[dict[str, Any]],
        frame_state: Any,
    ) -> list[dict[str, Any]]:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        if self.ball_temporal_enabled:
            scores = [
                self.ball_tracker.inspect_candidate(
                    candidate,
                    frame_id=frame_id,
                    frame_shape=frame_state.frame.shape,
                )
                for candidate in candidates
            ]
        else:
            scores = [None for _ in candidates]

        traces: list[dict[str, Any]] = []
        for index, (candidate, score) in enumerate(zip(candidates, scores)):
            center = candidate.get("center") or (None, None)
            box = candidate.get("box") or []
            traces.append(
                {
                    "index": int(index),
                    "box": [int(value) for value in box] if len(box) == 4 else None,
                    "center": (
                        [round(float(center[0]), 4), round(float(center[1]), 4)]
                        if center[0] is not None and center[1] is not None
                        else None
                    ),
                    "confidence": round(float(candidate.get("confidence", 0.0)), 6),
                    "association": self._rounded_score(score),
                }
            )
        return traces

    @staticmethod
    def _mark_candidate_decisions(
        traces: list[dict[str, Any]],
        selected_index: int | None,
    ) -> None:
        for trace in traces:
            index = int(trace["index"])
            details = trace.get("association")
            if selected_index is not None and index == int(selected_index):
                trace["decision"] = "selected"
            elif details is None:
                trace["decision"] = "not_selected_no_temporal_score"
            elif bool(details.get("plausible")):
                trace["decision"] = "not_selected_lower_score"
            else:
                trace["decision"] = "rejected_outside_gate"

    def _select_ball(self, candidates: list[dict[str, Any]], frame_state: Any):
        frame_id = int(getattr(frame_state, "frame_id", 0))
        traces = self._candidate_trace(candidates, frame_state)

        if not self.ball_temporal_enabled:
            if not candidates:
                association = {
                    "source": "missing",
                    "selected_index": None,
                    "score": None,
                    "distance_norm": None,
                    "gate_norm": None,
                    "rejected_candidates": 0,
                    "hold_age_frames": 0,
                    "track_state": "DISABLED",
                    "reacquired": False,
                    "challenger_streak": 0,
                    "challenger_confidence": None,
                    "candidate_count": 0,
                    "candidates": traces,
                    "config": self._tracker_config(),
                    **self._trajectory_payload(None),
                }
                return None, association

            selected_index = max(
                range(len(candidates)),
                key=lambda idx: float(candidates[idx].get("confidence", 0.0)),
            )
            ball = dict(candidates[selected_index])
            ball["interpolated"] = False
            ball["source"] = "detector_highest_conf"
            self._mark_candidate_decisions(traces, selected_index)
            association = {
                "source": "detector_highest_conf",
                "selected_index": int(selected_index),
                "score": round(float(ball.get("confidence", 0.0)), 6),
                "distance_norm": None,
                "gate_norm": None,
                "rejected_candidates": 0,
                "hold_age_frames": 0,
                "track_state": "DISABLED",
                "reacquired": False,
                "challenger_streak": 0,
                "challenger_confidence": None,
                "candidate_count": len(candidates),
                "candidates": traces,
                "config": self._tracker_config(),
                **self._trajectory_payload(ball),
            }
            return ball, association

        decision = self.ball_tracker.select(
            candidates=candidates,
            frame_id=frame_id,
            frame_shape=frame_state.frame.shape,
        )

        source = str(decision.source)
        ball = decision.ball
        # The tracker historically labelled a post-reset highest-confidence choice as
        # a reacquisition whenever the feature flag was enabled. It was not challenger-
        # confirmed, so keep the behaviour but correct the diagnostics semantics.
        if source == "detector_reacquired" and not bool(decision.reacquired):
            source = "detector_reinitialized"
            if ball is not None:
                ball = dict(ball)
                ball["source"] = source

        self._mark_candidate_decisions(traces, decision.selected_index)
        association = {
            "source": source,
            "selected_index": decision.selected_index,
            "score": None if decision.score is None else round(float(decision.score), 6),
            "distance_norm": (
                None
                if decision.distance_norm is None
                else round(float(decision.distance_norm), 6)
            ),
            "gate_norm": (
                None if decision.gate_norm is None else round(float(decision.gate_norm), 6)
            ),
            "rejected_candidates": int(decision.rejected_candidates),
            "hold_age_frames": int(decision.hold_age_frames),
            "track_state": str(decision.track_state),
            "reacquired": bool(decision.reacquired),
            "challenger_streak": int(decision.challenger_streak),
            "challenger_confidence": (
                None
                if decision.challenger_confidence is None
                else round(float(decision.challenger_confidence), 6)
            ),
            "candidate_count": len(candidates),
            "candidates": traces,
            "config": self._tracker_config(),
            **self._trajectory_payload(ball),
        }
        return ball, association

    def _parse(self, frame_state: Any, results: Any):
        super()._parse(frame_state, results)
        detector = frame_state.diagnostics.setdefault("detector", {})
        association = detector.get("ball_association", {}) or {}
        detector["ball_tracker_config"] = self._tracker_config()
        detector["ball_reacquire_config"] = self._tracker_config()["reacquire"]
        detector.setdefault("ball_gate_config", {})["velocity_alpha"] = round(
            float(self.ball_tracker.velocity_alpha), 6
        )
        detector["trajectory_ball_center"] = association.get("trajectory_center")
        detector["trajectory_ball_box"] = association.get("trajectory_box")
        detector["trajectory_ball_conf"] = association.get("trajectory_confidence")
        detector["track_state"] = association.get("track_state")
        detector["reacquired"] = bool(association.get("reacquired", False))
        detector["challenger_streak"] = int(association.get("challenger_streak", 0) or 0)
        detector["challenger_confidence"] = association.get("challenger_confidence")
