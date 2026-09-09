from __future__ import annotations

import os
from typing import Any

from services.vision.ball_hypothesis_tracker import OCSortBallHypothesisTracker
from services.vision.p0_association_detector import PerformanceDetector as P0PerformanceDetector
from services.vision.smart_p0_detector import PerformanceDetector as SmartPerformanceDetector


class PerformanceDetector(SmartPerformanceDetector):
    """Smart P0 detector with optional OC-SORT hypothesis/identity layer.

    Default behaviour is unchanged. Set ``BASKETBALL_BALL_HYPOTHESIS_TRACKER=1`` to
    run OC-SORT+CIoU in parallel and expose lifecycle/identity diagnostics. Set
    ``BASKETBALL_BALL_HYPOTHESIS_OUTPUT=1`` only for an explicit experimental run;
    in that mode concrete ball output is emitted exclusively from an observed TRACKED
    hypothesis approved by the frozen temporal seed model. LOST/TENTATIVE hypotheses
    are never emitted.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ball_hypothesis_enabled = self._env_bool(
            "BASKETBALL_BALL_HYPOTHESIS_TRACKER",
            False,
        )
        self.ball_hypothesis_output_enabled = self._env_bool(
            "BASKETBALL_BALL_HYPOTHESIS_OUTPUT",
            False,
        )
        identity_model_path = os.getenv("BASKETBALL_BALL_IDENTITY_MODEL", "").strip()
        frame_rate = self._env_float(
            "BASKETBALL_BALL_HYPOTHESIS_FPS",
            30.0,
            1.0,
            240.0,
        )
        self.ball_hypothesis_tracker: OCSortBallHypothesisTracker | None = None
        if self.ball_hypothesis_enabled:
            self.ball_hypothesis_tracker = OCSortBallHypothesisTracker(
                frame_rate=frame_rate,
                identity_model_path=identity_model_path or None,
            )
            status = (
                "available"
                if self.ball_hypothesis_tracker.available
                else f"unavailable:{self.ball_hypothesis_tracker.unavailable_reason}"
            )
            model_status = (
                self.ball_hypothesis_tracker.identity_model.version
                if self.ball_hypothesis_tracker.identity_model is not None
                else "none"
            )
            print(
                "🏀 OC-SORT hypothesis tracker: on "
                f"status={status} identity_model={model_status} "
                f"output={'on' if self.ball_hypothesis_output_enabled else 'diagnostic-only'}",
                flush=True,
            )
        else:
            print("🏀 OC-SORT hypothesis tracker: off", flush=True)

    def _tracker_config(self) -> dict[str, Any]:
        config = super()._tracker_config()
        tracker = getattr(self, "ball_hypothesis_tracker", None)
        config["hypothesis_tracker"] = {
            "enabled": bool(getattr(self, "ball_hypothesis_enabled", False)),
            "output_enabled": bool(
                getattr(self, "ball_hypothesis_output_enabled", False)
            ),
            "rollback_env": "BASKETBALL_BALL_HYPOTHESIS_TRACKER=0",
            "output_rollback_env": "BASKETBALL_BALL_HYPOTHESIS_OUTPUT=0",
            "available": bool(tracker.available) if tracker is not None else False,
            "profile": tracker.profile if tracker is not None else None,
            "identity_model": (
                tracker._identity_model_diagnostics() if tracker is not None else {"loaded": False}
            ),
            "safety": {
                "lost_can_emit_concrete_output": False,
                "tentative_can_emit_concrete_output": False,
                "hard_reacquisition_required": False,
            },
        }
        return config

    def _hypothesis_update(
        self,
        candidates: list[dict[str, Any]],
        frame_state: Any,
    ) -> dict[str, Any]:
        tracker = getattr(self, "ball_hypothesis_tracker", None)
        if not bool(getattr(self, "ball_hypothesis_enabled", False)) or tracker is None:
            return {
                "enabled": False,
                "available": False,
                "hypotheses": [],
                "selected_seed": None,
            }
        return tracker.update(
            candidates,
            frame_id=int(getattr(frame_state, "frame_id", 0)),
        )

    def _seed_association(
        self,
        *,
        enhanced: list[dict[str, Any]],
        association_candidates: list[dict[str, Any]],
        hypothesis_diag: dict[str, Any],
        smart_diag: dict[str, Any],
        frame_state: Any,
    ):
        traces = self._candidate_trace(association_candidates, frame_state)
        selected_seed = hypothesis_diag.get("selected_seed")
        selected_index = None
        if isinstance(selected_seed, dict):
            try:
                candidate_index = int(selected_seed.get("candidate_index"))
                if 0 <= candidate_index < len(enhanced):
                    selected_index = candidate_index
            except (TypeError, ValueError):
                selected_index = None

        ball = None
        source = "missing"
        score = None
        track_state = "ABSTAIN"
        if selected_index is not None:
            selected = dict(enhanced[selected_index])
            candidate_source = str(selected.get("source", "full_frame"))
            selected["candidate_source"] = candidate_source
            selected["association_confidence"] = float(
                association_candidates[selected_index].get("confidence", 0.0)
            )
            selected["source"] = "ocsort_identity_seed"
            selected["interpolated"] = False
            ball = selected
            source = "ocsort_identity_seed"
            score = float(selected_seed.get("identity_probability", 0.0))
            track_state = "TRACKED"

        self._mark_candidate_decisions(traces, selected_index)
        association = {
            "source": source,
            "selected_index": selected_index,
            "score": None if score is None else round(score, 6),
            "distance_norm": None,
            "gate_norm": None,
            "rejected_candidates": 0,
            "hold_age_frames": 0,
            "track_state": track_state,
            "reacquired": False,
            "challenger_streak": 0,
            "challenger_confidence": None,
            "candidate_count": len(enhanced),
            "candidates": traces,
            "config": self._tracker_config(),
            "ball_hypotheses": hypothesis_diag,
            "smart_pipeline": smart_diag,
            "raw_candidate_count": int(smart_diag.get("raw_candidate_count", 0)),
            "enhanced_candidate_count": len(enhanced),
            **self._trajectory_payload(ball),
        }
        return ball, association

    def _select_ball(self, candidates: list[dict[str, Any]], frame_state: Any):
        # Exact legacy path when the new layer is disabled.
        if not bool(getattr(self, "ball_hypothesis_enabled", False)):
            return super()._select_ball(candidates, frame_state)

        pipeline = getattr(self, "smart_ball_pipeline", None)
        if pipeline is None:
            hypothesis_diag = self._hypothesis_update(candidates, frame_state)
            if bool(getattr(self, "ball_hypothesis_output_enabled", False)):
                smart_diag = {
                    "enabled": False,
                    "raw_candidate_count": len(candidates),
                }
                return self._seed_association(
                    enhanced=candidates,
                    association_candidates=candidates,
                    hypothesis_diag=hypothesis_diag,
                    smart_diag=smart_diag,
                    frame_state=frame_state,
                )
            ball, association = P0PerformanceDetector._select_ball(
                self,
                candidates,
                frame_state,
            )
            association["ball_hypotheses"] = hypothesis_diag
            return ball, association

        frame_id = int(getattr(frame_state, "frame_id", 0))
        enhanced, smart_diag = pipeline.enhance(
            frame_state.frame,
            candidates,
            frame_id=frame_id,
            tracker=self.ball_tracker,
        )
        from services.vision.ball_candidate_filter import filter_smart_candidates

        enhanced, filter_diag = filter_smart_candidates(
            enhanced,
            emit_refinements=self.smart_emit_refinements,
            zoom_min_geometry=self.smart_zoom_min_geometry,
            temporal_rescue_min_score=self.smart_temporal_rescue_min_score,
        )
        smart_diag.update(filter_diag)
        smart_diag["post_filter_candidate_count"] = len(enhanced)

        enhanced, restored = self._preserve_full_frame_candidates(
            candidates,
            enhanced,
            frame_state=frame_state,
        )
        smart_diag["full_frame_candidates_restored"] = int(restored)
        smart_diag["output_candidate_count"] = len(enhanced)
        smart_diag["raw_candidate_count"] = len(candidates)

        association_candidates = self._association_candidates(enhanced)
        hypothesis_diag = self._hypothesis_update(enhanced, frame_state)

        if bool(getattr(self, "ball_hypothesis_output_enabled", False)):
            return self._seed_association(
                enhanced=enhanced,
                association_candidates=association_candidates,
                hypothesis_diag=hypothesis_diag,
                smart_diag=smart_diag,
                frame_state=frame_state,
            )

        # Diagnostic-only mode: preserve the current Smart-v1 selection exactly.
        ball, association = P0PerformanceDetector._select_ball(
            self,
            association_candidates,
            frame_state,
        )
        selected_index = association.get("selected_index")
        if (
            selected_index is not None
            and 0 <= int(selected_index) < len(enhanced)
            and ball is not None
            and not bool(ball.get("interpolated", False))
        ):
            selected = dict(enhanced[int(selected_index)])
            candidate_source = str(selected.get("source", "full_frame"))
            selected["candidate_source"] = candidate_source
            selected["association_confidence"] = float(
                association_candidates[int(selected_index)].get("confidence", 0.0)
            )
            selected["source"] = str(association.get("source") or "detector")
            selected["interpolated"] = False
            ball = selected

        association["smart_pipeline"] = smart_diag
        association["raw_candidate_count"] = len(candidates)
        association["enhanced_candidate_count"] = len(enhanced)
        association["candidate_count"] = len(enhanced)
        association["ball_hypotheses"] = hypothesis_diag
        return ball, association
