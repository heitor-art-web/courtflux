from __future__ import annotations

from typing import Any

from services.vision.ball_candidate_pipeline import (
    SmartBallCandidatePipeline,
    SmartBallPipelineConfig,
)
from services.vision.p0_association_detector import PerformanceDetector as P0PerformanceDetector


class PerformanceDetector(P0PerformanceDetector):
    """P0 detector with conservative smart candidate generation before association.

    Full-frame ball detections remain the primary observations. A same-checkpoint crop
    pass can verify/refine ambiguous candidates, and a temporal prior may request one
    additional crop search when the full-frame stream has no plausible observation.
    Temporal state never creates a ball by itself.

    Set ``BASKETBALL_SMART_BALL_PIPELINE=0`` to restore the previous P0 candidate path.
    ``BASKETBALL_BALL_REACQUIRE`` remains independent and defaults OFF.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.smart_ball_pipeline: SmartBallCandidatePipeline | None = None
        self.smart_ball_pipeline_enabled = self._env_bool(
            "BASKETBALL_SMART_BALL_PIPELINE",
            True,
        )

        if (
            self.smart_ball_pipeline_enabled
            and self.split_ball_detection
            and self.ball_candidate_model is not None
        ):
            config = SmartBallPipelineConfig(
                enabled=True,
                verify_enabled=self._env_bool("BASKETBALL_BALL_VERIFY", True),
                zoom_enabled=self._env_bool("BASKETBALL_BALL_SELECTIVE_ZOOM", True),
                crop_size=self._env_int("BASKETBALL_BALL_CROP_SIZE", 640, 128, 1280),
                verify_trigger_conf=self._env_float(
                    "BASKETBALL_BALL_VERIFY_TRIGGER_CONF", 0.60, 0.0, 1.0
                ),
                verify_max_candidates=self._env_int(
                    "BASKETBALL_BALL_VERIFY_MAX_CANDIDATES", 1, 0, 8
                ),
                verify_match_px=self._env_float(
                    "BASKETBALL_BALL_VERIFY_MATCH_PX", 80.0, 10.0, 320.0
                ),
                zoom_trigger_rank=self._env_float(
                    "BASKETBALL_BALL_ZOOM_TRIGGER_RANK", 0.55, 0.0, 1.0
                ),
                temporal_zoom_max_age=self._env_int(
                    "BASKETBALL_BALL_ZOOM_MAX_AGE", 5, 1, 30
                ),
                dedupe_center_px=self._env_float(
                    "BASKETBALL_BALL_DEDUPE_CENTER_PX", 28.0, 1.0, 160.0
                ),
                detector_weight=self._env_float(
                    "BASKETBALL_BALL_RANK_DETECTOR", 0.60, 0.0, 1.0
                ),
                verifier_weight=self._env_float(
                    "BASKETBALL_BALL_RANK_VERIFY", 0.20, 0.0, 1.0
                ),
                temporal_weight=self._env_float(
                    "BASKETBALL_BALL_RANK_TEMPORAL", 0.15, 0.0, 1.0
                ),
                geometry_weight=self._env_float(
                    "BASKETBALL_BALL_RANK_GEOMETRY", 0.05, 0.0, 1.0
                ),
            )
            self.smart_ball_pipeline = SmartBallCandidatePipeline(
                model=self.ball_candidate_model,
                ball_class_ids=self.ball_candidate_class_ids,
                conf=self.ball_conf_thresh,
                iou=self.iou,
                imgsz=self.imgsz,
                max_det=self.ball_candidate_max_det,
                device=self.device,
                config=config,
            )
            print(
                "🏀 Smart ball pipeline: on "
                f"verify={'on' if config.verify_enabled else 'off'} "
                f"selective_zoom={'on' if config.zoom_enabled else 'off'} "
                f"crop={config.crop_size}px "
                f"rank=({config.detector_weight:.2f},"
                f"{config.verifier_weight:.2f},"
                f"{config.temporal_weight:.2f},"
                f"{config.geometry_weight:.2f})",
                flush=True,
            )
        else:
            print("🏀 Smart ball pipeline: off", flush=True)

    def _tracker_config(self) -> dict[str, Any]:
        config = super()._tracker_config()
        pipeline = getattr(self, "smart_ball_pipeline", None)
        if pipeline is None:
            config["smart_candidate_pipeline"] = {
                "enabled": False,
            }
            return config

        smart = pipeline.config
        config["smart_candidate_pipeline"] = {
            "enabled": True,
            "verify_enabled": bool(smart.verify_enabled),
            "zoom_enabled": bool(smart.zoom_enabled),
            "crop_size": int(smart.crop_size),
            "verify_trigger_conf": round(float(smart.verify_trigger_conf), 6),
            "verify_max_candidates": int(smart.verify_max_candidates),
            "verify_match_px": round(float(smart.verify_match_px), 6),
            "zoom_trigger_rank": round(float(smart.zoom_trigger_rank), 6),
            "temporal_zoom_max_age": int(smart.temporal_zoom_max_age),
            "rank_weights": {
                "detector": round(float(pipeline.weights[0]), 6),
                "verifier": round(float(pipeline.weights[1]), 6),
                "temporal": round(float(pipeline.weights[2]), 6),
                "geometry": round(float(pipeline.weights[3]), 6),
            },
            "rollback_env": "BASKETBALL_SMART_BALL_PIPELINE=0",
        }
        return config

    def _candidate_trace(
        self,
        candidates: list[dict[str, Any]],
        frame_state: Any,
    ) -> list[dict[str, Any]]:
        traces = super()._candidate_trace(candidates, frame_state)
        for candidate, trace in zip(candidates, traces):
            detector_conf = float(
                candidate.get("detector_confidence", candidate.get("confidence", 0.0))
            )
            association_conf = float(candidate.get("confidence", detector_conf))
            trace["confidence"] = round(detector_conf, 6)
            trace["detector_confidence"] = round(detector_conf, 6)
            trace["association_confidence"] = round(association_conf, 6)
            trace["verification_score"] = (
                None
                if candidate.get("verification_score") is None
                else round(float(candidate["verification_score"]), 6)
            )
            trace["verified"] = candidate.get("verified")
            trace["rank_score"] = (
                None
                if candidate.get("rank_score") is None
                else round(float(candidate["rank_score"]), 6)
            )
            trace["temporal_score"] = (
                None
                if candidate.get("temporal_score") is None
                else round(float(candidate["temporal_score"]), 6)
            )
            trace["temporal_plausible"] = candidate.get("temporal_plausible")
            trace["geometry_score"] = (
                None
                if candidate.get("geometry_score") is None
                else round(float(candidate["geometry_score"]), 6)
            )
            trace["source"] = str(candidate.get("candidate_source", candidate.get("source", "full_frame")))
            trace["merged_sources"] = list(candidate.get("merged_sources") or [])
            trace["zoom_reason"] = candidate.get("zoom_reason")
            trace["crop_box"] = candidate.get("crop_box")
        return traces

    @staticmethod
    def _same_box(a: dict[str, Any], b: dict[str, Any]) -> bool:
        box_a = a.get("box")
        box_b = b.get("box")
        if not isinstance(box_a, (list, tuple)) or not isinstance(box_b, (list, tuple)):
            return False
        if len(box_a) != 4 or len(box_b) != 4:
            return False
        return all(abs(float(left) - float(right)) <= 1.0 for left, right in zip(box_a, box_b))

    def _preserve_full_frame_candidates(
        self,
        raw_candidates: list[dict[str, Any]],
        enhanced: list[dict[str, Any]],
        *,
        frame_state: Any,
    ) -> tuple[list[dict[str, Any]], int]:
        """Guarantee that smart processing cannot reduce full-frame candidate recall."""
        pipeline = self.smart_ball_pipeline
        if pipeline is None:
            return enhanced, 0

        restored = 0
        output = list(enhanced)
        frame_id = int(getattr(frame_state, "frame_id", 0))
        for raw in raw_candidates:
            present = any(
                str(candidate.get("source", "")) == "full_frame"
                and self._same_box(raw, candidate)
                for candidate in output
            )
            if present:
                continue
            item = dict(raw)
            item["source"] = "full_frame"
            item["detector_confidence"] = float(item.get("confidence", 0.0))
            item.setdefault("verification_score", None)
            output.append(
                pipeline._rank_candidate(
                    item,
                    tracker=self.ball_tracker,
                    frame_id=frame_id,
                    frame_shape=frame_state.frame.shape,
                )
            )
            restored += 1

        output.sort(key=lambda candidate: float(candidate.get("rank_score", 0.0)), reverse=True)
        return output, restored

    @staticmethod
    def _association_candidates(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        association_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            scored = dict(candidate)
            detector_conf = float(candidate.get("detector_confidence", candidate.get("confidence", 0.0)))
            scored["detector_confidence"] = detector_conf
            scored["candidate_source"] = str(candidate.get("source", "full_frame"))
            scored["confidence"] = float(candidate.get("rank_score", detector_conf))
            association_candidates.append(scored)
        return association_candidates

    def _select_ball(self, candidates: list[dict[str, Any]], frame_state: Any):
        pipeline = getattr(self, "smart_ball_pipeline", None)
        if pipeline is None:
            return super()._select_ball(candidates, frame_state)

        frame_id = int(getattr(frame_state, "frame_id", 0))
        enhanced, smart_diag = pipeline.enhance(
            frame_state.frame,
            candidates,
            frame_id=frame_id,
            tracker=self.ball_tracker,
        )
        enhanced, restored = self._preserve_full_frame_candidates(
            candidates,
            enhanced,
            frame_state=frame_state,
        )
        smart_diag["full_frame_candidates_restored"] = int(restored)
        smart_diag["output_candidate_count"] = len(enhanced)

        association_candidates = self._association_candidates(enhanced)
        ball, association = super()._select_ball(association_candidates, frame_state)
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
        return ball, association
