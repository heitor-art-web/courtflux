from __future__ import annotations

from dataclasses import asdict
import os
from time import perf_counter
from typing import Any

from services.vision.ball_candidate_backend import create_ball_candidate_backend
from services.vision.performance_detector import PerformanceDetector as BasePerformanceDetector


class PerformanceDetector(BasePerformanceDetector):
    """P0.2-instrumented detector with independent raw ball candidate generation.

    Player identities still come from ``YOLO.track``. Ball candidates come from an
    independent detector backend so BoT-SORT cannot discard a valid ball detection before
    CourtFlux association sees it. The default backend remains the frozen YOLO path;
    ``BASKETBALL_BALL_DETECTOR_BACKEND=huggingface`` enables an HF challenger while
    preserving the downstream candidate contract and evaluation path.

    ``BASKETBALL_SPLIT_BALL_DETECTION=0`` restores the legacy track path for comparison.
    Controlled reacquisition remains independently gated by ``BASKETBALL_BALL_REACQUIRE``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.split_ball_detection = self._env_bool(
            "BASKETBALL_SPLIT_BALL_DETECTION",
            True,
        )
        self.ball_candidate_backend_name = os.getenv(
            "BASKETBALL_BALL_DETECTOR_BACKEND",
            "yolo",
        ).strip().lower() or "yolo"
        self.ball_candidate_backend: Any | None = None
        self.ball_candidate_model: Any | None = None
        self.ball_candidate_class_ids: list[int] = []
        self.player_tracking_class_ids = self._class_ids(self.model, self._is_player_class)
        self.ball_candidate_max_det = self._env_int(
            "BASKETBALL_BALL_MAX_DET",
            max(100, self.max_det),
            1,
            500,
        )

        if self.split_ball_detection:
            model_path = self._resolve_model_path(self.model)
            self.ball_candidate_backend = create_ball_candidate_backend(
                backend_name=self.ball_candidate_backend_name,
                yolo_model_path=model_path,
                device=self.device,
                ball_predicate=self._is_ball_class,
            )
            self.ball_candidate_backend_name = str(
                getattr(self.ball_candidate_backend, "name", self.ball_candidate_backend_name)
            )
            self.ball_candidate_model = getattr(
                self.ball_candidate_backend,
                "raw_model",
                None,
            )
            self.ball_candidate_class_ids = list(
                getattr(self.ball_candidate_backend, "class_ids", []) or []
            )

        metadata = self.ball_candidate_backend_metadata()
        print(
            "🏀 P0.1→P0.2 candidate handoff: "
            f"{self.ball_candidate_backend_name if self.split_ball_detection else 'legacy_yolo_track'} "
            f"ball_conf={self.ball_conf_thresh:.3f} "
            f"ball_max_det={self.ball_candidate_max_det} "
            f"model={metadata.get('model') if metadata else 'tracked_model'}",
            flush=True,
        )

    @staticmethod
    def _resolve_model_path(model: Any) -> str | None:
        candidates = [
            getattr(model, "ckpt_path", None),
            (getattr(model, "overrides", {}) or {}).get("model"),
        ]
        inner = getattr(model, "model", None)
        candidates.append(getattr(inner, "pt_path", None))
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return None

    @staticmethod
    def _class_ids(model: Any, predicate: Any) -> list[int]:
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            items = names.items()
        else:
            items = enumerate(names)
        return [
            int(class_id)
            for class_id, name in items
            if predicate(str(name).lower())
        ]

    def ball_candidate_backend_metadata(self) -> dict[str, Any]:
        backend = getattr(self, "ball_candidate_backend", None)
        if backend is None:
            return {
                "backend": "yolo_track",
                "model": self._resolve_model_path(self.model),
                "device": self.device,
            }
        metadata = getattr(backend, "metadata", None)
        if callable(metadata):
            return dict(metadata())
        return {
            "backend": str(getattr(backend, "name", "unknown")),
            "device": str(getattr(backend, "device", self.device)),
        }

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
            "candidate_handoff": {
                "split_ball_detection": bool(
                    getattr(self, "split_ball_detection", False)
                ),
                "source": (
                    str(getattr(self, "ball_candidate_backend_name", "yolo"))
                    if getattr(self, "split_ball_detection", False)
                    else "yolo_track"
                ),
                "ball_conf": round(float(self.ball_conf_thresh), 6),
                "ball_max_det": int(
                    getattr(self, "ball_candidate_max_det", self.max_det)
                ),
                "ball_class_ids": list(
                    getattr(self, "ball_candidate_class_ids", [])
                ),
                "backend": self.ball_candidate_backend_metadata(),
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
            trace = {
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
            for key in (
                "detector_backend",
                "detector_model",
                "detector_class_id",
                "detector_label",
            ):
                if candidate.get(key) is not None:
                    trace[key] = candidate.get(key)
            traces.append(trace)
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

        previous_frame_id = self.ball_tracker.last_frame_id
        reset_reinitialization = (
            previous_frame_id is not None
            and frame_id - int(previous_frame_id) > self.ball_tracker.reset_after_frames
        )
        decision = self.ball_tracker.select(
            candidates=candidates,
            frame_id=frame_id,
            frame_shape=frame_state.frame.shape,
        )

        source = str(decision.source)
        ball = decision.ball
        reacquired = bool(decision.reacquired)
        if source == "detector_reacquired" and reset_reinitialization:
            source = "detector_reinitialized"
            reacquired = False
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
            "reacquired": reacquired,
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

    def _track(self, frame: Any):
        """Track players only when split ball detection is active."""
        if not self.split_ball_detection:
            return super()._track(frame)

        kwargs: dict[str, Any] = {
            "persist": True,
            "tracker": self.tracker,
            "conf": self.player_conf_thresh,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "device": self.device,
            "verbose": False,
        }
        if self.player_tracking_class_ids:
            kwargs["classes"] = self.player_tracking_class_ids
        return self.model.track(frame, **kwargs)

    def _predict_ball_candidates(self, frame: Any) -> list[dict[str, Any]] | None:
        backend = getattr(self, "ball_candidate_backend", None)
        if not self.split_ball_detection or backend is None:
            return None
        backend_device = str(getattr(backend, "device", self.device))
        return backend.predict(
            frame,
            conf=self.ball_conf_thresh,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.ball_candidate_max_det,
            device=backend_device,
        )

    def _legacy_tracked_ball_candidates(self, tracked_results: Any) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if not tracked_results or tracked_results[0].boxes is None:
            return candidates
        for box in tracked_results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = str(self.model.names.get(cls_id, cls_id)).lower()
            if not self._is_ball_class(cls_name):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if conf < self.ball_conf_thresh or x2 <= x1 or y2 <= y1:
                continue
            candidates.append(
                {
                    "box": [x1, y1, x2, y2],
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "confidence": conf,
                    "detector_confidence": conf,
                    "detector_backend": "yolo_track",
                    "detector_class_id": cls_id,
                    "detector_label": str(self.model.names.get(cls_id, cls_id)),
                    "source": "full_frame",
                    "interpolated": False,
                }
            )
        return candidates

    def _parse(self, frame_state: Any, tracked_results: Any, ball_results: Any = None):
        players: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {}
        raw_total = 0
        raw_players = 0
        accepted_player_candidates = 0

        if tracked_results and tracked_results[0].boxes is not None:
            for box in tracked_results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = str(self.model.names.get(cls_id, cls_id)).lower()
                raw_total += 1
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                if not self._is_player_class(cls_name):
                    continue

                raw_players += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if conf < self.player_conf_thresh or x2 <= x1 or y2 <= y1:
                    continue

                track_id = int(box.id[0]) if box.id is not None else -1
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                accepted_player_candidates += 1
                players.append(
                    {
                        "track_id": track_id,
                        "box": [x1, y1, x2, y2],
                        "center": center,
                        "confidence": conf,
                        "team": None,
                        "shirt_num": None,
                        "player_name": None,
                    }
                )

        ball_candidates = (
            [dict(candidate) for candidate in ball_results]
            if ball_results is not None
            else self._legacy_tracked_ball_candidates(tracked_results)
        )
        raw_balls = len(ball_candidates)
        raw_ball_confidences = [
            float(candidate.get("detector_confidence", candidate.get("confidence", 0.0)))
            for candidate in ball_candidates
        ]
        raw_ball_max_conf = max(raw_ball_confidences) if raw_ball_confidences else None

        accepted_candidates: list[dict[str, Any]] = []
        for candidate in ball_candidates:
            item = dict(candidate)
            box = item.get("box") or []
            confidence = float(
                item.get("detector_confidence", item.get("confidence", 0.0))
            )
            if len(box) != 4 or confidence < self.ball_conf_thresh:
                continue
            x1, y1, x2, y2 = [int(value) for value in box]
            if x2 <= x1 or y2 <= y1:
                continue
            item["box"] = [x1, y1, x2, y2]
            item["center"] = item.get("center") or (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            )
            item["confidence"] = confidence
            item["detector_confidence"] = confidence
            item.setdefault("source", "full_frame")
            item.setdefault("interpolated", False)
            accepted_candidates.append(item)
        ball_candidates = accepted_candidates
        accepted_ball_candidates = len(ball_candidates)

        selected_ball, association = self._select_ball(ball_candidates, frame_state)
        ball_source = str(association.get("source") or "missing")
        direct_selected = selected_ball is not None and not bool(
            selected_ball.get("interpolated")
        )

        frame_state.players = players
        frame_state.ball = selected_ball
        frame_state.diagnostics.setdefault("analytics", {})["ball_source"] = ball_source
        backend_metadata = self.ball_candidate_backend_metadata()
        detector = {
            "raw_total": raw_total,
            "raw_class_counts": class_counts,
            "raw_player_candidates": raw_players,
            "raw_ball_candidates": raw_balls,
            "accepted_player_candidates": accepted_player_candidates,
            "accepted_ball_candidates": accepted_ball_candidates,
            "raw_ball_max_conf": raw_ball_max_conf,
            "raw_ball_confidences": [round(value, 6) for value in raw_ball_confidences],
            "rejected_outside_court_total": 0,
            "rejected_outside_court_by_class": {},
            "parsed_players": len(players),
            "selected_ball": direct_selected,
            "selected_ball_conf": (
                float(selected_ball.get("confidence", 0.0)) if direct_selected else None
            ),
            "selected_ball_center": (
                list(selected_ball["center"]) if direct_selected else None
            ),
            "ball_source": ball_source,
            "ball_temporal_enabled": self.ball_temporal_enabled,
            "ball_candidate_source": (
                str(backend_metadata.get("backend", self.ball_candidate_backend_name))
                if ball_results is not None
                else "yolo_track"
            ),
            "ball_detector_backend": backend_metadata,
            "split_ball_detection": bool(self.split_ball_detection),
            "ball_predict_max_det": int(self.ball_candidate_max_det),
            "ball_predict_class_ids": list(self.ball_candidate_class_ids),
            "ball_association": association,
            "ball_score_weights": {
                "confidence": round(self.ball_tracker.score_confidence_weight, 6),
                "proximity": round(self.ball_tracker.score_proximity_weight, 6),
                "size": round(self.ball_tracker.score_size_weight, 6),
            },
            "ball_gate_config": {
                "base": round(self.ball_tracker.base_gate_norm, 6),
                "growth_per_frame": round(self.ball_tracker.gate_growth_per_frame, 6),
                "max": round(self.ball_tracker.max_gate_norm, 6),
                "high_confidence_threshold": round(
                    self.ball_tracker.high_confidence_gate_threshold, 6
                ),
                "high_confidence_multiplier": round(
                    self.ball_tracker.high_confidence_gate_multiplier, 6
                ),
                "velocity_alpha": round(float(self.ball_tracker.velocity_alpha), 6),
            },
            "ball_tracker_config": self._tracker_config(),
            "ball_reacquire_config": self._tracker_config()["reacquire"],
            "trajectory_ball_center": association.get("trajectory_center"),
            "trajectory_ball_box": association.get("trajectory_box"),
            "trajectory_ball_conf": association.get("trajectory_confidence"),
            "track_state": association.get("track_state"),
            "reacquired": bool(association.get("reacquired", False)),
            "challenger_streak": int(association.get("challenger_streak", 0) or 0),
            "challenger_confidence": association.get("challenger_confidence"),
            "inference_conf": self.inference_conf_thresh,
            "player_conf": self.player_conf_thresh,
            "player_track_conf": self.player_conf_thresh,
            "ball_conf": self.ball_conf_thresh,
            "ball_predict_conf": self.ball_conf_thresh,
            "detector_conf": self.detector_conf,
            "imgsz": self.imgsz,
            "iou": self.iou,
            "tracker": self.tracker,
            "device": self.device,
            "detection_only": True,
            "jersey_enabled": self.enable_jersey,
            "jersey_interval_frames": self.jersey_interval_frames,
        }
        frame_state.diagnostics["detector"] = detector

    def process(self, frame_state: Any):
        timings: dict[str, float] = {}

        started = perf_counter()
        tracked_results = self._track(frame_state.frame)
        timings["yolo_track_players"] = (perf_counter() - started) * 1000.0

        ball_results = None
        if self.split_ball_detection:
            started = perf_counter()
            ball_results = self._predict_ball_candidates(frame_state.frame)
            backend = getattr(self, "ball_candidate_backend", None)
            timing_name = str(
                getattr(backend, "timing_name", "ball_candidate_inference")
            )
            timings[timing_name] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        self._parse(frame_state, tracked_results, ball_results)
        timings["parse"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        self._assign_teams(frame_state)
        timings["team_classification"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        attempts, successes, predictions = self._jerseys(frame_state)
        jersey_ms = (perf_counter() - started) * 1000.0
        timings["jersey_recognition"] = jersey_ms
        self.jersey_attempts_total += attempts
        self.jersey_successes_total += successes
        self.jersey_time_ms_total += jersey_ms

        diag = frame_state.diagnostics.setdefault("detector", {})
        diag["timing_ms"] = {name: round(ms, 4) for name, ms in timings.items()}
        diag["jersey_attempts"] = attempts
        diag["jersey_successes"] = successes
        diag["jersey_predictions"] = predictions
        diag["jersey_attempts_total"] = self.jersey_attempts_total
        diag["jersey_successes_total"] = self.jersey_successes_total

        frame_id = int(getattr(frame_state, "frame_id", 0))
        if frame_id > 0 and frame_id % 100 == 0:
            locked = sum(
                1 for profile in self.player_db.values() if profile.get("identity_locked")
            )
            ball_source = str(diag.get("ball_source") or "missing")
            print(
                f"[PROGRESS] frame={frame_id} players={len(frame_state.players)} "
                f"ball={ball_source} "
                f"ball_candidates={diag.get('accepted_ball_candidates', 0)} "
                f"candidate_source={diag.get('ball_candidate_source')} "
                f"jersey_attempts={self.jersey_attempts_total} "
                f"identity_locks={locked} "
                f"jersey_time={self.jersey_time_ms_total / 1000.0:.1f}s",
                flush=True,
            )

        return frame_state
