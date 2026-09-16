from __future__ import annotations

import gc
from time import perf_counter
from typing import Any

from services.vision.p0_association_detector import PerformanceDetector as SplitPerformanceDetector
from services.vision.shared_yolo_ball_backend import SharedYOLOBallCandidateBackend
from services.vision.single_forward_tracker import SingleForwardEntityTracker


class PerformanceDetector(SplitPerformanceDetector):
    """Run one full-frame YOLO forward and keep tracking downstream of detections.

    The single forward emits all detector classes. Player/referee detections are passed to
    BoT-SORT/ByteTrack through ``SingleForwardEntityTracker`` while raw basketball
    detections bypass that tracker and enter CourtFlux ball association directly. This
    preserves the key safety property of split-ball detection without a second full-frame
    YOLO inference.

    ``BASKETBALL_SINGLE_FORWARD=0`` restores the previous split-inference path.
    Non-YOLO ball challenger backends also retain the previous path because they represent
    an intentional second detector rather than duplicate inference with the same model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        requested = self._env_bool("BASKETBALL_SINGLE_FORWARD", True)
        backend_name = str(getattr(self, "ball_candidate_backend_name", "yolo")).lower()
        yolo_backend = backend_name in {"yolo", "ultralytics", "yolo_shared"}
        self.single_forward_enabled = bool(
            requested and self.split_ball_detection and yolo_backend
        )
        self.single_forward_tracker: SingleForwardEntityTracker | None = None

        if self.single_forward_enabled:
            # Replace the separately loaded same-checkpoint backend. Smart Ball crop
            # verification can reuse the primary model; full-frame ball candidates come
            # directly from the single raw prediction below.
            previous_backend = getattr(self, "ball_candidate_backend", None)
            try:
                shared = SharedYOLOBallCandidateBackend(
                    model=self.model,
                    device=self.device,
                    ball_predicate=self._is_ball_class,
                )
                self.ball_candidate_backend = shared
                self.ball_candidate_backend_name = shared.name
                self.ball_candidate_model = self.model
                self.ball_candidate_class_ids = list(shared.class_ids)
                if previous_backend is not None and previous_backend is not shared:
                    del previous_backend
                    gc.collect()

                self.single_forward_tracker = SingleForwardEntityTracker(
                    self.tracker,
                    device=self.device,
                )
            except Exception as exc:
                self.single_forward_enabled = False
                self.single_forward_tracker = None
                print(
                    "⚠️ Single-forward initialization failed; reverting to split inference: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        print(
            "🏀 Full-frame detector path: "
            f"{'single_forward' if self.single_forward_enabled else 'split_inference'} "
            f"rollback=BASKETBALL_SINGLE_FORWARD=0",
            flush=True,
        )

    def _tracked_class_ids_for_single_forward(self) -> set[int]:
        helper = getattr(self, "_tracked_entity_class_ids", None)
        if callable(helper):
            try:
                values = helper()
                if values:
                    return {int(value) for value in values}
            except Exception:
                pass
        return {int(value) for value in self.player_tracking_class_ids}

    def _bulk_raw_rows(self, result: Any) -> tuple[Any, list[dict[str, Any]]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return None, []
        cpu_boxes = boxes.cpu()
        if len(cpu_boxes) == 0:
            return cpu_boxes, []

        xyxy = cpu_boxes.xyxy.numpy()
        confs = cpu_boxes.conf.numpy()
        classes = cpu_boxes.cls.numpy()
        names = getattr(self.model, "names", {}) or {}
        rows: list[dict[str, Any]] = []
        for index, (coords, confidence, class_value) in enumerate(
            zip(xyxy, confs, classes)
        ):
            class_id = int(class_value)
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id)).lower()
            else:
                class_name = str(
                    names[class_id] if 0 <= class_id < len(names) else class_id
                ).lower()
            x1, y1, x2, y2 = [float(value) for value in coords[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append(
                {
                    "detection_index": int(index),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "box": [x1, y1, x2, y2],
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                }
            )
        return cpu_boxes, rows

    def _single_forward(self, frame: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
        timings: dict[str, float] = {}
        started = perf_counter()
        results = self.model.predict(
            frame,
            conf=self.inference_conf_thresh,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.ball_candidate_max_det,
            device=self.device,
            verbose=False,
        )
        timings["yolo_predict_full_frame"] = (perf_counter() - started) * 1000.0
        result = results[0] if results else None

        started = perf_counter()
        cpu_boxes, raw_rows = self._bulk_raw_rows(result) if result is not None else (None, [])
        timings["bulk_decode"] = (perf_counter() - started) * 1000.0

        tracked_ids = self._tracked_class_ids_for_single_forward()
        entity_indices = [
            row["detection_index"]
            for row in raw_rows
            if int(row["class_id"]) in tracked_ids
            and float(row["confidence"]) >= self.player_conf_thresh
        ]
        if cpu_boxes is None:
            entity_boxes = None
        elif entity_indices:
            entity_boxes = cpu_boxes[entity_indices]
        else:
            entity_boxes = cpu_boxes[:0]

        started = perf_counter()
        tracked_rows = (
            self.single_forward_tracker.update(entity_boxes, frame)
            if self.single_forward_tracker is not None and entity_boxes is not None
            else []
        )
        timings["entity_tracker"] = (perf_counter() - started) * 1000.0

        ball_candidates: list[dict[str, Any]] = []
        for row in raw_rows:
            if not self._is_ball_class(str(row["class_name"])):
                continue
            confidence = float(row["confidence"])
            if confidence < self.ball_conf_thresh:
                continue
            x1, y1, x2, y2 = [float(value) for value in row["box"]]
            ball_candidates.append(
                {
                    "box": [x1, y1, x2, y2],
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "confidence": confidence,
                    "detector_confidence": confidence,
                    "detector_backend": "yolo_single_forward",
                    "detector_model": self._resolve_model_path(self.model),
                    "detector_class_id": int(row["class_id"]),
                    "detector_label": str(row["class_name"]),
                    "source": "full_frame",
                    "interpolated": False,
                }
            )
        return tracked_rows, ball_candidates, raw_rows, timings

    def _parse_single_forward(
        self,
        frame_state: Any,
        tracked_rows: list[dict[str, Any]],
        ball_candidates: list[dict[str, Any]],
        raw_rows: list[dict[str, Any]],
    ) -> None:
        names = getattr(self.model, "names", {}) or {}
        is_referee = getattr(self, "_is_referee_class", lambda _name: False)
        referee_conf = float(getattr(self, "referee_conf_thresh", self.player_conf_thresh))

        players: list[dict[str, Any]] = []
        referees: list[dict[str, Any]] = []
        for tracked in tracked_rows:
            class_id = int(tracked.get("class_id", -1))
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id)).lower()
            else:
                class_name = str(
                    names[class_id] if 0 <= class_id < len(names) else class_id
                ).lower()
            confidence = float(tracked.get("confidence", 0.0))
            raw_box = tracked.get("box") or []
            if len(raw_box) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in raw_box]
            if x2 <= x1 or y2 <= y1:
                continue
            payload = {
                "track_id": int(tracked.get("track_id", -1)),
                "box": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "confidence": confidence,
                "class_name": class_name,
                "source": "yolo_single_forward+botsort",
            }
            if is_referee(class_name):
                if confidence >= referee_conf:
                    referees.append(payload)
                continue
            if self._is_player_class(class_name) and confidence >= self.player_conf_thresh:
                payload.update(
                    {
                        "class_name": "player",
                        "team": None,
                        "shirt_num": None,
                        "player_name": None,
                    }
                )
                players.append(payload)

        normalized_ball: list[dict[str, Any]] = []
        for candidate in ball_candidates:
            item = dict(candidate)
            raw_box = item.get("box") or []
            if len(raw_box) != 4:
                continue
            confidence = float(item.get("detector_confidence", item.get("confidence", 0.0)))
            if confidence < self.ball_conf_thresh:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in raw_box]
            if x2 <= x1 or y2 <= y1:
                continue
            item["box"] = [x1, y1, x2, y2]
            item["center"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            item["confidence"] = confidence
            item["detector_confidence"] = confidence
            normalized_ball.append(item)

        selected_ball, association = self._select_ball(normalized_ball, frame_state)
        ball_source = str(association.get("source") or "missing")
        direct_selected = selected_ball is not None and not bool(
            selected_ball.get("interpolated", False)
        )

        class_counts: dict[str, int] = {}
        for row in raw_rows:
            label = str(row.get("class_name") or "unknown")
            class_counts[label] = class_counts.get(label, 0) + 1
        raw_player_count = sum(
            1 for row in raw_rows if self._is_player_class(str(row.get("class_name") or ""))
        )
        raw_ref_count = sum(
            1 for row in raw_rows if is_referee(str(row.get("class_name") or ""))
        )
        ball_confidences = [
            float(item.get("detector_confidence", item.get("confidence", 0.0)))
            for item in normalized_ball
        ]
        p1_detections = [
            {
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "confidence": round(float(row["confidence"]), 6),
                "box": [round(float(value), 3) for value in row["box"]],
            }
            for row in raw_rows
        ]

        frame_state.players = players
        frame_state.referees = referees
        frame_state.ball = selected_ball
        frame_state.diagnostics.setdefault("analytics", {})["ball_source"] = ball_source
        backend_metadata = self.ball_candidate_backend_metadata()
        frame_state.diagnostics["detector"] = {
            "raw_total": len(raw_rows),
            "raw_class_counts": class_counts,
            "raw_player_candidates": raw_player_count,
            "raw_referee_candidates": raw_ref_count,
            "raw_ball_candidates": len(normalized_ball),
            "accepted_player_candidates": len(players),
            "accepted_referee_candidates": len(referees),
            "accepted_ball_candidates": len(normalized_ball),
            "raw_ball_max_conf": max(ball_confidences) if ball_confidences else None,
            "raw_ball_confidences": [round(value, 6) for value in ball_confidences],
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
            "ball_candidate_source": "yolo_single_forward_raw",
            "ball_detector_backend": backend_metadata,
            "split_ball_detection": True,
            "single_forward": True,
            "single_forward_policy": "one_yolo_predict_then_track_people_raw_ball_bypass",
            "p1_detections": p1_detections,
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
            "referee_conf": round(referee_conf, 6),
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

    def _process_single_forward(self, frame_state: Any):
        tracked_rows, ball_candidates, raw_rows, timings = self._single_forward(
            frame_state.frame
        )

        started = perf_counter()
        self._parse_single_forward(
            frame_state,
            tracked_rows,
            ball_candidates,
            raw_rows,
        )
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
        if self.single_forward_tracker is not None:
            diag["entity_tracker"] = self.single_forward_tracker.metadata()

        frame_id = int(getattr(frame_state, "frame_id", 0))
        if frame_id > 0 and frame_id % 100 == 0:
            locked = sum(
                1 for profile in self.player_db.values() if profile.get("identity_locked")
            )
            print(
                f"[PROGRESS] frame={frame_id} players={len(frame_state.players)} "
                f"refs={len(getattr(frame_state, 'referees', []) or [])} "
                f"ball={diag.get('ball_source', 'missing')} "
                f"single_forward=1 jersey_attempts={self.jersey_attempts_total} "
                f"identity_locks={locked}",
                flush=True,
            )
        return frame_state

    def process(self, frame_state: Any):
        if not self.single_forward_enabled:
            return super().process(frame_state)
        return self._process_single_forward(frame_state)
