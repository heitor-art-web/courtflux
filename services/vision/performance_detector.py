from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import numpy as np
from ultralytics import YOLO


class PerformanceDetector:
    """Lean detector/tracker path for results-only performance experiments.

    Includes only YOLO tracking, player parsing, team classification, ball selection,
    and optional throttled jersey recognition. It deliberately does not perform court
    filtering, backboard/rim work, drawing, or tactical analytics.
    """

    def __init__(
        self,
        model: YOLO,
        visual_analyzer: Any,
        profiling_path: str,
        device: str = "mps",
        detector_conf: float = 0.10,
        imgsz: int = 640,
        iou: float = 0.5,
        tracker: str = "botsort.yaml",
        max_det: int = 40,
        enable_jersey: bool = False,
        jersey_interval_frames: int = 12,
    ):
        self.model = model
        self.visual = visual_analyzer
        self.device = device
        self.detector_conf = float(detector_conf)
        self.imgsz = int(imgsz)
        self.iou = float(iou)
        self.tracker = tracker
        self.max_det = int(max_det)
        self.ball_conf_thresh = max(0.01, min(self.detector_conf, 0.10))
        self.enable_jersey = bool(enable_jersey)
        self.jersey_interval_frames = max(1, int(jersey_interval_frames))
        self.player_db: dict[int, dict[str, Any]] = {}
        self.jersey_attempts_total = 0
        self.jersey_successes_total = 0
        self.jersey_time_ms_total = 0.0

        # Keep the config load explicit so every benchmark still validates its
        # distinction context even when jersey inference is disabled.
        with open(profiling_path, "r", encoding="utf-8") as fh:
            self.match_config = json.load(fh)

    @staticmethod
    def _is_player_class(name: str) -> bool:
        name = name.lower()
        return "person" in name or "player" in name

    @staticmethod
    def _is_ball_class(name: str) -> bool:
        name = name.lower()
        # Keep current broad semantics for comparability with previous beta runs.
        return "ball" in name or "basketball" in name or "sports ball" in name

    def _track(self, frame: np.ndarray):
        return self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.detector_conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

    def _parse(self, frame_state, results):
        players = []
        best_ball = None
        best_ball_conf = 0.0
        class_counts: dict[str, int] = {}
        raw_total = 0
        raw_players = 0
        raw_balls = 0
        raw_ball_max_conf = None

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = str(self.model.names.get(cls_id, cls_id)).lower()
                raw_total += 1
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                is_player = self._is_player_class(cls_name)
                is_ball = self._is_ball_class(cls_name)
                if is_player:
                    raw_players += 1
                if is_ball:
                    raw_balls += 1
                    raw_ball_max_conf = conf if raw_ball_max_conf is None else max(raw_ball_max_conf, conf)

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else -1
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if is_player and x2 > x1 and y2 > y1:
                    players.append(
                        {
                            "track_id": track_id,
                            "box": [x1, y1, x2, y2],
                            "center": center,
                            "confidence": conf,
                            "team": None,
                            "shirt_num": None,
                        }
                    )
                elif is_ball and conf >= self.ball_conf_thresh and conf > best_ball_conf:
                    best_ball_conf = conf
                    best_ball = {
                        "box": [x1, y1, x2, y2],
                        "center": center,
                        "confidence": conf,
                        "interpolated": False,
                    }

        frame_state.players = players
        frame_state.ball = best_ball
        frame_state.diagnostics["detector"] = {
            "raw_total": raw_total,
            "raw_class_counts": class_counts,
            "raw_player_candidates": raw_players,
            "raw_ball_candidates": raw_balls,
            "raw_ball_max_conf": raw_ball_max_conf,
            "rejected_outside_court_total": 0,
            "rejected_outside_court_by_class": {},
            "parsed_players": len(players),
            "selected_ball": best_ball is not None,
            "selected_ball_conf": best_ball_conf if best_ball is not None else None,
            "detector_conf": self.detector_conf,
            "imgsz": self.imgsz,
            "iou": self.iou,
            "tracker": self.tracker,
            "device": self.device,
            "detection_only": True,
            "jersey_enabled": self.enable_jersey,
            "jersey_interval_frames": self.jersey_interval_frames,
        }

    def _assign_teams(self, frame_state):
        frame = frame_state.frame
        for player in frame_state.players:
            x1, y1, x2, y2 = player["box"]
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            track_id = player["track_id"]
            if track_id != -1:
                profile = self.player_db.setdefault(
                    track_id,
                    {"team": None, "shirt_num": None, "last_jersey_attempt": -10**9},
                )
                if not profile.get("team"):
                    profile["team"] = self.visual.get_team_color(crop)
                player["team"] = profile.get("team")
                if self.enable_jersey and hasattr(self.visual, "observe_track"):
                    self.visual.observe_track(crop, track_id)
            else:
                player["team"] = self.visual.get_team_color(crop)

    def _jerseys(self, frame_state):
        attempts = 0
        successes = 0
        if not self.enable_jersey:
            return attempts, successes

        frame_id = int(getattr(frame_state, "frame_id", 0))
        frame = frame_state.frame
        for player in frame_state.players:
            track_id = player["track_id"]
            if track_id == -1:
                continue
            profile = self.player_db.setdefault(
                track_id,
                {"team": None, "shirt_num": None, "last_jersey_attempt": -10**9},
            )
            player["shirt_num"] = profile.get("shirt_num")
            if profile.get("shirt_num"):
                continue
            if hasattr(self.visual, "ready") and not self.visual.ready(track_id):
                continue
            if frame_id - int(profile.get("last_jersey_attempt", -10**9)) < self.jersey_interval_frames:
                continue

            x1, y1, x2, y2 = player["box"]
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            profile["last_jersey_attempt"] = frame_id
            attempts += 1
            player["ocr_attempted"] = True
            num = self.visual.get_shirt_number(crop, track_id=track_id, team_name=player.get("team"))
            if num:
                profile["shirt_num"] = num
                player["shirt_num"] = num
                player["ocr_success"] = True
                successes += 1
        return attempts, successes

    def process(self, frame_state):
        timings: dict[str, float] = {}

        started = perf_counter()
        results = self._track(frame_state.frame)
        timings["yolo_track"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        self._parse(frame_state, results)
        timings["parse"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        self._assign_teams(frame_state)
        timings["team_classification"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        attempts, successes = self._jerseys(frame_state)
        jersey_ms = (perf_counter() - started) * 1000.0
        timings["jersey_recognition"] = jersey_ms
        self.jersey_attempts_total += attempts
        self.jersey_successes_total += successes
        self.jersey_time_ms_total += jersey_ms

        diag = frame_state.diagnostics.setdefault("detector", {})
        diag["timing_ms"] = {name: round(ms, 4) for name, ms in timings.items()}
        diag["jersey_attempts"] = attempts
        diag["jersey_successes"] = successes
        diag["jersey_attempts_total"] = self.jersey_attempts_total
        diag["jersey_successes_total"] = self.jersey_successes_total

        frame_id = int(getattr(frame_state, "frame_id", 0))
        if frame_id > 0 and frame_id % 100 == 0:
            print(
                f"[PROGRESS] frame={frame_id} players={len(frame_state.players)} "
                f"ball={'yes' if frame_state.ball is not None else 'no'} "
                f"jersey_attempts={self.jersey_attempts_total} "
                f"jersey_hits={self.jersey_successes_total} "
                f"jersey_time={self.jersey_time_ms_total / 1000.0:.1f}s",
                flush=True,
            )

        return frame_state
