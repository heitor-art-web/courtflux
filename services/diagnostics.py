from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class RunDiagnostics:
    """Write one comparable diagnostic bundle for each beta analysis run."""

    def __init__(
        self,
        root: Path,
        video_source: str,
        start_time: float,
        end_time: Optional[float],
        model_path: str,
        detector_conf: float,
        imgsz: int,
        device: str,
        tracker: str,
    ):
        self.started = time.time()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = root / "evaluation" / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.run_dir / "frames.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.meta_path = self.run_dir / "run_config.json"

        self.config = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "video_source": video_source,
            "start_time_sec": start_time,
            "end_time_sec": end_time,
            "model": model_path,
            "detector_conf": detector_conf,
            "imgsz": imgsz,
            "device": device,
            "tracker": tracker,
        }
        self.meta_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")

        self.total_frames = 0
        self.gameplay_frames = 0
        self.raw_player_detections = 0
        self.parsed_player_detections = 0
        self.final_player_detections = 0
        self.raw_ball_candidate_frames = 0
        self.selected_ball_frames = 0
        self.predicted_ball_frames = 0
        self.missing_ball_frames = 0
        self.ball_rejected_outside_court = 0
        self.valid_homography_frames = 0
        self.keypoint_model_frames = 0
        self.selected_ball_confidences = []

    def log_frame(self, frame_state: Any) -> None:
        detector: Dict[str, Any] = frame_state.diagnostics.get("detector", {})
        analytics: Dict[str, Any] = frame_state.diagnostics.get("analytics", {})
        camera: Dict[str, Any] = frame_state.diagnostics.get("camera", {})

        valid_gameplay = bool(getattr(frame_state, "valid_gameplay", False))
        raw_ball_candidates = int(detector.get("raw_ball_candidates", 0) or 0)
        selected_ball = bool(detector.get("selected_ball", False))
        ball_source = analytics.get("ball_source", "missing")
        selected_conf = detector.get("selected_ball_conf")
        rejected_by_class = detector.get("rejected_outside_court_by_class", {}) or {}
        ball_rejected = sum(
            int(count)
            for name, count in rejected_by_class.items()
            if "ball" in str(name).lower()
        )

        homography_is_identity = bool(camera.get("homography_is_identity", True))
        homography_valid = bool(analytics.get("homography_active", False)) and not homography_is_identity
        keypoint_model_active = bool(camera.get("keypoint_model_active", False))

        record = {
            "frame_id": int(getattr(frame_state, "frame_id", 0)),
            "timestamp_sec": float(getattr(frame_state, "timestamp", 0.0)),
            "valid_gameplay": valid_gameplay,
            "raw_total": int(detector.get("raw_total", 0) or 0),
            "raw_class_counts": detector.get("raw_class_counts", {}) or {},
            "raw_player_candidates": int(detector.get("raw_player_candidates", 0) or 0),
            "raw_ball_candidates": raw_ball_candidates,
            "raw_ball_max_conf": detector.get("raw_ball_max_conf"),
            "rejected_outside_court_by_class": rejected_by_class,
            "parsed_players": int(detector.get("parsed_players", 0) or 0),
            "selected_ball": selected_ball,
            "selected_ball_conf": selected_conf,
            "ball_source": ball_source,
            "players_final": int(analytics.get("players_final", len(getattr(frame_state, "players", []))) or 0),
            "keypoint_model_active": keypoint_model_active,
            "homography_is_identity": homography_is_identity,
            "homography_valid": homography_valid,
            "events": list(getattr(frame_state, "events", []) or []),
        }

        with self.frames_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.total_frames += 1
        if valid_gameplay:
            self.gameplay_frames += 1
            self.raw_player_detections += record["raw_player_candidates"]
            self.parsed_player_detections += record["parsed_players"]
            self.final_player_detections += record["players_final"]
            self.ball_rejected_outside_court += ball_rejected
            if raw_ball_candidates > 0:
                self.raw_ball_candidate_frames += 1
            if selected_ball:
                self.selected_ball_frames += 1
            if ball_source == "predicted_hold":
                self.predicted_ball_frames += 1
            elif ball_source == "missing":
                self.missing_ball_frames += 1
            if homography_valid:
                self.valid_homography_frames += 1
            if keypoint_model_active:
                self.keypoint_model_frames += 1
            if selected_conf is not None:
                self.selected_ball_confidences.append(float(selected_conf))

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return round(100.0 * numerator / max(denominator, 1), 2)

    @staticmethod
    def _average(total: int, denominator: int) -> float:
        return round(total / max(denominator, 1), 3)

    def finish(self, extra: Optional[Dict[str, Any]] = None) -> Path:
        confidences = sorted(self.selected_ball_confidences)
        mean_conf = round(sum(confidences) / len(confidences), 4) if confidences else None
        median_conf = confidences[len(confidences) // 2] if confidences else None

        summary = {
            **self.config,
            "elapsed_sec": round(time.time() - self.started, 2),
            "total_frames": self.total_frames,
            "gameplay_frames": self.gameplay_frames,
            "players": {
                "raw_per_gameplay_frame": self._average(self.raw_player_detections, self.gameplay_frames),
                "parsed_per_gameplay_frame": self._average(self.parsed_player_detections, self.gameplay_frames),
                "final_per_gameplay_frame": self._average(self.final_player_detections, self.gameplay_frames),
            },
            "ball": {
                "raw_candidate_frame_rate": self._rate(self.raw_ball_candidate_frames, self.gameplay_frames),
                "selected_detection_frame_rate": self._rate(self.selected_ball_frames, self.gameplay_frames),
                "predicted_hold_frame_rate": self._rate(self.predicted_ball_frames, self.gameplay_frames),
                "missing_frame_rate": self._rate(self.missing_ball_frames, self.gameplay_frames),
                "rejected_outside_court": self.ball_rejected_outside_court,
                "selected_conf_mean": mean_conf,
                "selected_conf_median": median_conf,
            },
            "court": {
                "keypoint_model_active_rate": self._rate(self.keypoint_model_frames, self.gameplay_frames),
                "valid_homography_rate": self._rate(self.valid_homography_frames, self.gameplay_frames),
            },
            "homography_active_rate": self._rate(self.valid_homography_frames, self.gameplay_frames),
            "frames_file": str(self.frames_path),
        }
        if extra:
            summary["extra"] = extra

        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return self.summary_path
