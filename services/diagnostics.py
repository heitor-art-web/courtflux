from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class RunDiagnostics:
    """Write comparable beta diagnostics with buffered frame I/O and identity summaries."""

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
        detection_only: bool = False,
        enable_jersey: bool = True,
        jersey_interval_frames: int = 12,
        batch_size: int = 100,
    ):
        self.started = time.perf_counter()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = root / "evaluation" / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.run_dir / "frames.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.meta_path = self.run_dir / "run_config.json"
        self.batch_size = max(1, int(batch_size))
        self._frame_buffer: list[dict[str, Any]] = []

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
            "detection_only": bool(detection_only),
            "jersey_enabled": bool(enable_jersey),
            "jersey_interval_frames": int(jersey_interval_frames),
            "diagnostics_batch_size": self.batch_size,
        }
        self._thresholds_captured = False
        self._write_meta()

        self.total_frames = 0
        self.gameplay_frames = 0
        self.raw_player_detections = 0
        self.parsed_player_detections = 0
        self.final_player_detections = 0
        self.team_classified_detections = 0
        self.jersey_identified_detections = 0
        self.raw_ball_candidate_frames = 0
        self.raw_ball_candidates_total = 0
        self.accepted_ball_candidates_total = 0
        self.multi_ball_candidate_frames = 0
        self.selected_ball_frames = 0
        self.predicted_ball_frames = 0
        self.missing_ball_frames = 0
        self.ball_rejected_outside_court = 0
        self.valid_homography_frames = 0
        self.keypoint_model_frames = 0
        self.selected_ball_confidences: list[float] = []

        # Direct-detector continuity. These deliberately ignore predicted_hold so that
        # we can measure whether the detector itself is good enough for passes/shots.
        self.ball_detection_streaks: list[int] = []
        self.ball_gap_lengths: list[int] = []
        self._current_ball_detection_streak = 0
        self._current_ball_gap = 0
        self.ball_reacquisitions = 0
        self.ball_center_step_norm: list[float] = []
        self.ball_large_jump_count = 0
        self._last_ball_center: tuple[float, float] | None = None
        self._last_ball_frame_id: int | None = None

        # Effective trajectory continuity includes accepted direct observations and
        # conservative predicted_hold frames. It is kept separate from direct metrics.
        self.ball_trajectory_streaks: list[int] = []
        self.ball_trajectory_gap_lengths: list[int] = []
        self._current_ball_trajectory_streak = 0
        self._current_ball_trajectory_gap = 0
        self.ball_trajectory_reacquisitions = 0
        self.ball_hold_ages: list[int] = []
        self.ball_association_rejected_candidates = 0
        self.ball_association_rejection_frames = 0
        self.ball_association_decision_frames = 0

        self.team_counts: Dict[str, int] = {}
        self.detector_timing_ms = defaultdict(float)
        self.detector_timing_calls = defaultdict(int)
        self.jersey_attempts = 0
        self.jersey_successes = 0

        self.raw_jersey_predictions = defaultdict(int)
        self.jersey_candidate_predictions = defaultdict(int)
        self.jersey_rejection_reasons = defaultdict(int)
        self.identified_track_ids: set[int] = set()
        self.team_confidence_total = 0.0
        self.team_confidence_samples = 0
        self.team_margin_total = 0.0
        self.team_margin_samples = 0
        self.checkpoint_diagnostics: dict[str, Any] = {}

    def _write_meta(self) -> None:
        self.meta_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")

    def capture_match_config(self, source_path: Path) -> Optional[Path]:
        source_path = Path(source_path)
        if not source_path.exists():
            self.config["match_config"] = {"status": "missing", "source": str(source_path)}
            self._write_meta()
            return None
        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.config["match_config"] = {
                "status": "invalid",
                "source": str(source_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._write_meta()
            return None

        snapshot_path = self.run_dir / "match_config.json"
        snapshot_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        metadata = data.get("metadata", {}) or {}
        teams = data.get("teams", {}) or {}
        self.config["match_config"] = {
            "status": "loaded",
            "source": str(source_path),
            "snapshot": str(snapshot_path),
            "home_team": metadata.get("home_team"),
            "away_team": metadata.get("away_team"),
            "active_kits": metadata.get("active_kits"),
            "teams": list(teams.keys()),
        }
        self._write_meta()
        return snapshot_path

    def _flush_frames(self) -> None:
        if not self._frame_buffer:
            return
        with self.frames_path.open("a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in self._frame_buffer)
        self._frame_buffer.clear()

    def log_frame(self, frame_state: Any) -> None:
        detector: Dict[str, Any] = frame_state.diagnostics.get("detector", {}) or {}
        analytics: Dict[str, Any] = frame_state.diagnostics.get("analytics", {}) or {}
        camera: Dict[str, Any] = frame_state.diagnostics.get("camera", {}) or {}

        if not self._thresholds_captured and detector:
            captured = False
            for target, source in (
                ("inference_conf", "inference_conf"),
                ("player_conf", "player_conf"),
                ("ball_conf", "ball_conf"),
            ):
                if detector.get(source) is not None:
                    self.config[target] = float(detector[source])
                    captured = True
            if captured:
                self._thresholds_captured = True
                self._write_meta()

        valid_gameplay = bool(getattr(frame_state, "valid_gameplay", False))
        raw_ball_candidates = int(detector.get("raw_ball_candidates", 0) or 0)
        accepted_ball_candidates = int(detector.get("accepted_ball_candidates", 0) or 0)
        selected_ball = bool(detector.get("selected_ball", False))
        ball_source = str(analytics.get("ball_source", "detector" if selected_ball else "missing") or "missing")
        selected_conf = detector.get("selected_ball_conf")
        selected_center = detector.get("selected_ball_center")
        association = detector.get("ball_association", {}) or {}
        hold_age_frames = int(association.get("hold_age_frames", 0) or 0)
        association_rejected = int(association.get("rejected_candidates", 0) or 0)
        trajectory_available = bool(selected_ball or ball_source == "predicted_hold")
        rejected_by_class = detector.get("rejected_outside_court_by_class", {}) or {}
        ball_rejected = sum(int(count) for name, count in rejected_by_class.items() if "ball" in str(name).lower())

        detection_only = bool(self.config.get("detection_only"))
        homography_is_identity = bool(camera.get("homography_is_identity", True))
        homography_valid = False if detection_only else (
            bool(analytics.get("homography_active", False)) and not homography_is_identity
        )
        keypoint_model_active = False if detection_only else bool(camera.get("keypoint_model_active", False))

        players = list(getattr(frame_state, "players", []) or [])
        frame_team_counts: Dict[str, int] = {}
        players_with_team = 0
        players_with_jersey = 0
        for player in players:
            if not isinstance(player, dict):
                continue
            team = player.get("team")
            if team and str(team).lower() not in {"unknown", "none"}:
                team_name = str(team)
                players_with_team += 1
                frame_team_counts[team_name] = frame_team_counts.get(team_name, 0) + 1
            team_conf = player.get("team_confidence")
            team_margin = player.get("team_margin")
            if team_conf is not None:
                self.team_confidence_total += float(team_conf)
                self.team_confidence_samples += 1
            if team_margin is not None:
                self.team_margin_total += float(team_margin)
                self.team_margin_samples += 1

            jersey = player.get("shirt_num") or player.get("shirt_number")
            if jersey is not None and str(jersey).strip():
                players_with_jersey += 1
                track_id = player.get("track_id")
                if isinstance(track_id, int) and track_id >= 0:
                    self.identified_track_ids.add(track_id)

        timings = detector.get("timing_ms", {}) or {}
        for name, ms in timings.items():
            try:
                self.detector_timing_ms[str(name)] += float(ms)
                self.detector_timing_calls[str(name)] += 1
            except (TypeError, ValueError):
                pass
        self.jersey_attempts += int(detector.get("jersey_attempts", 0) or 0)
        self.jersey_successes += int(detector.get("jersey_successes", 0) or 0)

        team_predictions = list(detector.get("team_predictions", []) or [])
        jersey_predictions = list(detector.get("jersey_predictions", []) or [])
        for prediction in jersey_predictions:
            if not isinstance(prediction, dict):
                continue
            raw = prediction.get("raw_canonical") or prediction.get("raw_number")
            if raw is not None:
                self.raw_jersey_predictions[str(raw)] += 1
            candidate = prediction.get("candidate_number")
            if candidate is not None:
                self.jersey_candidate_predictions[str(candidate)] += 1
            reason = str(prediction.get("rejection_reason") or "unknown")
            self.jersey_rejection_reasons[reason] += 1
            checkpoint = prediction.get("checkpoint")
            if isinstance(checkpoint, dict) and checkpoint:
                self.checkpoint_diagnostics = dict(checkpoint)

        record = {
            "frame_id": int(getattr(frame_state, "frame_id", 0)),
            "timestamp_sec": float(getattr(frame_state, "timestamp", 0.0)),
            "valid_gameplay": valid_gameplay,
            "raw_total": int(detector.get("raw_total", 0) or 0),
            "raw_class_counts": detector.get("raw_class_counts", {}) or {},
            "raw_player_candidates": int(detector.get("raw_player_candidates", 0) or 0),
            "accepted_player_candidates": int(detector.get("accepted_player_candidates", 0) or 0),
            "raw_ball_candidates": raw_ball_candidates,
            "accepted_ball_candidates": accepted_ball_candidates,
            "raw_ball_max_conf": detector.get("raw_ball_max_conf"),
            "raw_ball_confidences": detector.get("raw_ball_confidences", []) or [],
            "rejected_outside_court_by_class": rejected_by_class,
            "parsed_players": int(detector.get("parsed_players", 0) or 0),
            "selected_ball": selected_ball,
            "selected_ball_conf": selected_conf,
            "selected_ball_center": selected_center,
            "ball_source": ball_source,
            "trajectory_available": trajectory_available,
            "ball_association": association,
            "inference_conf": detector.get("inference_conf"),
            "player_conf": detector.get("player_conf"),
            "ball_conf": detector.get("ball_conf"),
            "players_final": int(analytics.get("players_final", len(players)) or 0),
            "players_with_team": players_with_team,
            "players_with_jersey": players_with_jersey,
            "team_counts": frame_team_counts,
            "team_predictions": team_predictions,
            "jersey_predictions": jersey_predictions,
            "timing_ms": timings,
            "events": [str(event) for event in (getattr(frame_state, "events", []) or [])],
        }
        if not detection_only:
            record.update(
                {
                    "keypoint_model_active": keypoint_model_active,
                    "homography_is_identity": homography_is_identity,
                    "homography_valid": homography_valid,
                }
            )

        self._frame_buffer.append(record)
        if len(self._frame_buffer) >= self.batch_size:
            self._flush_frames()

        self.total_frames += 1
        if valid_gameplay:
            self.gameplay_frames += 1
            self.raw_player_detections += record["raw_player_candidates"]
            self.parsed_player_detections += record["parsed_players"]
            self.final_player_detections += record["players_final"]
            self.team_classified_detections += players_with_team
            self.jersey_identified_detections += players_with_jersey
            for team_name, count in frame_team_counts.items():
                self.team_counts[team_name] = self.team_counts.get(team_name, 0) + int(count)

            self.ball_rejected_outside_court += ball_rejected
            self.raw_ball_candidates_total += raw_ball_candidates
            self.accepted_ball_candidates_total += accepted_ball_candidates
            if raw_ball_candidates > 0:
                self.raw_ball_candidate_frames += 1
            if raw_ball_candidates > 1:
                self.multi_ball_candidate_frames += 1

            if association:
                self.ball_association_decision_frames += 1
                self.ball_association_rejected_candidates += association_rejected
                if association_rejected > 0:
                    self.ball_association_rejection_frames += 1

            if selected_ball:
                self.selected_ball_frames += 1
                if self._current_ball_gap > 0:
                    self.ball_gap_lengths.append(self._current_ball_gap)
                    self.ball_reacquisitions += 1
                    self._current_ball_gap = 0
                self._current_ball_detection_streak += 1

                if selected_center is not None and len(selected_center) >= 2:
                    center = (float(selected_center[0]), float(selected_center[1]))
                    frame_id = int(record["frame_id"])
                    if self._last_ball_center is not None and self._last_ball_frame_id is not None:
                        delta_frames = max(frame_id - self._last_ball_frame_id, 1)
                        dx = center[0] - self._last_ball_center[0]
                        dy = center[1] - self._last_ball_center[1]
                        distance = (dx * dx + dy * dy) ** 0.5
                        frame = getattr(frame_state, "frame", None)
                        if frame is not None and getattr(frame, "shape", None) is not None:
                            height, width = frame.shape[:2]
                            diagonal = max((float(width) ** 2 + float(height) ** 2) ** 0.5, 1.0)
                            step_norm = distance / diagonal / delta_frames
                            self.ball_center_step_norm.append(step_norm)
                            if step_norm > 0.15:
                                self.ball_large_jump_count += 1
                    self._last_ball_center = center
                    self._last_ball_frame_id = frame_id
            else:
                if self._current_ball_detection_streak > 0:
                    self.ball_detection_streaks.append(self._current_ball_detection_streak)
                    self._current_ball_detection_streak = 0
                self._current_ball_gap += 1

            if trajectory_available:
                if self._current_ball_trajectory_gap > 0:
                    self.ball_trajectory_gap_lengths.append(self._current_ball_trajectory_gap)
                    self.ball_trajectory_reacquisitions += 1
                    self._current_ball_trajectory_gap = 0
                self._current_ball_trajectory_streak += 1
            else:
                if self._current_ball_trajectory_streak > 0:
                    self.ball_trajectory_streaks.append(self._current_ball_trajectory_streak)
                    self._current_ball_trajectory_streak = 0
                self._current_ball_trajectory_gap += 1

            if ball_source == "predicted_hold":
                self.predicted_ball_frames += 1
                if hold_age_frames > 0:
                    self.ball_hold_ages.append(hold_age_frames)
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
    def _average(total: float, denominator: int) -> float:
        return round(total / max(denominator, 1), 3)

    @staticmethod
    def _sorted_counts(values: dict[str, int], limit: int = 20) -> dict[str, int]:
        return dict(sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit])

    @staticmethod
    def _quantile(values: list[float] | list[int], q: float):
        if not values:
            return None
        ordered = sorted(values)
        index = int(round((len(ordered) - 1) * max(0.0, min(float(q), 1.0))))
        value = ordered[index]
        return round(float(value), 6) if isinstance(value, float) else int(value)

    def finish(self, extra: Optional[Dict[str, Any]] = None, performance: Optional[Dict[str, Any]] = None) -> Path:
        self._flush_frames()
        confidences = sorted(self.selected_ball_confidences)
        mean_conf = round(sum(confidences) / len(confidences), 4) if confidences else None
        median_conf = self._quantile(confidences, 0.50)
        p10_conf = self._quantile(confidences, 0.10)
        p90_conf = self._quantile(confidences, 0.90)

        detection_streaks = list(self.ball_detection_streaks)
        gap_lengths = list(self.ball_gap_lengths)
        if self._current_ball_detection_streak > 0:
            detection_streaks.append(self._current_ball_detection_streak)
        if self._current_ball_gap > 0:
            gap_lengths.append(self._current_ball_gap)

        trajectory_streaks = list(self.ball_trajectory_streaks)
        trajectory_gap_lengths = list(self.ball_trajectory_gap_lengths)
        if self._current_ball_trajectory_streak > 0:
            trajectory_streaks.append(self._current_ball_trajectory_streak)
        if self._current_ball_trajectory_gap > 0:
            trajectory_gap_lengths.append(self._current_ball_trajectory_gap)

        timing_summary = {}
        for name, total_ms in sorted(self.detector_timing_ms.items(), key=lambda item: item[1], reverse=True):
            calls = self.detector_timing_calls[name]
            timing_summary[name] = {
                "calls": calls,
                "total_sec": round(total_ms / 1000.0, 3),
                "avg_ms": round(total_ms / max(calls, 1), 3),
            }

        avg_team_conf = self.team_confidence_total / max(self.team_confidence_samples, 1)
        avg_team_margin = self.team_margin_total / max(self.team_margin_samples, 1)
        identity_diagnostics = {
            "identified_tracks": len(self.identified_track_ids),
            "raw_prediction_counts": self._sorted_counts(self.raw_jersey_predictions),
            "roster_candidate_counts": self._sorted_counts(self.jersey_candidate_predictions),
            "rejection_reasons": self._sorted_counts(self.jersey_rejection_reasons),
            "checkpoint": self.checkpoint_diagnostics,
            "avg_team_confidence": round(avg_team_conf, 4),
            "avg_team_margin": round(avg_team_margin, 6),
        }

        center_steps = self.ball_center_step_norm
        ball_continuity = {
            "scope": "direct_associated_detections_only",
            "reacquisitions": self.ball_reacquisitions,
            "detection_streak_count": len(detection_streaks),
            "detection_streak_median_frames": self._quantile(detection_streaks, 0.50),
            "detection_streak_p90_frames": self._quantile(detection_streaks, 0.90),
            "detection_streak_max_frames": max(detection_streaks) if detection_streaks else None,
            "miss_gap_count": len(gap_lengths),
            "miss_gap_median_frames": self._quantile(gap_lengths, 0.50),
            "miss_gap_p90_frames": self._quantile(gap_lengths, 0.90),
            "miss_gap_max_frames": max(gap_lengths) if gap_lengths else None,
            "center_step_norm_diag_median": self._quantile(center_steps, 0.50),
            "center_step_norm_diag_p90": self._quantile(center_steps, 0.90),
            "center_step_norm_diag_max": round(max(center_steps), 6) if center_steps else None,
            "large_jump_rate_gt_0_15_diag_per_frame": self._rate(
                self.ball_large_jump_count,
                len(center_steps),
            ),
        }

        effective_frames = self.selected_ball_frames + self.predicted_ball_frames
        ball_trajectory = {
            "scope": "direct_associated_plus_predicted_hold",
            "effective_coverage_frame_rate": self._rate(effective_frames, self.gameplay_frames),
            "direct_frame_rate": self._rate(self.selected_ball_frames, self.gameplay_frames),
            "predicted_hold_frame_rate": self._rate(self.predicted_ball_frames, self.gameplay_frames),
            "missing_frame_rate": self._rate(self.missing_ball_frames, self.gameplay_frames),
            "reacquisitions": self.ball_trajectory_reacquisitions,
            "trajectory_streak_count": len(trajectory_streaks),
            "trajectory_streak_median_frames": self._quantile(trajectory_streaks, 0.50),
            "trajectory_streak_p90_frames": self._quantile(trajectory_streaks, 0.90),
            "trajectory_streak_max_frames": max(trajectory_streaks) if trajectory_streaks else None,
            "trajectory_gap_count": len(trajectory_gap_lengths),
            "trajectory_gap_median_frames": self._quantile(trajectory_gap_lengths, 0.50),
            "trajectory_gap_p90_frames": self._quantile(trajectory_gap_lengths, 0.90),
            "trajectory_gap_max_frames": max(trajectory_gap_lengths) if trajectory_gap_lengths else None,
            "hold_age_median_frames": self._quantile(self.ball_hold_ages, 0.50),
            "hold_age_p90_frames": self._quantile(self.ball_hold_ages, 0.90),
            "hold_age_max_frames": max(self.ball_hold_ages) if self.ball_hold_ages else None,
            "association_decision_frames": self.ball_association_decision_frames,
            "association_rejected_candidates_total": self.ball_association_rejected_candidates,
            "association_frames_with_rejections": self.ball_association_rejection_frames,
            "association_frame_rejection_rate": self._rate(
                self.ball_association_rejection_frames,
                self.ball_association_decision_frames,
            ),
            "association_candidate_rejection_rate": self._rate(
                self.ball_association_rejected_candidates,
                self.accepted_ball_candidates_total,
            ),
        }

        summary = {
            **self.config,
            "elapsed_sec": round(time.perf_counter() - self.started, 2),
            "total_frames": self.total_frames,
            "gameplay_frames": self.gameplay_frames,
            "players": {
                "raw_per_gameplay_frame": self._average(self.raw_player_detections, self.gameplay_frames),
                "parsed_per_gameplay_frame": self._average(self.parsed_player_detections, self.gameplay_frames),
                "final_per_gameplay_frame": self._average(self.final_player_detections, self.gameplay_frames),
                "team_classification_rate": self._rate(self.team_classified_detections, self.final_player_detections),
                "jersey_identification_rate": self._rate(self.jersey_identified_detections, self.final_player_detections),
                "jersey_attempts": self.jersey_attempts,
                "jersey_successes": self.jersey_successes,
                "team_counts": self.team_counts,
                "identity_diagnostics": identity_diagnostics,
            },
            "ball": {
                "raw_candidate_frame_rate": self._rate(self.raw_ball_candidate_frames, self.gameplay_frames),
                "selected_detection_frame_rate": self._rate(self.selected_ball_frames, self.gameplay_frames),
                "predicted_hold_frame_rate": self._rate(self.predicted_ball_frames, self.gameplay_frames),
                "effective_trajectory_frame_rate": self._rate(effective_frames, self.gameplay_frames),
                "missing_frame_rate": self._rate(self.missing_ball_frames, self.gameplay_frames),
                "raw_candidates_per_gameplay_frame": self._average(self.raw_ball_candidates_total, self.gameplay_frames),
                "accepted_candidates_per_gameplay_frame": self._average(self.accepted_ball_candidates_total, self.gameplay_frames),
                "multi_candidate_frame_rate": self._rate(self.multi_ball_candidate_frames, self.gameplay_frames),
                "rejected_outside_court": self.ball_rejected_outside_court,
                "selected_conf_mean": mean_conf,
                "selected_conf_p10": p10_conf,
                "selected_conf_median": median_conf,
                "selected_conf_p90": p90_conf,
                "continuity": ball_continuity,
                "trajectory": ball_trajectory,
            },
            "detector_timing": timing_summary,
            "performance": performance or {},
            "frames_file": str(self.frames_path),
        }
        if not self.config.get("detection_only"):
            summary["court"] = {
                "keypoint_model_active_rate": self._rate(self.keypoint_model_frames, self.gameplay_frames),
                "valid_homography_rate": self._rate(self.valid_homography_frames, self.gameplay_frames),
            }
            summary["homography_active_rate"] = self._rate(self.valid_homography_frames, self.gameplay_frames)
        if extra:
            summary["extra"] = extra

        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return self.summary_path
