from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

import numpy as np
from ultralytics import YOLO

from services.vision.ball_tracker import TemporalBallTracker


class PerformanceDetector:
    """Lean detector/tracker with persistent team/player identity and temporal ball association."""

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
        self.player_conf_thresh = self.detector_conf
        self.inference_conf_thresh = self._env_conf("BASKETBALL_INFERENCE_CONF", self.detector_conf)
        self.ball_conf_thresh = self._env_conf(
            "BASKETBALL_BALL_CONF",
            max(0.01, min(self.detector_conf, 0.10)),
        )
        self.inference_conf_thresh = min(
            self.inference_conf_thresh,
            self.player_conf_thresh,
            self.ball_conf_thresh,
        )
        self.imgsz = int(imgsz)
        self.iou = float(iou)
        self.tracker = tracker
        self.max_det = int(max_det)
        self.enable_jersey = bool(enable_jersey)
        self.jersey_interval_frames = max(1, int(jersey_interval_frames))
        self.player_db: dict[int, dict[str, Any]] = {}
        self.jersey_attempts_total = 0
        self.jersey_successes_total = 0
        self.jersey_time_ms_total = 0.0

        self.ball_temporal_enabled = self._env_bool("BASKETBALL_BALL_TEMPORAL", True)
        self.ball_tracker = TemporalBallTracker(
            max_hold_frames=self._env_int("BASKETBALL_BALL_HOLD_FRAMES", 5, 0, 30),
            reset_after_frames=self._env_int("BASKETBALL_BALL_RESET_FRAMES", 12, 2, 120),
            base_gate_norm=self._env_float("BASKETBALL_BALL_GATE_BASE", 0.12, 0.02, 0.50),
            gate_growth_per_frame=self._env_float(
                "BASKETBALL_BALL_GATE_GROWTH", 0.025, 0.0, 0.25
            ),
            max_gate_norm=self._env_float("BASKETBALL_BALL_GATE_MAX", 0.28, 0.02, 0.75),
            velocity_alpha=self._env_float(
                "BASKETBALL_BALL_VELOCITY_ALPHA", 0.40, 0.0, 1.0
            ),
            score_confidence_weight=self._env_float(
                "BASKETBALL_BALL_SCORE_CONF", 0.58, 0.0, 1.0
            ),
            score_proximity_weight=self._env_float(
                "BASKETBALL_BALL_SCORE_PROX", 0.32, 0.0, 1.0
            ),
            score_size_weight=self._env_float(
                "BASKETBALL_BALL_SCORE_SIZE", 0.10, 0.0, 1.0
            ),
            high_confidence_gate_threshold=self._env_float(
                "BASKETBALL_BALL_HIGH_CONF_GATE_THRESHOLD", 0.85, 0.0, 1.0
            ),
            high_confidence_gate_multiplier=self._env_float(
                "BASKETBALL_BALL_HIGH_CONF_GATE_MULTIPLIER", 1.35, 1.0, 3.0
            ),
        )

        with open(profiling_path, "r", encoding="utf-8") as fh:
            self.match_config = json.load(fh)

        print(
            "🏀 Detection thresholds: "
            f"inference={self.inference_conf_thresh:.3f} "
            f"player={self.player_conf_thresh:.3f} "
            f"ball={self.ball_conf_thresh:.3f}",
            flush=True,
        )
        print(
            "🏀 Ball temporal association: "
            f"{'on' if self.ball_temporal_enabled else 'off'} "
            f"hold={self.ball_tracker.max_hold_frames}f "
            f"reset={self.ball_tracker.reset_after_frames}f "
            f"weights=({self.ball_tracker.score_confidence_weight:.2f},"
            f"{self.ball_tracker.score_proximity_weight:.2f},"
            f"{self.ball_tracker.score_size_weight:.2f}) "
            f"gate=({self.ball_tracker.base_gate_norm:.3f}+"
            f"{self.ball_tracker.gate_growth_per_frame:.3f}/f "
            f"max={self.ball_tracker.max_gate_norm:.3f}) "
            f"high_conf_gate=x{self.ball_tracker.high_confidence_gate_multiplier:.2f}",
            flush=True,
        )

    @staticmethod
    def _env_conf(name: str, fallback: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return float(fallback)
        try:
            return max(0.001, min(float(raw), 0.95))
        except ValueError:
            print(f"⚠️ Invalid {name}={raw!r}; using {fallback:.3f}.", flush=True)
            return float(fallback)

    @staticmethod
    def _env_bool(name: str, fallback: bool) -> bool:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return bool(fallback)
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _env_int(name: str, fallback: int, minimum: int, maximum: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return int(fallback)
        try:
            return max(minimum, min(int(raw), maximum))
        except ValueError:
            print(f"⚠️ Invalid {name}={raw!r}; using {fallback}.", flush=True)
            return int(fallback)

    @staticmethod
    def _env_float(name: str, fallback: float, minimum: float, maximum: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return float(fallback)
        try:
            return max(float(minimum), min(float(raw), float(maximum)))
        except ValueError:
            print(f"⚠️ Invalid {name}={raw!r}; using {fallback:.3f}.", flush=True)
            return float(fallback)

    @staticmethod
    def _is_player_class(name: str) -> bool:
        name = name.lower()
        return "person" in name or "player" in name

    @staticmethod
    def _is_ball_class(name: str) -> bool:
        name = name.lower()
        return "ball" in name or "basketball" in name or "sports ball" in name

    def _profile(self, track_id: int) -> dict[str, Any]:
        return self.player_db.setdefault(
            track_id,
            {
                "team": None,
                "team_score_sums": {},
                "team_observations": 0,
                "team_confidence": 0.0,
                "team_margin": 0.0,
                "shirt_num": None,
                "player_name": None,
                "identity_team": None,
                "identity_locked": False,
                "last_jersey_attempt": -10**9,
                "jersey_votes": {},
                "jersey_vote_counts": {},
            },
        )

    def _track(self, frame: np.ndarray):
        return self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.inference_conf_thresh,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

    def _select_ball(self, candidates: list[dict[str, Any]], frame_state):
        frame_id = int(getattr(frame_state, "frame_id", 0))
        if not self.ball_temporal_enabled:
            if not candidates:
                return None, {
                    "source": "missing",
                    "selected_index": None,
                    "score": None,
                    "distance_norm": None,
                    "gate_norm": None,
                    "rejected_candidates": 0,
                    "hold_age_frames": 0,
                }
            selected_index = max(
                range(len(candidates)),
                key=lambda idx: float(candidates[idx].get("confidence", 0.0)),
            )
            ball = dict(candidates[selected_index])
            ball["interpolated"] = False
            ball["source"] = "detector_highest_conf"
            return ball, {
                "source": "detector_highest_conf",
                "selected_index": selected_index,
                "score": float(ball.get("confidence", 0.0)),
                "distance_norm": None,
                "gate_norm": None,
                "rejected_candidates": 0,
                "hold_age_frames": 0,
            }

        decision = self.ball_tracker.select(
            candidates=candidates,
            frame_id=frame_id,
            frame_shape=frame_state.frame.shape,
        )
        return decision.ball, {
            "source": decision.source,
            "selected_index": decision.selected_index,
            "score": None if decision.score is None else round(float(decision.score), 6),
            "distance_norm": (
                None if decision.distance_norm is None else round(float(decision.distance_norm), 6)
            ),
            "gate_norm": None if decision.gate_norm is None else round(float(decision.gate_norm), 6),
            "rejected_candidates": int(decision.rejected_candidates),
            "hold_age_frames": int(decision.hold_age_frames),
        }

    def _parse(self, frame_state, results):
        players = []
        ball_candidates: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {}
        raw_total = 0
        raw_players = 0
        raw_balls = 0
        accepted_player_candidates = 0
        accepted_ball_candidates = 0
        raw_ball_max_conf = None
        raw_ball_confidences: list[float] = []

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
                    raw_ball_confidences.append(conf)
                    raw_ball_max_conf = conf if raw_ball_max_conf is None else max(raw_ball_max_conf, conf)

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else -1
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if is_player and conf >= self.player_conf_thresh and x2 > x1 and y2 > y1:
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
                elif is_ball and conf >= self.ball_conf_thresh and x2 > x1 and y2 > y1:
                    accepted_ball_candidates += 1
                    ball_candidates.append(
                        {
                            "box": [x1, y1, x2, y2],
                            "center": center,
                            "confidence": conf,
                            "interpolated": False,
                        }
                    )

        selected_ball, association = self._select_ball(ball_candidates, frame_state)
        ball_source = str(association.get("source") or "missing")
        direct_selected = selected_ball is not None and not bool(selected_ball.get("interpolated"))

        frame_state.players = players
        frame_state.ball = selected_ball
        frame_state.diagnostics.setdefault("analytics", {})["ball_source"] = ball_source
        frame_state.diagnostics["detector"] = {
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
            },
            "inference_conf": self.inference_conf_thresh,
            "player_conf": self.player_conf_thresh,
            "ball_conf": self.ball_conf_thresh,
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
        team_diagnostics = []
        for player in frame_state.players:
            x1, y1, x2, y2 = player["box"]
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            track_id = player["track_id"]

            if track_id == -1:
                result = self.visual.score_teams(crop) if hasattr(self.visual, "score_teams") else {}
                player["team"] = result.get("best_team") if result else self.visual.get_team_color(crop)
                continue

            profile = self._profile(track_id)
            result = self.visual.score_teams(crop) if hasattr(self.visual, "score_teams") else {}
            scores = dict(result.get("scores", {}) or {})
            if scores:
                profile["team_observations"] += 1
                for team_name, value in scores.items():
                    profile["team_score_sums"][team_name] = float(
                        profile["team_score_sums"].get(team_name, 0.0)
                    ) + float(value)

                observations = max(int(profile["team_observations"]), 1)
                averages = {
                    team_name: total / observations
                    for team_name, total in profile["team_score_sums"].items()
                }
                ordered = sorted(averages.items(), key=lambda item: item[1], reverse=True)
                if ordered:
                    best_team, best_score = ordered[0]
                    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
                    margin = max(0.0, best_score - second_score)
                    total_score = sum(max(0.0, score) for _, score in ordered) or 1e-9
                    confidence = max(0.0, best_score) / total_score
                    profile["team_confidence"] = confidence
                    profile["team_margin"] = margin
                    if best_score >= 0.018 and margin >= 0.004:
                        profile["team"] = best_team

            if profile.get("identity_locked") and profile.get("identity_team"):
                profile["team"] = profile["identity_team"]

            player["team"] = profile.get("team")
            player["team_confidence"] = round(float(profile.get("team_confidence", 0.0)), 4)
            player["team_margin"] = round(float(profile.get("team_margin", 0.0)), 6)
            player["shirt_num"] = profile.get("shirt_num")
            player["player_name"] = profile.get("player_name")

            if self.enable_jersey and hasattr(self.visual, "observe_track"):
                self.visual.observe_track(crop, track_id)

            team_diagnostics.append(
                {
                    "track_id": track_id,
                    "team": player.get("team"),
                    "confidence": player.get("team_confidence"),
                    "margin": player.get("team_margin"),
                    "scores": scores,
                }
            )

        frame_state.diagnostics.setdefault("detector", {})["team_predictions"] = team_diagnostics

    def _jerseys(self, frame_state):
        attempts = 0
        locks = 0
        predictions = []
        if not self.enable_jersey:
            return attempts, locks, predictions

        frame_id = int(getattr(frame_state, "frame_id", 0))
        frame = frame_state.frame
        for player in frame_state.players:
            track_id = player["track_id"]
            if track_id == -1:
                continue
            profile = self._profile(track_id)
            player["shirt_num"] = profile.get("shirt_num")
            player["player_name"] = profile.get("player_name")
            if profile.get("identity_locked"):
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

            if hasattr(self.visual, "predict_jersey"):
                prediction = dict(
                    self.visual.predict_jersey(
                        track_id=track_id,
                        team_name=profile.get("team"),
                    ) or {}
                )
            else:
                number = self.visual.get_shirt_number(
                    crop,
                    track_id=track_id,
                    team_name=profile.get("team"),
                )
                prediction = {
                    "accepted": bool(number),
                    "candidate_number": number or None,
                    "rejection_reason": "accepted" if number else "legacy_empty",
                }

            prediction.update(
                {
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "team": profile.get("team"),
                    "team_confidence": round(float(profile.get("team_confidence", 0.0)), 4),
                    "team_margin": round(float(profile.get("team_margin", 0.0)), 6),
                }
            )

            candidate = prediction.get("candidate_number")
            if prediction.get("accepted") and candidate:
                candidate = str(candidate)
                posterior = float(prediction.get("candidate_posterior", 1.0) or 0.0)
                evidence = max(0.0, posterior) * max(
                    0.35,
                    float(profile.get("team_confidence", 0.0)),
                )
                profile["jersey_votes"][candidate] = float(
                    profile["jersey_votes"].get(candidate, 0.0)
                ) + evidence
                profile["jersey_vote_counts"][candidate] = int(
                    profile["jersey_vote_counts"].get(candidate, 0)
                ) + 1

                ordered_votes = sorted(
                    profile["jersey_votes"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                top_number, top_evidence = ordered_votes[0]
                second_evidence = ordered_votes[1][1] if len(ordered_votes) > 1 else 0.0
                top_count = int(profile["jersey_vote_counts"].get(top_number, 0))

                if (
                    top_count >= 2
                    and top_evidence >= 0.35
                    and (top_evidence - second_evidence) >= 0.12
                ):
                    profile["shirt_num"] = top_number
                    profile["identity_team"] = profile.get("team")
                    profile["identity_locked"] = True
                    if hasattr(self.visual, "resolve_player_name"):
                        profile["player_name"] = self.visual.resolve_player_name(
                            profile.get("team"),
                            top_number,
                        )
                    player["shirt_num"] = top_number
                    player["player_name"] = profile.get("player_name")
                    player["ocr_success"] = True
                    locks += 1
                    prediction["identity_locked"] = True
                    prediction["resolved_player_name"] = profile.get("player_name")

            prediction["vote_counts"] = dict(profile.get("jersey_vote_counts", {}))
            predictions.append(prediction)

        return attempts, locks, predictions

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
            locked = sum(1 for profile in self.player_db.values() if profile.get("identity_locked"))
            ball_source = str(diag.get("ball_source") or "missing")
            print(
                f"[PROGRESS] frame={frame_id} players={len(frame_state.players)} "
                f"ball={ball_source} "
                f"jersey_attempts={self.jersey_attempts_total} "
                f"identity_locks={locked} "
                f"jersey_time={self.jersey_time_ms_total / 1000.0:.1f}s",
                flush=True,
            )

        return frame_state
