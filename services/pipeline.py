import os
from pathlib import Path
import statistics

import numpy as np

from services.vision.court_state_v1 import CourtStateEstimatorV1
from services.vision.diagnostic_preview import DiagnosticPreview


class Pipeline:
    def __init__(
        self,
        gameplay_filter,
        camera_tracker,
        detector,
        database,
        renderer,
        profiler,
        engine,
        backboard_roi,
        db_conn=None,
        detection_only: bool = False,
        persist_results: bool = True,
    ):
        self.gameplay_filter = gameplay_filter
        self.camera_tracker = camera_tracker
        self.detector = detector
        self.database = database
        self.renderer = renderer
        self.profiler = profiler
        self.engine = engine
        self.backboard_roi = backboard_roi
        self.db_conn = db_conn
        self.detection_only = bool(detection_only)
        self.persist_results = bool(persist_results)
        self.court_state_estimator = CourtStateEstimatorV1()
        self._camera_tracker_initialized = False
        self._renderer_closed = False

        if self.renderer is None and self.detection_only:
            preview_path = os.getenv("BASKETBALL_PREVIEW_PATH", "").strip()
            if preview_path:
                match_config = os.getenv("BASKETBALL_PREVIEW_MATCH_CONFIG", "").strip()
                interval = int(os.getenv("BASKETBALL_PREVIEW_INTERVAL", "15") or "15")
                video_path = os.getenv("BASKETBALL_PREVIEW_VIDEO_PATH", "").strip()
                video_fps = float(os.getenv("BASKETBALL_PREVIEW_VIDEO_FPS", "4") or "4")
                self.renderer = DiagnosticPreview(
                    output_path=Path(preview_path),
                    match_config_path=Path(match_config) if match_config else None,
                    interval_frames=interval,
                    video_output_path=Path(video_path) if video_path else None,
                    video_fps=video_fps,
                )

    def _ensure_camera_tracker(self, frame: np.ndarray):
        if not self._camera_tracker_initialized and self.camera_tracker is not None:
            if callable(self.camera_tracker) and not hasattr(self.camera_tracker, "process"):
                self.camera_tracker = self.camera_tracker(first_frame=frame, use_keypoint_model=True)
            self._camera_tracker_initialized = True

    @staticmethod
    def _serializable_point(value):
        if value is None:
            return None
        try:
            if len(value) < 2:
                return None
            return [round(float(value[0]), 6), round(float(value[1]), 6)]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _serializable_box(value):
        if value is None:
            return None
        try:
            if len(value) < 4:
                return None
            return [round(float(value[index]), 3) for index in range(4)]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _plausible_court_xy(value, margin_m: float = 0.5) -> bool:
        if value is None:
            return False
        try:
            if len(value) < 2:
                return False
            x_m = float(value[0])
            y_m = float(value[1])
        except (TypeError, ValueError):
            return False
        return (
            np.isfinite(x_m)
            and np.isfinite(y_m)
            and -margin_m <= x_m <= 28.0 + margin_m
            and -margin_m <= y_m <= 15.0 + margin_m
        )

    def _enrich_player_tracking_diagnostics(self, frame_state):
        """Persist P1.1 observations without changing detector/tracker decisions.

        Court coordinates are serialized only when calibration is explicitly valid and
        the projected point is plausible on a FIBA court. Frame-size and box-height
        diagnostics let the evaluator distinguish broadcast close-ups from wide views.
        """
        detector_diag = frame_state.diagnostics.setdefault("detector", {})
        existing = list(detector_diag.get("team_predictions", []) or [])
        by_track = {}
        for item in existing:
            if not isinstance(item, dict):
                continue
            try:
                track_id = int(item.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if track_id >= 0 and track_id not in by_track:
                by_track[track_id] = dict(item)

        frame = getattr(frame_state, "frame", None)
        frame_h = int(frame.shape[0]) if frame is not None and getattr(frame, "shape", None) else None
        frame_w = int(frame.shape[1]) if frame is not None and getattr(frame, "shape", None) else None

        camera_diag = frame_state.diagnostics.get("camera", {}) or {}
        homography_valid = bool(camera_diag.get("valid", False)) and not bool(
            camera_diag.get("homography_is_identity", True)
        )
        homography_source = camera_diag.get("source")
        reprojection_rmse_px = camera_diag.get("reprojection_rmse_px")

        snapshots = []
        player_height_ratios = []
        for player in list(getattr(frame_state, "players", []) or []):
            if not isinstance(player, dict):
                continue
            try:
                track_id = int(player.get("track_id", -1))
            except (TypeError, ValueError):
                track_id = -1

            snapshot = dict(by_track.get(track_id, {})) if track_id >= 0 else {}
            box = self._serializable_box(player.get("box"))
            height_ratio = None
            if box is not None and frame_h:
                height_ratio = max(0.0, float(box[3] - box[1])) / max(float(frame_h), 1.0)
                player_height_ratios.append(height_ratio)

            raw_court_xy = self._serializable_point(player.get("court_xy"))
            court_xy_valid = bool(homography_valid and self._plausible_court_xy(raw_court_xy))
            court_xy = raw_court_xy if court_xy_valid else None

            snapshot.update(
                {
                    "track_id": track_id,
                    "box": box,
                    "center": self._serializable_point(player.get("center")),
                    "confidence": (
                        round(float(player.get("confidence")), 6)
                        if player.get("confidence") is not None
                        else None
                    ),
                    "team": player.get("team"),
                    "team_confidence": player.get("team_confidence"),
                    "team_margin": player.get("team_margin"),
                    "shirt_num": player.get("shirt_num") or player.get("shirt_number"),
                    "player_name": player.get("player_name"),
                    "court_xy": court_xy,
                    "court_xy_valid": court_xy_valid,
                    "court_projection_source": (
                        player.get("court_projection_source") if court_xy_valid else None
                    ),
                    "homography_valid": homography_valid,
                    "homography_source": homography_source,
                    "homography_reprojection_rmse_px": reprojection_rmse_px,
                    "active_on_court": player.get("active_on_court"),
                    "identity_eligible": player.get("identity_eligible"),
                    "court_eligibility_source": player.get("court_eligibility_source"),
                    "local_court_surface_ratio": player.get("local_court_surface_ratio"),
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                    "player_box_height_ratio": (
                        round(height_ratio, 6) if height_ratio is not None else None
                    ),
                }
            )
            snapshots.append(snapshot)

        median_height_ratio = statistics.median(player_height_ratios) if player_height_ratios else None
        wide_view_proxy = bool(
            len(player_height_ratios) >= 2
            and median_height_ratio is not None
            and median_height_ratio <= 0.45
        )

        detector_diag["team_predictions"] = snapshots
        detector_diag["p1_tracking_schema"] = 3
        detector_diag["p1_tracking_frame"] = {
            "frame_width": frame_w,
            "frame_height": frame_h,
            "player_boxes": len(player_height_ratios),
            "median_player_box_height_ratio": (
                round(float(median_height_ratio), 6) if median_height_ratio is not None else None
            ),
            "wide_view_proxy": wide_view_proxy,
            "wide_view_proxy_semantics": "at_least_2_players_and_median_box_height_ratio_lte_0_45",
            "homography_valid": homography_valid,
            "homography_source": homography_source,
            "homography_reprojection_rmse_px": reprojection_rmse_px,
        }
        return frame_state

    @staticmethod
    def _apply_court_identity_gate(frame_state):
        """Suppress identity claims only for explicit off-court detections.

        The detection remains in ``frame_state.players`` for visual QA. UNKNOWN/ambiguous
        court eligibility is intentionally not rejected.
        """
        for player in list(getattr(frame_state, "players", []) or []):
            if not isinstance(player, dict) or player.get("active_on_court") is not False:
                continue
            player["team"] = None
            player["team_confidence"] = 0.0
            player["team_margin"] = 0.0
            player["shirt_num"] = None
            player["shirt_number"] = None
            player["player_name"] = None
            player["identity_eligible"] = False
            player["identity_rejection_reason"] = "explicit_off_court"

    def process(self, frame_state):
        self.profiler.start("Total")
        self.profiler.start("Gameplay Filter")
        is_gameplay = self.gameplay_filter.is_gameplay(frame_state.frame)
        frame_state.valid_gameplay = is_gameplay
        if hasattr(self.gameplay_filter, "last_diagnostics"):
            frame_state.diagnostics["gameplay_filter"] = dict(
                getattr(self.gameplay_filter, "last_diagnostics", {}) or {}
            )
        self.profiler.stop("Gameplay Filter")

        if not is_gameplay:
            if self.camera_tracker is not None and hasattr(self.camera_tracker, "reset_flow"):
                self.camera_tracker.reset_flow()
            self.profiler.frame_skipped()
            self.profiler.stop("Total")
            return frame_state

        if self.camera_tracker is not None:
            self.profiler.start("Camera Tracker")
            self._ensure_camera_tracker(frame_state.frame)
            frame_state = self.camera_tracker.process(frame_state)
            self.profiler.stop("Camera Tracker")

        self.profiler.start("Detector")
        frame_state = self.detector.process(frame_state)
        self.profiler.stop("Detector")

        if self.camera_tracker is not None and hasattr(self.camera_tracker, "project_entities"):
            self.profiler.start("Court Projection")
            frame_state = self.camera_tracker.project_entities(frame_state)
            self.profiler.stop("Court Projection")

        self.profiler.start("CourtState")
        court_state = self.court_state_estimator.update(
            frame_state,
            gameplay_diagnostics=frame_state.diagnostics.get("gameplay_filter", {}) or {},
        )
        frame_state.court_state = court_state
        frame_state.diagnostics["court_state"] = court_state.as_dict()
        self._apply_court_identity_gate(frame_state)
        if self.detection_only:
            frame_state.valid_gameplay = bool(court_state.analysis_eligible)
            gameplay_diag = frame_state.diagnostics.setdefault("gameplay_filter", {})
            gameplay_diag["pre_court_state_result"] = bool(is_gameplay)
            gameplay_diag["post_court_state_result"] = bool(frame_state.valid_gameplay)
            gameplay_diag["post_court_state_reason"] = court_state.eligibility_reason
        self.profiler.stop("CourtState")

        frame_state = self._enrich_player_tracking_diagnostics(frame_state)
        detector_diag = frame_state.diagnostics.get("detector", {}) or {}
        for name, milliseconds in (detector_diag.get("timing_ms", {}) or {}).items():
            self.profiler.add(f"Detector/{name}", float(milliseconds) / 1000.0)

        if not self.detection_only and self.engine is not None:
            self.profiler.start("Analytics")
            frame_state = self.engine.process(frame_state)
            self.profiler.stop("Analytics")
        else:
            detector_ball_source = str(detector_diag.get("ball_source") or "").strip()
            if not detector_ball_source:
                detector_ball_source = "detector" if getattr(frame_state, "ball", None) is not None else "missing"
            camera_diag = frame_state.diagnostics.get("camera", {}) or {}
            analysis_players = [
                player
                for player in list(getattr(frame_state, "players", []) or [])
                if not isinstance(player, dict) or player.get("active_on_court") is not False
            ]
            frame_state.diagnostics.setdefault("analytics", {}).update(
                {
                    "ball_source": detector_ball_source,
                    "players_final": len(analysis_players),
                    "players_detected_total": len(getattr(frame_state, "players", []) or []),
                    "homography_active": bool(camera_diag.get("valid", False)),
                    "homography_diagnostic_only": self.camera_tracker is not None,
                    "detection_only": True,
                    "court_state_mode": court_state.mode,
                    "court_state_analysis_eligible": court_state.analysis_eligible,
                }
            )

        if self.persist_results and self.database is not None:
            self.profiler.start("Database Buffer")
            self.database.save(frame_state)
            self.profiler.stop("Database Buffer")
            if not self.detection_only:
                self.profiler.start("Event Snapshots")
                self.database.save_events(frame_state)
                self.profiler.stop("Event Snapshots")

        if self.renderer is not None:
            stage_name = "Diagnostic Preview" if isinstance(self.renderer, DiagnosticPreview) else "Renderer"
            self.profiler.start(stage_name)
            frame_state = self.renderer.draw(frame_state)
            self.profiler.stop(stage_name)

        self.profiler.frame_processed()
        self.profiler.stop("Total")
        return frame_state

    def flush(self):
        if self.database is not None and hasattr(self.database, "flush"):
            self.profiler.start("Database Flush")
            self.database.flush()
            self.profiler.stop("Database Flush")
        if not self._renderer_closed and self.renderer is not None and hasattr(self.renderer, "close"):
            self.renderer.close()
            self._renderer_closed = True
