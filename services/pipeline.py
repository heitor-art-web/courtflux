import os
from pathlib import Path

import numpy as np

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

    def _enrich_player_tracking_diagnostics(self, frame_state):
        """Persist normalized player observations through the existing frames.jsonl path.

        RunDiagnostics already serializes ``detector.team_predictions``.  P1.1 enriches
        that existing diagnostic field after court projection so the benchmark can use
        track IDs, geometry and court coordinates without introducing a second inference
        pipeline or changing player/tracker decisions.
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

        snapshots = []
        for player in list(getattr(frame_state, "players", []) or []):
            if not isinstance(player, dict):
                continue
            try:
                track_id = int(player.get("track_id", -1))
            except (TypeError, ValueError):
                track_id = -1
            snapshot = dict(by_track.get(track_id, {})) if track_id >= 0 else {}
            snapshot.update(
                {
                    "track_id": track_id,
                    "box": self._serializable_box(player.get("box")),
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
                    "court_xy": self._serializable_point(player.get("court_xy")),
                    "active_on_court": player.get("active_on_court"),
                }
            )
            snapshots.append(snapshot)

        detector_diag["team_predictions"] = snapshots
        detector_diag["p1_tracking_schema"] = 1
        return frame_state

    def process(self, frame_state):
        self.profiler.start("Total")
        self.profiler.start("Gameplay Filter")
        is_gameplay = self.gameplay_filter.is_gameplay(frame_state.frame)
        frame_state.valid_gameplay = is_gameplay
        if hasattr(self.gameplay_filter, "last_diagnostics"):
            frame_state.diagnostics["gameplay_filter"] = dict(getattr(self.gameplay_filter, "last_diagnostics", {}) or {})
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
            frame_state.diagnostics.setdefault("analytics", {}).update(
                {
                    "ball_source": detector_ball_source,
                    "players_final": len(getattr(frame_state, "players", []) or []),
                    "homography_active": bool(camera_diag.get("valid", False)),
                    "homography_diagnostic_only": self.camera_tracker is not None,
                    "detection_only": True,
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
