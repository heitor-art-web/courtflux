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

        # The local Streamlit beta can request a lightweight rolling preview through
        # environment variables. Remote workers do not pay this cost unless enabled.
        if self.renderer is None and self.detection_only:
            preview_path = os.getenv("BASKETBALL_PREVIEW_PATH", "").strip()
            if preview_path:
                match_config = os.getenv("BASKETBALL_PREVIEW_MATCH_CONFIG", "").strip()
                interval = int(os.getenv("BASKETBALL_PREVIEW_INTERVAL", "15") or "15")
                self.renderer = DiagnosticPreview(
                    output_path=Path(preview_path),
                    match_config_path=Path(match_config) if match_config else None,
                    interval_frames=interval,
                )

    def _ensure_camera_tracker(self, frame: np.ndarray):
        if not self._camera_tracker_initialized and self.camera_tracker is not None:
            if callable(self.camera_tracker) and not hasattr(self.camera_tracker, "process"):
                self.camera_tracker = self.camera_tracker(
                    first_frame=frame,
                    use_keypoint_model=True,
                    model_id="basketball-court-detection-2/14",
                )
            self._camera_tracker_initialized = True

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
            if not self.detection_only and self.camera_tracker is not None and hasattr(self.camera_tracker, "reset_flow"):
                self.camera_tracker.reset_flow()
            self.profiler.frame_skipped()
            self.profiler.stop("Total")
            return frame_state

        # Detection-only beta deliberately avoids camera/homography work.
        if not self.detection_only and self.camera_tracker is not None:
            self.profiler.start("Camera Tracker")
            self._ensure_camera_tracker(frame_state.frame)
            frame_state = self.camera_tracker.process(frame_state)
            self.profiler.stop("Camera Tracker")

        self.profiler.start("Detector")
        frame_state = self.detector.process(frame_state)
        self.profiler.stop("Detector")

        # Import detector sub-stage timings into the shared profiler.
        detector_diag = frame_state.diagnostics.get("detector", {}) or {}
        for name, milliseconds in (detector_diag.get("timing_ms", {}) or {}).items():
            self.profiler.add(f"Detector/{name}", float(milliseconds) / 1000.0)

        if not self.detection_only and self.engine is not None:
            self.profiler.start("Analytics")
            frame_state = self.engine.process(frame_state)
            self.profiler.stop("Analytics")
        else:
            frame_state.diagnostics.setdefault("analytics", {})
            frame_state.diagnostics["analytics"].update(
                {
                    "ball_source": "detector" if getattr(frame_state, "ball", None) is not None else "missing",
                    "players_final": len(getattr(frame_state, "players", []) or []),
                    "homography_active": False,
                    "detection_only": True,
                }
            )

        if self.persist_results and self.database is not None:
            self.profiler.start("Database Buffer")
            self.database.save(frame_state)
            self.profiler.stop("Database Buffer")

            # Full analytics mode may still save event images, but event DB rows are
            # already handled by save() and are never inserted twice.
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
