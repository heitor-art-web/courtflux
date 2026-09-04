import numpy as np


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

        self._camera_tracker_initialized = False

    def _ensure_camera_tracker(self, frame: np.ndarray):
        """Lazy-initialize CameraTracker on the first frame if passed as a factory."""
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

        # -------------------------------------------------
        # 1. Gameplay Filter
        # -------------------------------------------------
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

            self.profiler.stop("Total")
            return frame_state

        # -------------------------------------------------
        # 2. Camera Tracking (Homography / Court Matrix)
        # -------------------------------------------------
        self.profiler.start("Camera Tracker")
        if self.camera_tracker is not None:
            self._ensure_camera_tracker(frame_state.frame)
            frame_state = self.camera_tracker.process(frame_state)
        self.profiler.stop("Camera Tracker")

        # -------------------------------------------------
        # 3. Detection (Players & Ball)
        # -------------------------------------------------
        self.profiler.start("Detector")
        frame_state = self.detector.process(frame_state)
        self.profiler.stop("Detector")

        # -------------------------------------------------
        # 4. Analytics
        # -------------------------------------------------
        self.profiler.start("Analytics")
        frame_state = self.engine.process(frame_state)
        self.profiler.stop("Analytics")

        # -------------------------------------------------
        # 5. Database Logging
        # -------------------------------------------------
        self.profiler.start("Database")
        self.database.save(frame_state)
        self.database.save_events(frame_state)
        self.profiler.stop("Database")

        # -------------------------------------------------
        # 6. Optional rendering
        # -------------------------------------------------
        # The beta path is intentionally results-first. No radar / 2D court is
        # rendered. A renderer can still be supplied by other entry points when
        # annotated media is explicitly needed.
        if self.renderer is not None:
            self.profiler.start("Renderer")
            frame_state = self.renderer.draw(frame_state)
            self.profiler.stop("Renderer")

        self.profiler.stop("Total")
        return frame_state
