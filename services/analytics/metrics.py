"""
Analysis Metrics Logger.

Collects high-level pipeline statistics and exports them
to a JSON report for comparing different model versions.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from core import config
import os


class AnalysisMetrics:

    def __init__(self):

        # General
        self.model = ""
        self.video = ""
        self.total_frames = 0

        # Detection
        self.players_detected = 0
        self.ball_detected = 0

        # OCR
        self.ocr_attempts = 0
        self.ocr_success = 0

        # Team Classification
        self.team_predictions = 0
        self.team_unknown = 0

        # Tracking
        self.track_switches = 0

        # Basketball Events
        self.shot_attempts = 0
        self.shots_made = 0
        self.rebounds = 0
        self.turnovers = 0
        self.possessions = 0

    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    def log_detection(
        self,
        players: int,
        ball_detected: bool,
    ):

        self.players_detected += players

        if ball_detected:
            self.ball_detected += 1

    # --------------------------------------------------
    # OCR
    # --------------------------------------------------

    def log_ocr(
        self,
        success: bool,
    ):

        self.ocr_attempts += 1

        if success:
            self.ocr_success += 1

    # --------------------------------------------------
    # Team Classification
    # --------------------------------------------------

    def log_team(
        self,
        known: bool,
    ):

        self.team_predictions += 1

        if not known:
            self.team_unknown += 1

    # --------------------------------------------------
    # Tracking
    # --------------------------------------------------

    def log_track_switch(self):

        self.track_switches += 1

    # --------------------------------------------------
    # Basketball Events
    # --------------------------------------------------

    def log_shot(
        self,
        made: bool,
    ):

        self.shot_attempts += 1

        if made:
            self.shots_made += 1

    def log_rebound(self):

        self.rebounds += 1

    def log_turnover(self):

        self.turnovers += 1

    def log_possession(self):

        self.possessions += 1

    # --------------------------------------------------
    # Export
    # --------------------------------------------------

    def export(self):

        folder = Path("evaluation")
        folder.mkdir(parents=True, exist_ok=True)

        video = Path(self.video).stem if self.video else "analysis"
        model = config.settings.model or "unknown"
        
        filename = folder / f"{video}_{datetime.now():%Y%m%d_%H%M%S}.json"

        report = {

            "timestamp": datetime.now().isoformat(),

            "model": model,

            "video": self.video,

            "detections": {

                "frames_processed": self.total_frames,

                "players_detected": self.players_detected,

                "average_players_per_frame": round(
                    self.players_detected / max(self.total_frames, 1),
                    2,
                ),

                "ball_detected": self.ball_detected,

                "ball_detection_rate": round(
                    100 * self.ball_detected / max(self.total_frames, 1),
                    2,
                ),

            },

            "ocr": {

                "attempts":
                    self.ocr_attempts,

                "success":
                    self.ocr_success,

                "success_rate":
                    round(
                        self.ocr_success /
                        max(self.ocr_attempts, 1),
                        3,
                    ),

            },

            "teams": {

                "predictions":
                    self.team_predictions,

                "unknown":
                    self.team_unknown,

            },

            "tracking": {

                "track_switches":
                    self.track_switches,

            },

            "events": {

                "shot_attempts":
                    self.shot_attempts,

                "shots_made":
                    self.shots_made,

                "rebounds":
                    self.rebounds,

                "turnovers":
                    self.turnovers,

                "possessions":
                    self.possessions,

            },

        }

        with open(filename, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Metrics saved to {filename}")

        return filename