import cv2
import numpy as np
import json
from typing import List, Dict, Any, Optional
from ultralytics import YOLO


class Detector:
    # Map the JSON text colors to OpenCV BGR tuples for UI drawing
    UI_COLOR_MAP = {
        "Blue": (220, 100, 0),
        "Black": (40, 40, 40),
        "White": (245, 245, 245),
        "Red": (0, 0, 220),
        "Green": (0, 200, 0),
        "Yellow": (0, 215, 255),
        "Grey": (150, 150, 150),
        "Gray": (150, 150, 150),
    }
    DEFAULT_COLOR = (180, 180, 180)

    def __init__(
        self,
        model: YOLO,
        visual_analyzer,
        profiling_path: str,
        fps: int = 30,
        ball_in_basket_class_id: int = 1,
        jump_shot_class_id: int = 5,
        layup_dunk_class_id: int = 6,
        device: str = "mps",
        detector_conf: float = 0.40,
        imgsz: int = 640,
        iou: float = 0.5,
        tracker: str = "botsort.yaml",
        max_det: int = 40,
    ):
        self.model = model
        self.visual = visual_analyzer

        self.ball_in_basket_class_id = ball_in_basket_class_id
        self.jump_shot_class_id = jump_shot_class_id
        self.layup_dunk_class_id = layup_dunk_class_id
        self.ball_conf_thresh = 0.10
        self.has_jump_shot = False
        self.has_layup_dunk = False
        self.has_ball_in_basket = False
        self.player_persistent_db = {}

        # Runtime inference settings. Defaults intentionally match the old hard-coded baseline
        # so beta runs can be compared before changing detector behaviour.
        self.device = device
        self.detector_conf = float(detector_conf)
        self.imgsz = int(imgsz)
        self.iou = float(iou)
        self.tracker = tracker
        self.max_det = int(max_det)

        # Load team UI colors dynamically from JSON
        self.team_colors = self._load_ui_colors(profiling_path)
        self.team_colors["Referees"] = (0, 215, 255)

    def _load_ui_colors(self, profiling_path: str) -> dict:
        """Parses match JSON and extracts a primary drawing color for each team."""
        colors = {}
        try:
            with open(profiling_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rosters = data.get("rosters", data.get("teams", data))

            team_index = 0
            for team_name, team_data in rosters.items():
                kits = team_data.get("kits", {})
                active_kit = "home" if team_index == 0 else "away"

                primary_color_str = None
                if active_kit in kits and kits[active_kit]:
                    primary_color_str = kits[active_kit][0]

                if primary_color_str and primary_color_str.title() in self.UI_COLOR_MAP:
                    colors[team_name] = self.UI_COLOR_MAP[primary_color_str.title()]
                else:
                    colors[team_name] = self.DEFAULT_COLOR

                team_index += 1

        except Exception as e:
            print(f"⚠️ Error loading UI colors from JSON: {e}")

        return colors

    # --------------------------------------------------
    # PIPELINE HELPER METHODS
    # --------------------------------------------------

    def _run_detector(self, frame: np.ndarray):
        """Step 1: Runs the YOLO tracking model on the frame."""
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

    @staticmethod
    def _is_player_class(cls_name: str) -> bool:
        return "person" in cls_name or "player" in cls_name

    @staticmethod
    def _is_ball_class(cls_name: str) -> bool:
        return "ball" in cls_name or "sports ball" in cls_name or "basketball" in cls_name

    def _parse_detections(self, frame_state, results):
        """
        Step 2: Extract players, balls, and action classes from YOLO results.

        The beta diagnostics deliberately record counts BEFORE and AFTER the court polygon
        filter. This lets us determine whether misses originate in the trained detector,
        spatial filtering, or later pipeline stages.
        """
        players = []
        best_ball = None
        max_ball_conf = 0.0

        has_jump_shot = False
        has_layup_dunk = False
        has_ball_in_basket = False

        diagnostics = {
            "raw_total": 0,
            "raw_class_counts": {},
            "raw_player_candidates": 0,
            "raw_ball_candidates": 0,
            "raw_ball_max_conf": None,
            "rejected_outside_court_total": 0,
            "rejected_outside_court_by_class": {},
            "parsed_players": 0,
            "selected_ball": False,
            "selected_ball_conf": None,
            "detector_conf": self.detector_conf,
            "imgsz": self.imgsz,
            "iou": self.iou,
            "tracker": self.tracker,
            "device": self.device,
        }

        if not results or results[0].boxes is None:
            frame_state.diagnostics["detector"] = diagnostics
            return players, best_ball, has_jump_shot, has_layup_dunk, has_ball_in_basket

        boxes = results[0].boxes
        polygon = getattr(frame_state, "court_polygon", None)
        polygon_array = None
        if polygon is not None and np.asarray(polygon).size > 0:
            candidate = np.array(polygon, np.int32)
            if len(candidate) >= 3:
                polygon_array = candidate

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = (self.model.names[cls_id] if cls_id in self.model.names else str(cls_id)).lower()

            diagnostics["raw_total"] += 1
            diagnostics["raw_class_counts"][cls_name] = diagnostics["raw_class_counts"].get(cls_name, 0) + 1

            is_player = self._is_player_class(cls_name)
            is_ball = self._is_ball_class(cls_name)

            if is_player:
                diagnostics["raw_player_candidates"] += 1
            if is_ball:
                diagnostics["raw_ball_candidates"] += 1
                current_max = diagnostics["raw_ball_max_conf"]
                diagnostics["raw_ball_max_conf"] = conf if current_max is None else max(current_max, conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = int(box.id[0]) if box.id is not None else -1
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Baseline court polygon behaviour is retained for this first beta.
            # Diagnostics tell us exactly which classes are lost here so a later run
            # can compare a ball-specific spatial policy without guesswork.
            if polygon_array is not None:
                if cv2.pointPolygonTest(polygon_array, (center_x, center_y), False) < 0:
                    diagnostics["rejected_outside_court_total"] += 1
                    diagnostics["rejected_outside_court_by_class"][cls_name] = (
                        diagnostics["rejected_outside_court_by_class"].get(cls_name, 0) + 1
                    )
                    continue

            if cls_id == self.jump_shot_class_id or "jump_shot" in cls_name:
                has_jump_shot = True
            elif cls_id == self.layup_dunk_class_id or "layup" in cls_name or "dunk" in cls_name:
                has_layup_dunk = True
            elif cls_id == self.ball_in_basket_class_id or "basket" in cls_name:
                has_ball_in_basket = True

            if is_player:
                if y2 > y1 and x2 > x1:
                    players.append(
                        {
                            "track_id": track_id,
                            "box": [x1, y1, x2, y2],
                            "center": (center_x, center_y),
                            "confidence": conf,
                            "team": None,
                            "shirt_num": None,
                        }
                    )

            elif is_ball:
                if conf > max_ball_conf and conf >= self.ball_conf_thresh:
                    max_ball_conf = conf
                    best_ball = {
                        "box": [x1, y1, x2, y2],
                        "center": (center_x, center_y),
                        "confidence": conf,
                        "interpolated": False,
                    }

        diagnostics["parsed_players"] = len(players)
        diagnostics["selected_ball"] = best_ball is not None
        diagnostics["selected_ball_conf"] = max_ball_conf if best_ball is not None else None
        frame_state.diagnostics["detector"] = diagnostics

        return players, best_ball, has_jump_shot, has_layup_dunk, has_ball_in_basket

    def _assign_teams(self, frame_state):
        """Step 3: Delegates team classification to VisualAnalyzer."""
        frame = frame_state.frame
        for player in frame_state.players:
            track_id = player["track_id"]
            x1, y1, x2, y2 = player["box"]
            player_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

            if track_id != -1:
                if track_id not in self.player_persistent_db:
                    self.player_persistent_db[track_id] = {"team": None, "shirt_num": None}

                if not self.player_persistent_db[track_id]["team"]:
                    self.player_persistent_db[track_id]["team"] = self.visual.get_team_color(player_crop)

                player["team"] = self.player_persistent_db[track_id]["team"]
            else:
                player["team"] = self.visual.get_team_color(player_crop)

    def _update_player_numbers(self, frame_state):
        """Step 4: Delegates temporal jersey recognition to VisualAnalyzer."""
        frame = frame_state.frame

        for player in frame_state.players:
            track_id = player["track_id"]
            team = player["team"]
            x1, y1, x2, y2 = player["box"]

            player_crop = frame[
                max(0, y1):min(frame.shape[0], y2),
                max(0, x1):min(frame.shape[1], x2),
            ]

            player["ocr_attempted"] = False
            player["ocr_success"] = False

            if track_id != -1:
                profile = self.player_persistent_db[track_id]

                if not profile.get("shirt_num") or profile.get("shirt_num") == "...":
                    player["ocr_attempted"] = True
                    num = self.visual.get_shirt_number(
                        player_crop,
                        track_id=track_id,
                        team_name=team,
                    )

                    if num:
                        profile["shirt_num"] = num
                        player["ocr_success"] = True

                player["shirt_num"] = profile.get("shirt_num")

            else:
                # Do not feed all untracked players into the same temporal track_id=-1 buffer.
                # This beta records them as unresolved; jersey temporal inference resumes once
                # the player has a stable tracker ID.
                player["shirt_num"] = None

    def _detect_backboards(self, frame_state):
        """Step 5: Delegates backboard detection to VisualAnalyzer."""
        roi = getattr(frame_state, "backboard_roi", None)
        backboard = self.visual.detect_backboard(frame_state.frame, roi=roi)
        return [backboard] if backboard else []

    def _estimate_rims(self, backboards):
        """Step 6: Delegates rim geometric estimation to VisualAnalyzer."""
        return self.visual.estimate_rims(backboards)

    def _draw_annotations(self, frame_state, backboards=None, rims=None):
        """Step 7: Renders UI elements directly onto the frame."""
        if backboards is None:
            backboards = []
        if rims is None:
            rims = []

        frame = frame_state.frame
        if frame is None:
            return

        for player in getattr(frame_state, "players", []):
            self.draw_foot_marker(
                frame=frame,
                bbox=player["box"],
                team_name=player["team"],
                shirt_num=player["shirt_num"],
            )

        for bb in backboards:
            bx, by, bw, bh = bb["bbox"]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)

        for rim in rims:
            rx, ry = rim["center"]
            r = rim["radius"]
            cv2.circle(frame, (rx, ry), r, (0, 0, 255), 2)

    def draw_foot_marker(self, frame: np.ndarray, bbox: list, team_name: Optional[str], shirt_num: str):
        """Draws the ellipse at the player's feet and their team/number label."""
        x1, y1, x2, y2 = bbox
        x_center = int((x1 + x2) / 2)
        y_foot = int(y2)
        box_width = int(x2 - x1)

        color = self.team_colors.get(team_name, self.DEFAULT_COLOR)

        cv2.ellipse(
            frame,
            center=(x_center, y_foot),
            axes=(box_width // 2, int(box_width * 0.2)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        if team_name or shirt_num:
            label = f"{team_name or 'Unknown'} {shirt_num or ''}".strip()
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                frame,
                (x_center - text_width // 2 - 5, y_foot + 10),
                (x_center + text_width // 2 + 5, y_foot + 10 + text_height + 5),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x_center - text_width // 2, y_foot + 10 + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------

    def process(self, frame_state):
        if frame_state.frame is None:
            return frame_state

        results = self._run_detector(frame_state.frame)
        if results is None:
            return frame_state

        players, ball, has_jump, has_layup, has_basket = self._parse_detections(frame_state, results)
        frame_state.players = players
        frame_state.ball = ball

        self._assign_teams(frame_state)
        self._update_player_numbers(frame_state)

        backboards = self._detect_backboards(frame_state)
        rims = self._estimate_rims(backboards)
        self._draw_annotations(frame_state, backboards, rims)

        frame_state.diagnostics.setdefault("actions", {})
        frame_state.diagnostics["actions"].update(
            {
                "jump_shot_class_seen": bool(has_jump),
                "layup_dunk_class_seen": bool(has_layup),
                "ball_in_basket_class_seen": bool(has_basket),
            }
        )

        return frame_state
