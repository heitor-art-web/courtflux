import cv2
import numpy as np
from enum import Enum, auto
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

# ==========================================
# 1. SHOT EVENT TRACKER (STATE MACHINE)
# ==========================================
class ShotState(Enum):
    IDLE = auto()
    SHOT_IN_FLIGHT = auto()
    COOLDOWN = auto()


class ShotEventTracker:
    def __init__(
        self,
        reset_time_frames: int,
        minimum_frames_between_starts: int,
        cooldown_frames_after_made: int,
    ):
        self.reset_time_frames = reset_time_frames
        self.minimum_frames_between_starts = minimum_frames_between_starts
        self.cooldown_frames_after_made = cooldown_frames_after_made

        self.state = ShotState.IDLE
        self.shot_start_frame: Optional[int] = None
        self.last_shot_start_frame: int = -9999
        self.cooldown_start_frame: Optional[int] = None
        self.current_shot_type: Optional[str] = None

    def update(
        self,
        frame_index: int,
        has_jump_shot: bool,
        has_layup_dunk: bool,
        has_ball_in_basket: bool,
    ) -> List[Dict[str, Any]]:
        events = []

        if self.state == ShotState.COOLDOWN:
            if frame_index - self.cooldown_start_frame >= self.cooldown_frames_after_made:
                self.state = ShotState.IDLE
            else:
                return events

        if self.state == ShotState.IDLE:
            is_attempt = has_jump_shot or has_layup_dunk
            can_start = (frame_index - self.last_shot_start_frame) >= self.minimum_frames_between_starts

            if is_attempt and can_start:
                self.state = ShotState.SHOT_IN_FLIGHT
                self.shot_start_frame = frame_index
                self.last_shot_start_frame = frame_index
                self.current_shot_type = "layup_dunk" if has_layup_dunk else "jump_shot"

                if has_ball_in_basket:
                    events.append({
                        "event": "SHOT_MADE",
                        "shot_type": self.current_shot_type,
                        "frame": frame_index,
                        "duration_frames": 0,
                    })
                    self.state = ShotState.COOLDOWN
                    self.cooldown_start_frame = frame_index

            return events

        if self.state == ShotState.SHOT_IN_FLIGHT:
            elapsed_frames = frame_index - self.shot_start_frame

            if has_ball_in_basket:
                events.append({
                    "event": "SHOT_MADE",
                    "shot_type": self.current_shot_type,
                    "frame": frame_index,
                    "duration_frames": elapsed_frames,
                })
                self.state = ShotState.COOLDOWN
                self.cooldown_start_frame = frame_index

            elif elapsed_frames >= self.reset_time_frames:
                events.append({
                    "event": "SHOT_MISSED",
                    "shot_type": self.current_shot_type,
                    "frame": frame_index,
                    "duration_frames": elapsed_frames,
                })
                self.state = ShotState.IDLE

        return events


# ==========================================
# 2. UNIFIED DETECTOR CLASS
# ==========================================
class Detector:
    TEAM_COLORS = {
        "FC Porto": (220, 100, 0),     # Blue
        "Ovarense": (40, 40, 40),      # Black / Dark Gray
        "Referees": (0, 215, 255),     # Yellow
    }
    DEFAULT_COLOR = (180, 180, 180)    # Gray fallback

    def __init__(
        self,
        model: YOLO,
        visual_analyzer,
        fps: int = 30,
        ball_in_basket_class_id: int = 1,
        jump_shot_class_id: int = 5,
        layup_dunk_class_id: int = 6,
        reader=None
    ):
        self.model = model
        self.visual = visual_analyzer
        self.reader = reader
        self._ocr_reader = None

        self.ball_in_basket_class_id = ball_in_basket_class_id
        self.jump_shot_class_id = jump_shot_class_id
        self.layup_dunk_class_id = layup_dunk_class_id

        self.shot_tracker = ShotEventTracker(
            reset_time_frames=int(fps * 1.7),
            minimum_frames_between_starts=int(fps * 0.5),
            cooldown_frames_after_made=int(fps * 0.5),
        )

        self.player_persistent_db = {}
        # Lowered ball confidence threshold to match global track conf (0.10)
        self.ball_conf_thresh = 0.10  

    @property
    def ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(['en'], gpu=True)
        return self._ocr_reader

    def draw_foot_marker(self, frame: np.ndarray, bbox: list, team_name: Optional[str], shirt_num: str):
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) // 2
        y_foot = y2
        box_width = x2 - x1

        color = self.TEAM_COLORS.get(team_name, self.DEFAULT_COLOR)

        ellipse_rx = max(12, int(box_width * 0.40))
        ellipse_ry = max(4, int(ellipse_rx * 0.35))
        
        cv2.ellipse(frame, (x_center, y_foot), (ellipse_rx + 1, ellipse_ry + 1), 0, 0, 360, (0, 0, 0), 2)
        cv2.ellipse(frame, (x_center, y_foot), (ellipse_rx, ellipse_ry), 0, 0, 360, color, -1)

        if shirt_num and shirt_num != "...":
            label = f"{team_name} #{shirt_num}" if team_name else f"#{shirt_num}"
        else:
            label = team_name if team_name else "Player"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        padding = 4
        badge_x1 = x_center - (text_w // 2) - padding
        badge_y1 = y_foot + ellipse_ry + 4
        badge_x2 = x_center + (text_w // 2) + padding
        badge_y2 = badge_y1 + text_h + (padding * 2)

        h_frame, w_frame = frame.shape[:2]
        if badge_x1 < 0: badge_x1, badge_x2 = 2, 2 + text_w + (padding * 2)
        if badge_x2 > w_frame: badge_x2, badge_x1 = w_frame - 2, w_frame - 2 - text_w - (padding * 2)
        if badge_y2 > h_frame: badge_y1, badge_y2 = y_foot - ellipse_ry - text_h - (padding * 2), y_foot - ellipse_ry

        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 20), -1)
        cv2.line(frame, (badge_x1, badge_y1), (badge_x2, badge_y1), color, 2)
        cv2.putText(
            frame, 
            label, 
            (badge_x1 + padding, badge_y2 - padding - 1), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness, 
            cv2.LINE_AA
        )

    def process(self, frame_state):
        frame = frame_state.frame
        frame_count = frame_state.frame_id

        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=0.10,
            iou=0.5,
            imgsz=640,
            device="mps",
            verbose=False,
        )

        players = []
        best_ball = None
        max_ball_conf = 0.0

        has_jump_shot = False
        has_layup_dunk = False
        has_ball_in_basket = False

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = (self.model.names[cls_id] if cls_id in self.model.names else str(cls_id)).lower()

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else -1
                
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                # Safe Court Polygon Boundary Check
                polygon = getattr(frame_state, 'court_polygon', None)
                if polygon is not None and np.asarray(polygon).size > 0:
                    polygon_array = np.array(polygon, np.int32)
                    if len(polygon_array) >= 3:
                        if cv2.pointPolygonTest(polygon_array, (center_x, center_y), False) < 0:
                            continue

                if cls_id == self.jump_shot_class_id or "jump_shot" in cls_name:
                    has_jump_shot = True
                elif cls_id == self.layup_dunk_class_id or "layup" in cls_name or "dunk" in cls_name:
                    has_layup_dunk = True
                elif cls_id == self.ball_in_basket_class_id or "basket" in cls_name:
                    has_ball_in_basket = True

                if "person" in cls_name or "player" in cls_name:
                    if y2 <= y1 or x2 <= x1:
                        continue

                    player_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

                    if track_id != -1:
                        if track_id not in self.player_persistent_db:
                            team = self.visual.get_team_color(player_crop)
                            shirt = self.visual.get_shirt_number(player_crop, track_id=track_id, team_name=team)
                            self.player_persistent_db[track_id] = {
                                "team": team,
                                "shirt_num": shirt,
                            }
                        else:
                            profile = self.player_persistent_db[track_id]
                            if not profile["team"]:
                                profile["team"] = self.visual.get_team_color(player_crop)
                            if not profile["shirt_num"] or profile["shirt_num"] == "...":
                                num = self.visual.get_shirt_number(
                                    player_crop, 
                                    track_id=track_id, 
                                    team_name=profile["team"]
                                )
                                if num:
                                    profile["shirt_num"] = num
                        profile = self.player_persistent_db[track_id]
                    else:
                        team = self.visual.get_team_color(player_crop)
                        shirt = self.visual.get_shirt_number(player_crop, track_id=-1, team_name=team)
                        profile = {"team": team, "shirt_num": shirt}

                    self.draw_foot_marker(
                        frame=frame, 
                        bbox=[x1, y1, x2, y2], 
                        team_name=profile["team"], 
                        shirt_num=profile["shirt_num"]
                    )

                    players.append({
                        "track_id": track_id,
                        "team": profile["team"],
                        "shirt_num": profile["shirt_num"],
                        "box": [x1, y1, x2, y2],
                        "center": (center_x, center_y),
                    })

                # Expanded Ball Detection Logic (Supports custom model class names like "basketball")
                elif "ball" in cls_name or "sports ball" in cls_name or "basketball" in cls_name:
                    if conf > max_ball_conf and conf >= self.ball_conf_thresh:
                        max_ball_conf = conf
                        best_ball = {
                            "box": [x1, y1, x2, y2],
                            "center": (center_x, center_y),
                            "confidence": conf
                        }

        frame_state.players = players
        frame_state.ball = best_ball
        
        frame_state.events = self.shot_tracker.update(
            frame_index=frame_count,
            has_jump_shot=has_jump_shot,
            has_layup_dunk=has_layup_dunk,
            has_ball_in_basket=has_ball_in_basket,
        )

        return frame_state