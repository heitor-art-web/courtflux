import numpy as np
import supervision as sv
from collections import deque
from typing import Dict, Any, Optional

class Engine:
    def __init__(self, fps: int = 30, history_window_sec: float = 0.5, rim_pixel_coords=None):
        self.fps = fps
        self.player_y_history: Dict[int, deque] = {}
        self.history_frames = int(fps * history_window_sec)
        
        self.player_class_id = 5
        self.ball_class_id = 3

        # Game Analytics & Event States
        self.rim_coords = rim_pixel_coords
        self.prev_ball_pos = None
        self.last_known_ball = None
        self.ball_lost_frames = 0
        self.current_possession_team = None
        self.last_shot_frame = -100

    def process(self, frame_state: Any) -> Any:
        current_frame_id = getattr(frame_state, "frame_id", 0)
        H = getattr(frame_state, "homography_matrix", None)

        # 1. Extract and interpolate ball state from frame_state.ball
        ball_px = self._extract_ball_center(frame_state)
        if frame_state.ball is not None:
            self.last_known_ball = frame_state.ball
            self.ball_lost_frames = 0
        else:
            self.ball_lost_frames += 1

        if self.ball_lost_frames < 30 and self.last_known_ball:
            frame_state.ball = self.last_known_ball 
            if isinstance(frame_state.ball, dict):
                frame_state.ball["interpolated"] = True

        # 2. Extract players from frame_state.players populated by detector
        raw_players = getattr(frame_state, "players", [])
        
        # 🔍 DIAGNOSTIC PRINT (Every 100 frames)
        if current_frame_id > 0 and current_frame_id % 100 == 0:
            print(
                f"🔍 [DEBUG Frame {current_frame_id}] Ball detected:"
                f" {ball_px is not None} | Players count: {len(raw_players)} |"
                f" Homography active: {H is not None} |"
                f" Possession team: {self.current_possession_team}"
            )

        jump_events_count = 0
        players_list = []

        for p in raw_players:
            if isinstance(p, dict):
                tracker_id = p.get("track_id") or p.get("id")
                bbox = p.get("box") or p.get("bbox")
            else:
                tracker_id = getattr(p, "track_id", None) or getattr(p, "id", None)
                bbox = getattr(p, "box", None) or getattr(p, "bbox", None)

            if tracker_id is None or bbox is None:
                continue

            x_min, y_min, x_max, y_max = bbox
            feet_y = y_max  
            foot_point = ((x_min + x_max) / 2.0, y_max)
            center_point = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

            # Update vertical history for jump detection
            if tracker_id not in self.player_y_history:
                self.player_y_history[tracker_id] = deque(maxlen=self.history_frames)
            self.player_y_history[tracker_id].append(feet_y)

            is_jumping = self._check_vertical_jump(tracker_id, jump_threshold_px=12)
            has_ball = self._check_ball_proximity(bbox, ball_px, expansion=30)

            on_court = True
            if H is not None:
                on_court = self._is_on_court(foot_point, H)

            # Assign label directly inside player dict instead of a separate attribute
            if is_jumping and has_ball and on_court:
                label = f"ID {tracker_id}: JUMP SHOT!"
                jump_events_count += 1
                if hasattr(frame_state, "add_event"):
                    frame_state.add_event("JUMP_SHOT", {"player_id": tracker_id})
            else:
                label = f"ID {tracker_id}"

            p_dict = {
                "track_id": tracker_id,
                "bbox": bbox,
                "center": center_point,
                "feet": foot_point,
                "label": label  # Renderer can read this directly
            }
            players_list.append(p_dict)

        # 3. Check for Rim Shots & Turnovers
        shot_triggered = False
        if self.rim_coords and ball_px and self.prev_ball_pos:
            shot_triggered = self.check_rim_interaction(ball_px, self.prev_ball_pos)

        if not shot_triggered:
            shot_triggered = self.check_and_log_shots(current_frame_id, ball_px, H, players_list)

        turnover_triggered = self.check_turnovers(frame_state, ball_px)

        # 4. Update ball position history
        if ball_px is not None:
            self.prev_ball_pos = ball_px

        # 5. Package results back into valid FrameState attributes
        if not hasattr(frame_state, "events") or frame_state.events is None:
            frame_state.events = []

        if jump_events_count > 0 or shot_triggered:
            if "SHOT_ATTEMPT" not in frame_state.events:
                frame_state.events.append("SHOT_ATTEMPT")
            frame_state.is_snapshot_moment = True
            print(f"\n🎯 [EVENTO] Lançamento detetado no frame {current_frame_id}!")

        if turnover_triggered:
            frame_state.events.append("TURNOVER")
            frame_state.is_snapshot_moment = True
            print(f"\n🔄 [EVENTO] Turnover detetado no frame {current_frame_id}!")

        # Only assign attributes that officially exist in FrameState schema
        frame_state.players = players_list

        return frame_state

    def _extract_ball_center(self, frame_state) -> Optional[np.ndarray]:
        ball = getattr(frame_state, "ball", None)
        if not ball:
            return None
        if isinstance(ball, dict):
            if "center" in ball and ball["center"]:
                return np.array(ball["center"])
            box = ball.get("box") or ball.get("bbox")
            if box and len(box) >= 4:
                return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
        return None

    def _check_vertical_jump(self, tracker_id: int, jump_threshold_px: int) -> bool:
        history = self.player_y_history[tracker_id]
        if len(history) < 5:
            return False
        oldest_y = history[0]
        current_y = history[-1]
        y_delta = oldest_y - current_y  
        return y_delta > jump_threshold_px

    def _check_ball_proximity(self, player_bbox, ball_center, expansion: int) -> bool:
        if ball_center is None:
            return False
        x_min, y_min, x_max, y_max = player_bbox
        return (
            (x_min - expansion <= ball_center[0] <= x_max + expansion) and 
            (y_min - expansion <= ball_center[1] <= y_max + expansion)
        )

    def _is_on_court(self, foot_xy: tuple, H: np.ndarray, margin_meters: float = 0.5) -> bool:
        try:
            point = np.array([foot_xy[0], foot_xy[1], 1.0], dtype=np.float32).reshape(3, 1)
            world_pt = np.dot(H, point)
            if world_pt[2] == 0:
                return True
            x_m = world_pt[0][0] / world_pt[2][0]
            y_m = world_pt[1][0] / world_pt[2][0]
            return (-margin_meters <= x_m <= 28.0 + margin_meters) and (-margin_meters <= y_m <= 15.0 + margin_meters)
        except Exception:
            return True

    def check_rim_interaction(self, ball_pos, previous_ball_pos) -> bool:
        if ball_pos is None or previous_ball_pos is None or not self.rim_coords:
            return False
        bx, by = ball_pos
        prev_bx, prev_by = previous_ball_pos
        rx, ry = self.rim_coords
        distance_to_rim = ((bx - rx) ** 2 + (by - ry) ** 2) ** 0.5
        return distance_to_rim < 45 and by > prev_by and prev_by <= ry

    def check_and_log_shots(self, frame_count, ball_pos_px, H, players) -> bool:
        if (frame_count - self.last_shot_frame) < 60:
            return False
        shot_triggered = False
        if players and H is not None:
            for p in players:
                try:
                    box = p.get("bbox")
                    if box is None:
                        continue
                    cx, cy = (box[0] + box[2]) / 2.0, box[3]
                    pt = np.array([cx, cy, 1.0], dtype=np.float32)
                    proj = np.dot(H, pt)
                    if proj[2] == 0:
                        continue
                    px_m, py_m = proj[0] / proj[2], proj[1] / proj[2]
                    if np.sqrt((px_m - 1.6) ** 2 + (py_m - 7.5) ** 2) <= 3.5 or np.sqrt((px_m - 26.4) ** 2 + (py_m - 7.5) ** 2) <= 3.5:
                        shot_triggered = True
                        break
                except Exception:
                    continue
        if shot_triggered:
            self.last_shot_frame = frame_count
            return True
        return False

    def check_turnovers(self, frame_state, ball_px) -> bool:
        players = getattr(frame_state, "players", [])
        if ball_px is None or not players:
            return False
        bx, by = ball_px
        closest_player = None
        min_dist = float("inf")
        for p in players:
            center = p.get("center")
            if not center:
                box = p.get("box") or p.get("bbox")
                if box and len(box) >= 4:
                    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            if center:
                dist = np.sqrt((center[0] - bx) ** 2 + (center[1] - by) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_player = p
        if closest_player and min_dist < 80:
            player_team = closest_player.get("team") or closest_player.get("team_id")
            if player_team is not None:
                if self.current_possession_team is not None and self.current_possession_team != player_team:
                    self.current_possession_team = player_team
                    return True
                self.current_possession_team = player_team
        return False