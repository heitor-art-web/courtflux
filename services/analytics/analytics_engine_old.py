from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Define root and ensure snapshots directory exists
ROOT_DIR = (
    Path(__file__).resolve().parent.parent.parent
)  # Adjust depending on file depth
SNAPSHOT_DIR = ROOT_DIR / "data" / "events"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


class GameAnalyticsEngine:

  def __init__(self, cursor, fps, rim_pixel_coords=None):
    self.cursor = cursor
    self.fps = fps
    self.player_prev_pos = {}
    self.player_prev_time = {}
    self.ball_handler_id = None
    self.rim_coords = rim_pixel_coords
    self.prev_ball_pos = None  # Tracks the ball position from the last frame
    self.current_possession_team = (
        None  # Tracks current team in possession for turnovers
    )
    self.last_shot_frame = -100

    # Transition tracking state
    self.turnover_frame = None
    self.transition_active = False
    self.last_known_ball = None
    self.ball_lost_frames = 0

  def compute_speed(self, track_id, x_m, y_m, frame_count) -> float:
    """Calculates instantaneous player speed in meters per second (m/s)."""
    if track_id not in self.player_prev_pos:
      self.player_prev_pos[track_id] = (x_m, y_m)
      self.player_prev_time[track_id] = frame_count
      return 0.0

    prev_x, prev_y = self.player_prev_pos[track_id]
    prev_frame = self.player_prev_time[track_id]

    dx = x_m - prev_x
    dy = y_m - prev_y
    dist = np.sqrt(dx**2 + dy**2)

    frames_elapsed = frame_count - prev_frame
    if frames_elapsed <= 0:
      return 0.0

    time_elapsed = frames_elapsed / self.fps
    speed = dist / time_elapsed

    # Safeguard against unrealistic tracking jumps (scene cuts / tracking errors)
    if speed > 12.0:
      return 0.0

    self.player_prev_pos[track_id] = (x_m, y_m)
    self.player_prev_time[track_id] = frame_count
    return float(speed)

  def process_possession_and_passes(self, frame_count, players, ball_pos_px):
    """Determines which player currently has possession of the ball based on proximity."""
    if ball_pos_px is None or not players:
      return None, None

    bx, by = ball_pos_px
    closest_player = None
    min_dist = float("inf")

    for p in players:
      # Use bounding box center or feet
      box = p.get("bbox") or p.get("box")
      if not box:
        continue

      px_center = (box[0] + box[2]) / 2
      py_center = (box[1] + box[3]) / 2

      dist = np.sqrt((px_center - bx) ** 2 + (py_center - by) ** 2)
      if dist < min_dist:
        min_dist = dist
        closest_player = p

    # If within 40 pixels, consider it ball possession
    if min_dist < 40 and closest_player:
      self.ball_handler_id = closest_player.get("track_id")
      return self.ball_handler_id, closest_player

    return None, None

  def compute_defender_distance(self, ball_handler, players, offensive_team):
    """Finds the distance from the ball handler to the nearest opposing defender."""
    if not ball_handler or not offensive_team:
      return None

    bx, by = ball_handler["x_m"], ball_handler["y_m"]
    min_dist = float("inf")

    for p in players:
      if p.get("team") != offensive_team:
        dist = np.sqrt((p["x_m"] - bx) ** 2 + (p["y_m"] - by) ** 2)
        if dist < min_dist:
          min_dist = dist

    return float(min_dist) if min_dist != float("inf") else None

  def log_team_metrics(self, frame_count, players, offensive_team):
    """Logs team-level telemetry into the database."""
    pass  # Placeholder for DB batch inserts if needed

  def check_rim_interaction(self, ball_pos, previous_ball_pos):
    """Detects if the ball passes through or near the rim."""
    if not ball_pos or not previous_ball_pos or not self.rim_coords:
      return False

    bx, by = ball_pos
    prev_bx, prev_by = previous_ball_pos
    rx, ry = self.rim_coords

    # Define a bounding zone around the rim (45-pixel radius)
    distance_to_rim = ((bx - rx) ** 2 + (by - ry) ** 2) ** 0.5

    # Check if ball is close to the rim AND moving downwards (pixel Y increases downwards)
    is_near_rim = distance_to_rim < 45
    is_moving_down = by > prev_by

    # If it was above/near the rim previously and drops through
    if is_near_rim and is_moving_down and prev_by <= ry:
      return True

    return False

  def check_and_log_shots(
      self, frame_count, ball_pos_px, H, players, ball_handler=None
  ):
    """Detects shots and dunks using both floor proximity AND YOLO-Pose skeleton

    (detecting raised arms/wrists near the hoop).
    """
    if (frame_count - self.last_shot_frame) < 60:
      return False

    shot_triggered = False

    # 1. POSE & PROXIMITY CHECK (Leveraging YOLO-pose keypoints)
    if players and H is not None:
      for p in players:
        try:
          box = p.get("box") or p.get("bbox")
          if box is None:
            continue

          cx = (box[0] + box[2]) / 2.0
          cy = box[3]  # feet position

          # Project feet to metric space
          pt = np.array([cx, cy, 1.0], dtype=np.float32)
          proj = np.dot(H, pt)
          if proj[2] == 0:
            continue

          px_m = proj[0] / proj[2]
          py_m = proj[1] / proj[2]

          # Check distance to FIBA hoops (1.6m or 26.4m)
          left_hoop_dist = np.sqrt((px_m - 1.6) ** 2 + (py_m - 7.5) ** 2)
          right_hoop_dist = np.sqrt((px_m - 26.4) ** 2 + (py_m - 7.5) ** 2)

          if left_hoop_dist <= 3.5 or right_hoop_dist <= 3.5:
            keypoints = p.get("keypoints", None)
            if keypoints is not None and len(keypoints) >= 11:
              # Keypoints format: 5=L Shoulder, 6=R Shoulder, 9=L Wrist, 10=R Wrist
              l_wrist_y, r_wrist_y = keypoints[9][1], keypoints[10][1]
              l_shoulder_y, r_shoulder_y = keypoints[5][1], keypoints[6][1]

              if (l_wrist_y < l_shoulder_y) or (r_wrist_y < r_shoulder_y):
                shot_triggered = True
                break
            else:
              shot_triggered = True
              break
        except Exception:
          continue

    # 2. JUMPER / BALL WARP CHECK
    if not shot_triggered and ball_pos_px is not None and H is not None:
      bx, by = ball_pos_px
      try:
        point = np.array([bx, by, 1.0], dtype=np.float32)
        projected = np.dot(H, point)
        if projected[2] != 0:
          x_m = projected[0] / projected[2]
          y_m = projected[1] / projected[2]
          if (x_m < 5.5 and 1.0 < y_m < 14.0) or (x_m > 22.5 and 1.0 < y_m < 14.0):
            shot_triggered = True
      except Exception:
        pass

    if shot_triggered:
      self.last_shot_frame = frame_count
      return True

    return False

  def check_turnovers(self, frame_state,ball_px):
      """Detects turnovers with flexible key checks (team / team_id / color)."""
      players = getattr(frame_state, "players", [])
      if not ball_px or not players:
        return False

      bx, by = ball_px
      closest_player = None
      min_dist = float("inf")

      for p in players:
        # Get player center position
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

      # Increased distance threshold to 80px to accommodate high-res video
      if closest_player and min_dist < 80:
        # Check team attribute using multiple common keys
        player_team = (
            closest_player.get("team")
            if closest_player.get("team") is not None
            else closest_player.get("team_id")
        )

        if player_team is not None:
          if (
              self.current_possession_team is not None
              and self.current_possession_team != player_team
          ):
            self.current_possession_team = player_team
            return True
          self.current_possession_team = player_team

      return False

  # --- PRO TACTICAL MODULES ---

  def check_spacing_and_paint(self, players, ball_handler):
    """Calculates Corner Occupancy and Paint Touches based on FIBA 28x15m court."""
    corner_occupied = False
    in_paint = False

    if not ball_handler:
      return corner_occupied, in_paint

    for p in players:
      x, y = p.get("x_m", 0), p.get("y_m", 0)

      if (x < 5.0 or x > 23.0) and (y < 3.0 or y > 12.0):
        corner_occupied = True

      if (5.8 >= x >= 0 or 28 >= x >= 22.2) and (3.0 <= y <= 12.0):
        if p.get("team") == ball_handler.get("team"):
          in_paint = True

    return corner_occupied, in_paint

  def check_transition_recovery(
      self, frame_count, turnover_detected, players, offensive_team
  ):
    """Measures how many seconds it takes defenders to cross half-court (x=14m) after a turnover."""
    if turnover_detected:
      self.turnover_frame = frame_count
      self.transition_active = True
      return None

    if self.transition_active and self.turnover_frame:
      defenders = [p for p in players if p.get("team") != offensive_team]
      if defenders:
        all_recovered = all(p.get("x_m", 99) <= 14.0 for p in defenders)
        if all_recovered or (frame_count - self.turnover_frame) > (
            self.fps * 6
        ):
          recovery_time = (frame_count - self.turnover_frame) / self.fps
          self.transition_active = False
          return recovery_time
    return None

  def get_workload_profiles(self, players):
    """Classifies player speeds into Jogging, Running, and Explosive Sprint."""
    profiles = {}
    for p in players:
      speed = p.get("speed_mps", 0.0)
      tid = p.get("track_id")
      if speed < 3.0:
        effort = "Jogging"
      elif speed <= 5.5:
        effort = "Running"
      else:
        effort = "Explosive Sprint"
      profiles[tid] = {"team": p.get("team"), "effort": effort, "speed": speed}
    return profiles

  def compute_convex_hull_area(self, players, offensive_team) -> float:
    """Calculates total square meters of court space occupied by the offense."""
    offense_players = [
        [p["x_m"], p["y_m"]]
        for p in players
        if p.get("team") == offensive_team and "x_m" in p and "y_m" in p
    ]
    if len(offense_players) < 3:
      return 0.0

    points = np.array(offense_players)
    try:
      hull = ConvexHull(points)
      return float(hull.volume)  # 2D volume = polygon area in m^2
    except QhullError:
      return 0.0

  def detect_pick_and_roll(self, players, ball_handler) -> bool:
    """Detects if an offensive player is screening near the ball handler."""
    if not ball_handler or "x_m" not in ball_handler:
      return False

    bx, by = ball_handler["x_m"], ball_handler["y_m"]
    team = ball_handler.get("team")

    for p in players:
      if (
          p.get("team") == team
          and p.get("track_id") != ball_handler.get("track_id")
          and "x_m" in p
      ):
        dist = np.sqrt((p["x_m"] - bx) ** 2 + (p["y_m"] - by) ** 2)
        if dist <= 1.5:
          return True
    return False

  def _extract_ball_center(self, frame_state):
      """Extracts ball (x, y) coordinates regardless of detector dictionary format."""
      ball = getattr(frame_state, "ball", None)
      if not ball or not isinstance(ball, dict):
        return None

      # Direct center if provided
      if "center" in ball and ball["center"]:
        return ball["center"]

      # Compute from bounding box if center is missing
      box = ball.get("box") or ball.get("bbox")
      if box and len(box) >= 4:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

      return None
  
  def process(self, frame_state):
    """Processes the FrameState safely, detecting both shots and turnovers."""
    H = getattr(frame_state, "homography", None)
    players = getattr(frame_state, "players", [])
    current_frame_id = getattr(frame_state, "frame_id", 0)

    # 1. Safely extract ball coordinates
    ball_px = self._extract_ball_center(frame_state)
    
    if frame_state.ball is not None:
      self.last_known_ball = frame_state.ball
      self.ball_lost_frames = 0
    else:
      self.ball_lost_frames += 1
      # If the ball is missing for less than 30 frames (1 second), 
      # assume it's still near its last known location/player
    if self.ball_lost_frames < 30 and self.last_known_ball:
        frame_state.ball = self.last_known_ball 
        frame_state.ball["interpolated"] = True # Flag it so you know it's a guess

    # Diagnostic print every 100 frames to verify pipeline health
    if current_frame_id > 0 and current_frame_id % 100 == 0:
        print(
            f"🔍 [DEBUG Frame {current_frame_id}] Ball detected:"
            f" {ball_px is not None} | Players count: {len(players)} |"
            f" Possession team: {self.current_possession_team}"
        )

    # 2. Check for Shot
    shot_triggered = False
    if self.rim_coords and ball_px and self.prev_ball_pos:
        shot_triggered = self.check_rim_interaction(
            ball_px, self.prev_ball_pos
        )

    if not shot_triggered:
        shot_triggered = self.check_and_log_shots(
            frame_count=current_frame_id,
            ball_pos_px=ball_px,
            H=H,
            players=players,
        )

      # 3. Check for Turnover
    turnover_triggered = self.check_turnovers(frame_state, ball_px)

      # 4. Update ball position history
    if ball_px is not None:
        self.prev_ball_pos = ball_px

    # 5. Handle event flags
    if not hasattr(frame_state, "events") or frame_state.events is None:
        frame_state.events = []
    if shot_triggered:
        frame_state.events.append("SHOT_ATTEMPT")
        frame_state.is_snapshot_moment = True
        print(f"\n🎯 [EVENTO] Lançamento detetado no frame {current_frame_id}!")

    if turnover_triggered:
        frame_state.events.append("TURNOVER")
        frame_state.is_snapshot_moment = True
        print(f"\n🔄 [EVENTO] Turnover detetado no frame {current_frame_id}!")

    return frame_state