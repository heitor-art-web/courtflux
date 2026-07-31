import cv2
import numpy as np

class RadarVisualizer:
    def __init__(self, scale: int = 25, team_colors: dict = None):
        # FIBA Court is 28m x 15m. A scale of 25 makes a 700x375 pixel radar canvas.
        self.scale = scale
        self.court_w = 28 * self.scale
        self.court_h = 15 * self.scale
        
        # BGR Colors for tactical drawing
        self.bg_color = (40, 48, 43)      # Dark tactical green
        self.line_color = (200, 220, 200) # Off-white lines
        
        # Team colors configuration injected from match.yaml
        self.team_colors = team_colors or {
            "home": (230, 230, 230),
            "away": (20, 20, 20),
            "Unknown": (128, 128, 128)
        }

    def _draw_fiba_court(self) -> np.ndarray:
        """Creates the tactical 2D court background layout."""
        radar = np.full((self.court_h, self.court_w, 3), self.bg_color, dtype=np.uint8)
        t = 2 # line thickness

        # 1. Outer Boundary
        cv2.rectangle(radar, (0, 0), (self.court_w, self.court_h), self.line_color, t)
        
        # 2. Center Line & Center Circle (1.8m radius)
        mid_x = 14 * self.scale
        cv2.line(radar, (mid_x, 0), (mid_x, self.court_h), self.line_color, t)
        cv2.circle(radar, (mid_x, int(7.5 * self.scale)), int(1.8 * self.scale), self.line_color, t)
        
        # 3. Keys (Paint) - 5.8m width, 4.9m height
        key_top = int(5.05 * self.scale)
        key_bot = int(9.95 * self.scale)
        cv2.rectangle(radar, (0, key_top), (int(5.8 * self.scale), key_bot), self.line_color, t)
        cv2.rectangle(radar, (self.court_w - int(5.8 * self.scale), key_top), (self.court_w, key_bot), self.line_color, t)
        
        # 4. Rims
        cv2.circle(radar, (int(1.575 * self.scale), int(7.5 * self.scale)), int(0.2 * self.scale), (0, 100, 255), -1)
        cv2.circle(radar, (self.court_w - int(1.575 * self.scale), int(7.5 * self.scale)), int(0.2 * self.scale), (0, 100, 255), -1)

        return radar

    def render(self, frame_state, camera_tracker) -> np.ndarray:
        """Generates the radar frame with player dots mapped to their real-world coords."""
        radar = self._draw_fiba_court()

        H = getattr(frame_state, "homography_matrix", None)
        if H is None:
            return radar

        players = getattr(frame_state, "players", [])
        for player in players:
            # Use pre-calculated feet point from engine
            foot_xy = player.get("feet") or player.get("center")
            if foot_xy is None:
                continue
            
            # Transform Image Pixels -> 2D Court Meters
            world_coords = camera_tracker.transform_feet_to_2d_court(tuple(foot_xy), H)
            
            if world_coords:
                x_m, y_m = world_coords
                
                # Check boundaries (+ 1 meter buffer for out-of-bounds plays)
                if -1.0 <= x_m <= 29.0 and -1.0 <= y_m <= 16.0:
                    px = int(x_m * self.scale)
                    py = int(y_m * self.scale)

                    # Resolve team color using match configuration mapping
                    team = player.get("team") or player.get("team_id") or "Unknown"
                    color = self.team_colors.get(team, self.team_colors.get("Unknown", (128, 128, 128)))

                    # Draw tactical player dot
                    cv2.circle(radar, (px, py), 10, color, -1)
                    cv2.circle(radar, (px, py), 10, (255, 255, 255), 1) # Outline

                    # Overlay track ID or jersey number
                    num = player.get("track_id", "")
                    if num is not None and str(num) != "":
                        label_str = str(num)
                        text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        txt_x = px - (text_size[0] // 2)
                        txt_y = py + (text_size[1] // 2)
                        cv2.putText(radar, label_str, (txt_x, txt_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        return radar