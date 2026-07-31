from pathlib import Path
import cv2
import numpy as np


class Renderer:

  def __init__(self, save_snapshots: bool = True, snapshot_dir: str = None):
    self.save_snapshots = save_snapshots
    if snapshot_dir:
      self.snapshot_dir = Path(snapshot_dir)
    else:
      self.snapshot_dir = (
          Path(__file__).resolve().parent.parent.parent
          / "data"
          / "snapshots"
      )

    if self.save_snapshots:
      self.snapshot_dir.mkdir(parents=True, exist_ok=True)

  def _sanitize_polygon(self, polygon):
    """Safely converts numpy arrays, lists, or Shapely polygons into valid OpenCV int32 points."""
    if polygon is None:
      return None

    try:
      # Handle Shapely Polygon objects
      if hasattr(polygon, "exterior"):
        pts = np.array(polygon.exterior.coords)
      else:
        pts = np.array(polygon)

      # 1. Reject empty arrays or invalid shapes
      if pts.size == 0 or pts.ndim < 2:
        return None

      # 2. Reject NaNs or Infs (prevents OpenCV negative integer overflow)
      if np.isnan(pts).any() or np.isinf(pts).any():
        return None

      # 3. Ensure at least 3 points exist to form a closed polygon
      pts = pts.reshape(-1, 2)
      if len(pts) < 3:
        return None

      return pts.astype(np.int32).reshape((-1, 1, 2))

    except Exception:
      return None

  def draw(self, frame_state):
    """Draws tracking overlays safely onto the frame."""
    annotated = frame_state.frame.copy()
    frame = getattr(frame_state, "frame", None)
    # 1. Court Polygon (Safe Rendering)
    polygon = getattr(frame_state, "court_polygon", None)
    pts_np = self._sanitize_polygon(polygon)

    if pts_np is not None:
      cv2.polylines(
          annotated,
          [pts_np],
          isClosed=True,
          color=(0, 255, 0),
          thickness=2,
      )

    # 2. Draw Player Labels
    for player in frame_state.players:
      if "box" not in player:
        continue

      box = player["box"]
      x1, y1, x2, y2 = map(int, box)

      # Get track_id or fallback to "#"
      track_id = player.get("track_id")
      label = f"ID {track_id}" if track_id is not None else "#"
      
      # If your engine flagged a jump shot for this player, include it
      if player.get("is_jumping") and player.get("has_ball"):
          label = f"{label}: JUMP SHOT!"

      # Position the text (e.g., just above the bounding box top-left corner)
      text_position = (x1, max(y1 - 10, 20))

      # Draw ONLY the text label using OpenCV
      cv2.putText(
          img=frame,
          text=label,
          org=text_position,
          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
          fontScale=0.6,
          color=(0, 255, 0),  # Green text (or change as you like)
          thickness=2,
          lineType=cv2.LINE_AA
      )

    # 3. Draw Ball
    if getattr(frame_state, "ball", None) is not None:
      try:
        center = frame_state.ball.get("center")
        if center is not None:
          bx, by = map(int, center)
          cv2.circle(
              annotated,
              (bx, by),
              6,
              (0, 255, 255),
              -1,
          )
      except (ValueError, TypeError):
        pass

    frame_state.annotated_frame = annotated

    # 4. Save Event Snapshot
    if (
        self.save_snapshots
        and getattr(frame_state, "is_snapshot_moment", False)
        and hasattr(frame_state, "frame_id")
    ):
      filename = self.snapshot_dir / f"event_{frame_state.frame_id}.jpg"
      cv2.imwrite(str(filename), annotated)

    return frame_state