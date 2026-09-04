from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class DiagnosticPreview:
    """Lightweight annotated preview for visual validation during beta runs.

    This renderer is deliberately independent from court/homography analytics. It draws
    only detector outputs already available on the frame state and writes one rolling
    JPEG so the Streamlit UI can refresh it without producing an annotated video.
    """

    _PALETTE = [
        (255, 120, 40),
        (40, 90, 255),
        (80, 210, 120),
        (220, 80, 220),
    ]

    def __init__(
        self,
        output_path: Path,
        match_config_path: Path | None = None,
        interval_frames: int = 15,
        max_width: int = 1100,
        jpeg_quality: int = 78,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_frames = max(1, int(interval_frames))
        self.max_width = max(320, int(max_width))
        self.jpeg_quality = max(40, min(95, int(jpeg_quality)))
        self.team_colors = self._load_team_colors(Path(match_config_path)) if match_config_path else {}

    def _load_team_colors(self, path: Path) -> dict[str, tuple[int, int, int]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            teams = list((data.get("teams", {}) or {}).keys())
        except Exception:
            teams = []
        return {
            team: self._PALETTE[index % len(self._PALETTE)]
            for index, team in enumerate(teams)
        }

    def _color_for_team(self, team: str) -> tuple[int, int, int]:
        if team in self.team_colors:
            return self.team_colors[team]
        if team and team != "Unknown":
            color = self._PALETTE[len(self.team_colors) % len(self._PALETTE)]
            self.team_colors[team] = color
            return color
        return (180, 180, 180)

    def should_render(self, frame_id: int) -> bool:
        return frame_id == 0 or frame_id % self.interval_frames == 0

    @staticmethod
    def _safe_box(box: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def render(self, frame_state: Any) -> Path | None:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        if not self.should_render(frame_id):
            return None

        frame = getattr(frame_state, "frame", None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        canvas = frame.copy()
        height, width = canvas.shape[:2]
        players = list(getattr(frame_state, "players", []) or [])

        for player in players:
            if not isinstance(player, dict):
                continue
            box = self._safe_box(player.get("box"), width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            team = str(player.get("team") or "Unknown")
            jersey = player.get("shirt_num") or player.get("shirt_number")
            track_id = player.get("track_id", -1)
            confidence = player.get("confidence")
            color = self._color_for_team(team)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label_parts = [f"T{track_id}", team]
            if jersey is not None and str(jersey).strip():
                label_parts.append(f"#{jersey}")
            if confidence is not None:
                label_parts.append(f"{float(confidence):.2f}")
            label = " ".join(label_parts)

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            top = max(0, y1 - th - baseline - 8)
            cv2.rectangle(canvas, (x1, top), (min(width - 1, x1 + tw + 8), y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, max(th + 2, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        ball = getattr(frame_state, "ball", None)
        if isinstance(ball, dict):
            box = self._safe_box(ball.get("box"), width, height)
            if box is not None:
                x1, y1, x2, y2 = box
                ball_color = (0, 220, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), ball_color, 2)
                conf = ball.get("confidence")
                label = "BALL" if conf is None else f"BALL {float(conf):.2f}"
                cv2.putText(
                    canvas,
                    label,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    ball_color,
                    2,
                    cv2.LINE_AA,
                )

        timestamp = float(getattr(frame_state, "timestamp", 0.0) or 0.0)
        team_counts: dict[str, int] = {}
        for player in players:
            if isinstance(player, dict) and player.get("team"):
                name = str(player["team"])
                team_counts[name] = team_counts.get(name, 0) + 1
        status = f"frame {frame_id} | {timestamp:.2f}s | players {len(players)}"
        if team_counts:
            status += " | " + " / ".join(f"{name} {count}" for name, count in team_counts.items())
        cv2.rectangle(canvas, (8, 8), (min(width - 8, 18 + len(status) * 8), 38), (20, 20, 20), -1)
        cv2.putText(canvas, status, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)

        if width > self.max_width:
            scale = self.max_width / float(width)
            canvas = cv2.resize(canvas, (self.max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

        temp_path = self.output_path.with_suffix(".tmp.jpg")
        ok = cv2.imwrite(
            str(temp_path),
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return None
        temp_path.replace(self.output_path)
        return self.output_path

    def draw(self, frame_state: Any):
        path = self.render(frame_state)
        if path is not None:
            frame_state.diagnostics.setdefault("preview", {})
            frame_state.diagnostics["preview"].update(
                {
                    "path": str(path),
                    "frame_id": int(getattr(frame_state, "frame_id", 0)),
                }
            )
        return frame_state
