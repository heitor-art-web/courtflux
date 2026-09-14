from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class DiagnosticPreview:
    """Lightweight annotated preview shared by local and web Beta Lab runs."""

    _PALETTE = [(255, 120, 40), (40, 90, 255), (80, 210, 120), (220, 80, 220)]
    _REFEREE_COLOR = (0, 215, 255)
    _OFF_COURT_COLOR = (120, 120, 120)
    _COURT_COLOR = (70, 220, 70)

    def __init__(
        self,
        output_path: Path,
        match_config_path: Path | None = None,
        interval_frames: int = 15,
        max_width: int = 1100,
        jpeg_quality: int = 78,
        video_output_path: Path | None = None,
        video_fps: float = 4.0,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_frames = max(1, int(interval_frames))
        self.max_width = max(320, int(max_width))
        self.jpeg_quality = max(40, min(95, int(jpeg_quality)))
        self.team_colors = self._load_team_colors(Path(match_config_path)) if match_config_path else {}
        self.video_output_path = Path(video_output_path) if video_output_path else None
        self.video_fps = max(0.1, float(video_fps))
        self._video_writer: cv2.VideoWriter | None = None
        self._video_size: tuple[int, int] | None = None
        self._last_video_timestamp: float | None = None
        self._closed = False
        self._working_video_path = (
            self.video_output_path.with_name(f"{self.video_output_path.stem}.working{self.video_output_path.suffix}")
            if self.video_output_path
            else None
        )
        if self.video_output_path is not None:
            self.video_output_path.parent.mkdir(parents=True, exist_ok=True)
            self.video_output_path.unlink(missing_ok=True)
        if self._working_video_path is not None:
            self._working_video_path.unlink(missing_ok=True)

    def _load_team_colors(self, path: Path) -> dict[str, tuple[int, int, int]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            teams = list((data.get("teams", {}) or {}).keys())
        except Exception:
            teams = []
        return {team: self._PALETTE[index % len(self._PALETTE)] for index, team in enumerate(teams)}

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
        x1 = max(0, min(width - 1, x1)); x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1)); y2 = max(0, min(height - 1, y2))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    @staticmethod
    def _court_payload(frame_state: Any) -> dict[str, Any]:
        state = getattr(frame_state, "court_state", None)
        if state is None:
            return dict((getattr(frame_state, "diagnostics", {}) or {}).get("court_state", {}) or {})
        if hasattr(state, "as_dict"):
            return dict(state.as_dict())
        if isinstance(state, dict):
            return dict(state)
        return {}

    @staticmethod
    def _ball_state_payload(frame_state: Any) -> dict[str, Any]:
        state = getattr(frame_state, "ball_state", None)
        if state is not None:
            if hasattr(state, "as_dict"):
                return dict(state.as_dict())
            if isinstance(state, dict):
                return dict(state)
        ball = getattr(frame_state, "ball", None)
        if not isinstance(ball, dict):
            return {"status": "missing", "box": None, "confidence": 0.0, "source": "legacy_missing"}
        return {
            "status": "predicted" if bool(ball.get("interpolated")) else "observed",
            "box": ball.get("box"),
            "confidence": ball.get("confidence"),
            "source": ball.get("source"),
        }

    @staticmethod
    def _even_video_frame(canvas: np.ndarray) -> np.ndarray:
        """Keep dimensions compatible with 4:2:0 browser video decoders."""
        height, width = canvas.shape[:2]
        even_width = width - (width % 2)
        even_height = height - (height % 2)
        if even_width < 2 or even_height < 2:
            return canvas
        if (even_width, even_height) == (width, height):
            return canvas
        return cv2.resize(canvas, (even_width, even_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _ffmpeg_executable() -> str | None:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    def _write_video_frame(self, canvas: np.ndarray, timestamp: float) -> None:
        if self._working_video_path is None or self._closed:
            return
        canvas = self._even_video_frame(canvas)
        height, width = canvas.shape[:2]
        size = (int(width), int(height))
        if self._video_writer is None:
            writer = cv2.VideoWriter(
                str(self._working_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                size,
            )
            if not writer.isOpened():
                print("⚠️ Diagnostic preview video writer could not be opened.", flush=True)
                return
            self._video_writer = writer
            self._video_size = size
        if self._video_size != size:
            canvas = cv2.resize(canvas, self._video_size, interpolation=cv2.INTER_AREA)

        repeat = 1
        if self._last_video_timestamp is not None:
            delta = max(0.0, float(timestamp) - self._last_video_timestamp)
            repeat = max(1, int(round(delta * self.video_fps)))
        for _ in range(repeat):
            self._video_writer.write(canvas)
        self._last_video_timestamp = float(timestamp)

    def _finalize_video(self) -> None:
        if self.video_output_path is None or self._working_video_path is None:
            return
        if not self._working_video_path.exists() or self._working_video_path.stat().st_size == 0:
            return

        ffmpeg = self._ffmpeg_executable()
        if ffmpeg:
            finalizing_path = self.video_output_path.with_name(
                f"{self.video_output_path.stem}.finalizing{self.video_output_path.suffix}"
            )
            finalizing_path.unlink(missing_ok=True)
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel", "error",
                    "-i", str(self._working_video_path),
                    "-an",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(finalizing_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and finalizing_path.exists() and finalizing_path.stat().st_size > 0:
                finalizing_path.replace(self.video_output_path)
                self._working_video_path.unlink(missing_ok=True)
                return
            finalizing_path.unlink(missing_ok=True)
            detail = (result.stderr or "ffmpeg returned no error output").strip()
            print(f"⚠️ H.264 preview finalization failed; keeping MPEG-4 fallback: {detail}", flush=True)
        else:
            print("⚠️ ffmpeg is unavailable; keeping MPEG-4 fallback preview.", flush=True)

        self._working_video_path.replace(self.video_output_path)

    def render(self, frame_state: Any) -> Path | None:
        frame_id = int(getattr(frame_state, "frame_id", 0))
        if not self.should_render(frame_id):
            return None
        frame = getattr(frame_state, "frame", None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        canvas = frame.copy()
        height, width = canvas.shape[:2]
        court = self._court_payload(frame_state)
        polygon = court.get("court_polygon_px")
        if isinstance(polygon, list) and len(polygon) >= 3:
            try:
                points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [points], True, self._COURT_COLOR, 2, cv2.LINE_AA)
            except (TypeError, ValueError, cv2.error):
                pass

        players = list(getattr(frame_state, "players", []) or [])
        for player in players:
            if not isinstance(player, dict):
                continue
            box = self._safe_box(player.get("box"), width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            active_on_court = player.get("active_on_court")
            off_court = active_on_court is False
            team = str(player.get("team") or "Unknown")
            jersey = player.get("shirt_num") or player.get("shirt_number")
            track_id = player.get("track_id", -1)
            detection_conf = player.get("confidence")
            team_conf = player.get("team_confidence")
            player_name = str(player.get("player_name") or "").strip()
            color = self._OFF_COURT_COLOR if off_court else self._color_for_team(team)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            parts = [f"P T{track_id}"]
            if off_court:
                parts.append("OFFCOURT")
            else:
                parts.append(team)
                if team_conf is not None: parts.append(f"team:{float(team_conf):.2f}")
                if jersey is not None and str(jersey).strip(): parts.append(f"#{jersey}")
                if player_name: parts.append(player_name)
            if active_on_court is True:
                parts.append("COURT")
            elif active_on_court is None:
                parts.append("COURT?")
            if detection_conf is not None: parts.append(f"det:{float(detection_conf):.2f}")
            label = " ".join(parts)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            top = max(0, y1 - th - baseline - 8)
            cv2.rectangle(canvas, (x1, top), (min(width - 1, x1 + tw + 8), y1), color, -1)
            cv2.putText(canvas, label, (x1 + 4, max(th + 2, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        referees = list(getattr(frame_state, "referees", []) or [])
        for referee in referees:
            if not isinstance(referee, dict):
                continue
            box = self._safe_box(referee.get("box"), width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            track_id = referee.get("track_id", -1)
            conf = referee.get("confidence")
            off_court = referee.get("active_on_court") is False
            color = self._OFF_COURT_COLOR if off_court else self._REFEREE_COLOR
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"REF T{track_id}" + (" OFFCOURT" if off_court else "")
            if referee.get("active_on_court") is True:
                label += " COURT"
            elif referee.get("active_on_court") is None:
                label += " COURT?"
            if conf is not None:
                label += f" det:{float(conf):.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            top = max(0, y1 - th - baseline - 8)
            cv2.rectangle(canvas, (x1, top), (min(width - 1, x1 + tw + 8), y1), color, -1)
            cv2.putText(canvas, label, (x1 + 4, max(th + 2, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)

        ball_state = self._ball_state_payload(frame_state)
        ball_status = str(ball_state.get("status") or "missing").lower()
        ball_box = self._safe_box(ball_state.get("box"), width, height)
        if ball_status != "missing" and ball_box is not None:
            x1, y1, x2, y2 = ball_box
            color = (0, 220, 255)
            predicted = ball_status == "predicted"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1 if predicted else 2)
            conf = ball_state.get("confidence")
            prefix = "BALL PREDICTED" if predicted else "BALL OBSERVED"
            label = prefix if conf is None else f"{prefix} {float(conf):.2f}"
            cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        timestamp = float(getattr(frame_state, "timestamp", 0.0) or 0.0)
        team_counts: dict[str, int] = {}; identified = 0
        for player in players:
            if isinstance(player, dict) and player.get("active_on_court") is not False and player.get("team"):
                team = str(player["team"]); team_counts[team] = team_counts.get(team, 0) + 1
            if isinstance(player, dict) and player.get("active_on_court") is not False and (player.get("shirt_num") or player.get("shirt_number")):
                identified += 1
        player_state_count = len(getattr(frame_state, "player_states", []) or [])
        status = (
            f"frame {frame_id} | {timestamp:.2f}s | det P{len(players)}/R{len(referees)} "
            f"| PlayerState {player_state_count} | BallState {ball_status.upper()} | identified {identified}"
        )
        if team_counts: status += " | " + " / ".join(f"{name} {count}" for name, count in team_counts.items())
        cv2.rectangle(canvas, (8, 8), (min(width - 8, 18 + len(status) * 8), 38), (20, 20, 20), -1)
        cv2.putText(canvas, status, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)

        if court:
            mode = str(court.get("mode") or "unknown")
            eligible = bool(court.get("analysis_eligible", False))
            reason = str(court.get("eligibility_reason") or "unknown")
            on_players = int(court.get("on_court_player_count", 0) or 0)
            on_refs = int(court.get("on_court_referee_count", 0) or 0)
            ratio = court.get("court_surface_ratio")
            basket_count = len(court.get("basket_evidence", []) or [])
            court_line = (
                f"COURT {mode} | metrics:{'YES' if eligible else 'NO'} | "
                f"on-court P{on_players}/R{on_refs} | basket-evidence:{basket_count} | {reason}"
            )
            if ratio is not None:
                court_line += f" | surface:{float(ratio):.2f}"
            cv2.rectangle(canvas, (8, 42), (min(width - 8, 18 + len(court_line) * 8), 72), (20, 20, 20), -1)
            cv2.putText(
                canvas,
                court_line,
                (14, 63),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

        if width > self.max_width:
            scale = self.max_width / float(width)
            canvas = cv2.resize(canvas, (self.max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

        temp_path = self.output_path.with_suffix(".tmp.jpg")
        ok = cv2.imwrite(str(temp_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        temp_path.replace(self.output_path)
        self._write_video_frame(canvas, timestamp)
        return self.output_path

    def draw(self, frame_state: Any):
        path = self.render(frame_state)
        if path is not None:
            frame_state.diagnostics.setdefault("preview", {}).update({"path": str(path), "frame_id": int(getattr(frame_state, "frame_id", 0))})
        return frame_state

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._finalize_video()
