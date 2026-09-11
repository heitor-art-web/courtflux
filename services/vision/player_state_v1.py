from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from core.config import Settings, settings
from core.player_state import PlayerState


@dataclass(slots=True)
class _TrackMemory:
    frame_id: int
    timestamp_sec: float | None
    center_px: tuple[float, float]
    court_position_m: tuple[float, float] | None


class PlayerStateEstimatorV1:
    """Convert upstream tracked person detections into canonical P1 ``PlayerState[]``.

    Tracking identity/association remains owned by the existing player tracking baseline.
    This adapter requires an upstream track id, preserves player/referee separation, and
    adds per-track motion/provenance without introducing P2 event semantics.
    """

    def __init__(self, *, config: Settings | None = None, memory_frames: int = 90) -> None:
        self.config = config or settings
        self.memory_frames = max(1, int(memory_frames))
        self._memory: dict[int, _TrackMemory] = {}
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _class_name(row: dict[str, Any]) -> str:
        return str(row.get("class_name") or row.get("class") or "").strip().lower()

    @staticmethod
    def _track_id(row: dict[str, Any]) -> int | None:
        raw = row.get("track_id", row.get("id"))
        if raw is None:
            return None
        try:
            track_id = int(raw)
        except (TypeError, ValueError):
            return None
        # The established tracking baseline uses -1 for detections that have not been
        # assigned a persistent tracker identity. Such observations are useful diagnostics,
        # but they must never become canonical PlayerState identities.
        return track_id if track_id >= 0 else None

    @staticmethod
    def _bbox(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
        raw = row.get("bbox", row.get("box"))
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            return None
        return tuple(int(round(float(value))) for value in raw[:4])  # type: ignore[return-value]

    @staticmethod
    def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _court_position(row: dict[str, Any]) -> tuple[float, float] | None:
        raw = row.get("court_position_m")
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return None
        return float(raw[0]), float(raw[1])

    @staticmethod
    def _confidence(row: dict[str, Any]) -> float:
        return max(0.0, min(float(row.get("confidence", 0.0)), 1.0))

    def _expire_memory(self, frame_id: int) -> None:
        expired = [
            track_id
            for track_id, memory in self._memory.items()
            if int(frame_id) - memory.frame_id > self.memory_frames
        ]
        for track_id in expired:
            self._memory.pop(track_id, None)

    @staticmethod
    def _velocity(
        previous: _TrackMemory | None,
        *,
        frame_id: int,
        center_px: tuple[float, float],
    ) -> tuple[float, float]:
        if previous is None:
            return 0.0, 0.0
        dt_frames = int(frame_id) - previous.frame_id
        if dt_frames <= 0:
            return 0.0, 0.0
        return (
            (center_px[0] - previous.center_px[0]) / dt_frames,
            (center_px[1] - previous.center_px[1]) / dt_frames,
        )

    @staticmethod
    def _speed_mps(
        previous: _TrackMemory | None,
        *,
        timestamp_sec: float | None,
        court_position_m: tuple[float, float] | None,
    ) -> float | None:
        if (
            previous is None
            or previous.timestamp_sec is None
            or timestamp_sec is None
            or previous.court_position_m is None
            or court_position_m is None
        ):
            return None
        dt = float(timestamp_sec) - previous.timestamp_sec
        if dt <= 0.0:
            return None
        dx = court_position_m[0] - previous.court_position_m[0]
        dy = court_position_m[1] - previous.court_position_m[1]
        return (dx * dx + dy * dy) ** 0.5 / dt

    def update(
        self,
        tracked_people: list[dict[str, Any]],
        *,
        frame_id: int,
        timestamp_sec: float | None,
    ) -> list[PlayerState]:
        self._expire_memory(frame_id)
        states: list[PlayerState] = []
        diagnostics: Counter[str] = Counter()

        for row in list(tracked_people or []):
            class_name = self._class_name(row)
            if class_name == "referee":
                diagnostics["referee_excluded"] += 1
                continue
            if class_name != "player":
                diagnostics["non_player_excluded"] += 1
                continue

            track_id = self._track_id(row)
            bbox = self._bbox(row)
            if track_id is None:
                diagnostics["player_missing_track_id"] += 1
                continue
            if bbox is None:
                diagnostics["player_missing_bbox"] += 1
                continue

            center_px = self._center(bbox)
            court_position_m = self._court_position(row)
            confidence = self._confidence(row)
            previous = self._memory.get(track_id)
            velocity = self._velocity(previous, frame_id=frame_id, center_px=center_px)
            speed_mps = self._speed_mps(
                previous,
                timestamp_sec=timestamp_sec,
                court_position_m=court_position_m,
            )

            state = PlayerState(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                track_id=track_id,
                bbox=bbox,
                center_px=center_px,
                confidence=confidence,
                uncertainty=1.0 - confidence,
                velocity_px_per_frame=velocity,
                age_since_observation=0,
                source=str(row.get("source") or "tracked_player_detection"),
                court_position_m=court_position_m,
                speed_mps=speed_mps,
                provenance={
                    "detector_model": self.config.model,
                    "detector_version": self.config.detector_version,
                    "detector_baseline_status": self.config.detector_baseline_status,
                    "upstream_track_id": track_id,
                    "upstream_class": "player",
                    "player_referee_separation": True,
                    "tracking_association_owned_upstream": True,
                    "identity_resolution": "deferred",
                    "uncertainty_policy": "1_minus_upstream_confidence",
                },
            )
            states.append(state)
            diagnostics["player_states"] += 1
            self._memory[track_id] = _TrackMemory(
                frame_id=int(frame_id),
                timestamp_sec=None if timestamp_sec is None else float(timestamp_sec),
                center_px=center_px,
                court_position_m=court_position_m,
            )

        diagnostics["input_tracks"] = len(tracked_people or [])
        diagnostics["active_memory_tracks"] = len(self._memory)
        self.last_diagnostics = dict(diagnostics)
        return states

    def metadata(self) -> dict[str, Any]:
        return {
            "version": "player_state_v1",
            "detector_model": self.config.model,
            "detector_version": self.config.detector_version,
            "memory_frames": self.memory_frames,
            "requires_upstream_track_id": True,
            "negative_track_ids_are_untracked": True,
            "tracking_association_owned_upstream": True,
            "referees_are_player_states": False,
            "identity_resolution": "deferred_to_identity_consolidation",
            "p2_event_semantics": False,
        }
