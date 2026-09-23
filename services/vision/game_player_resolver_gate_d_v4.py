from __future__ import annotations

from typing import Iterable

from services.vision.game_player_resolver_gate_d_v3 import GateDGamePlayerResolverV3


class GateDGamePlayerResolverV4(GateDGamePlayerResolverV3):
    """Gate-D v3 plus short temporal-cohort visual-shift suppression.

    Real V4 showed that camera/pose-regime appearance shocks are not perfectly
    frame-synchronous: different established tracks cross the split boundary over a
    short interval. This layer keeps the v3 per-frame classifier intact and adds a
    rolling conflict cohort. Identity association and split thresholds are unchanged.
    """

    temporal_cohort_window_frames = 30
    temporal_cohort_min_tracks = 3
    temporal_cohort_cooldown_frames = 30

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gate_d_recent_conflict_frame_by_track: dict[int, int] = {}
        self._gate_d_temporal_shift_until_frame = -1

    def prepare_frame(
        self,
        *,
        frame_id: int,
        observations: Iterable[tuple[int, Iterable[float] | None]],
    ) -> dict[str, object]:
        frame_id = int(frame_id)
        context = super().prepare_frame(frame_id=frame_id, observations=observations)

        current_conflicts = {
            int(track_id)
            for track_id in list(context.get("conflict_track_ids", []) or [])
            if int(track_id) >= 0
        }
        for track_id in current_conflicts:
            self._gate_d_recent_conflict_frame_by_track[track_id] = frame_id

        cutoff = frame_id - int(self.temporal_cohort_window_frames)
        self._gate_d_recent_conflict_frame_by_track = {
            track_id: seen_frame
            for track_id, seen_frame in self._gate_d_recent_conflict_frame_by_track.items()
            if seen_frame >= cutoff
        }
        cohort_track_ids = sorted(self._gate_d_recent_conflict_frame_by_track)

        temporal_trigger = bool(
            current_conflicts
            and len(cohort_track_ids) >= int(self.temporal_cohort_min_tracks)
        )
        if temporal_trigger:
            self._gate_d_temporal_shift_until_frame = max(
                self._gate_d_temporal_shift_until_frame,
                frame_id + int(self.temporal_cohort_cooldown_frames),
            )

        temporal_active = bool(
            frame_id <= self._gate_d_temporal_shift_until_frame
            and self._gate_d_temporal_shift_until_frame >= 0
        )
        global_shift = bool(context.get("global_visual_shift", False) or temporal_active)

        if global_shift:
            comparable_ids = {
                int(track_id)
                for track_id, _ in observations
                if int(track_id) >= 0
            }
            self._gate_d_global_suppressed_track_ids = comparable_ids

        context.update(
            {
                "global_visual_shift": global_shift,
                "temporal_cohort_active": temporal_active,
                "temporal_cohort_triggered": temporal_trigger,
                "temporal_cohort_track_ids": cohort_track_ids,
                "temporal_cohort_tracks": len(cohort_track_ids),
                "temporal_cohort_until_frame": self._gate_d_temporal_shift_until_frame,
            }
        )
        self._gate_d_frame_context = dict(context)
        return dict(context)
