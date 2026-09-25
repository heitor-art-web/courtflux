from __future__ import annotations

from services.vision.game_player_resolver_gate_d_v5_team_stable import (
    GateDTeamStableGamePlayerResolverV5,
)
from services.vision.game_player_resolver_v1 import GamePlayerEntity, TrackEpisode


class GateDStoppageAwareGamePlayerResolverV5(GateDTeamStableGamePlayerResolverV5):
    """Team-stable Gate-D V5 with NON_GAMEPLAY time removed from continuity gaps.

    Freezing observations during a stoppage is not sufficient on its own: the next real
    gameplay frame may be hundreds of source frames after the last observation, which
    would otherwise trigger ``track_gap`` and make every pre-stoppage entity stale. This
    wrapper records global NON_GAMEPLAY frame ranges and subtracts only those frozen frames
    from within-track and cross-track temporal-gap checks.

    Episode frame ranges remain observation-truth: NON_GAMEPLAY frames are never inserted
    into an episode and ``resolve(track, frame)`` still abstains inside a stoppage. Only the
    temporal distance used by continuity/association is adjusted.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gate_d_frozen_ranges: list[tuple[int, int]] = []
        self._gate_d_active_frozen_range: tuple[int, int] | None = None

    def freeze_non_gameplay(self, frame_id: int) -> None:
        """Record one frozen NON_GAMEPLAY frame without touching identity state."""
        frame_id = int(frame_id)
        active = self._gate_d_active_frozen_range
        if active is None:
            self._gate_d_active_frozen_range = (frame_id, frame_id)
            return
        start, end = active
        if frame_id <= end:
            return
        if frame_id == end + 1:
            self._gate_d_active_frozen_range = (start, frame_id)
            return
        self._gate_d_frozen_ranges.append((start, end))
        self._gate_d_active_frozen_range = (frame_id, frame_id)

    def resume_gameplay(self, frame_id: int) -> None:
        """Close an active frozen range before the next observed gameplay frame."""
        frame_id = int(frame_id)
        active = self._gate_d_active_frozen_range
        if active is None:
            return
        start, end = active
        if frame_id > end:
            self._gate_d_frozen_ranges.append((start, end))
            self._gate_d_active_frozen_range = None

    def _frozen_overlap_frames(self, left_frame: int, right_frame: int) -> int:
        """Count frozen frames strictly between two observed frame IDs."""
        left = int(left_frame)
        right = int(right_frame)
        if right <= left + 1:
            return 0
        lo = left + 1
        hi = right - 1
        ranges = list(self._gate_d_frozen_ranges)
        if self._gate_d_active_frozen_range is not None:
            ranges.append(self._gate_d_active_frozen_range)
        total = 0
        for start, end in ranges:
            overlap_start = max(lo, int(start))
            overlap_end = min(hi, int(end))
            if overlap_end >= overlap_start:
                total += overlap_end - overlap_start + 1
        return total

    def _effective_gap(self, left_frame: int, right_frame: int) -> int:
        raw_gap = max(0, int(right_frame) - int(left_frame))
        return max(
            0,
            raw_gap - self._frozen_overlap_frames(left_frame, right_frame),
        )

    def _association_score(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> float | None:
        last_frame = self._entity_last_frame(entity)
        if last_frame is None:
            return None
        if (
            self._effective_gap(last_frame, episode.frame_start)
            > int(self.config.max_association_gap_frames)
        ):
            return None
        return self._association_score_without_gap(entity, episode)

    def _entity_within_association_window(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> bool:
        last_frame = self._entity_last_frame(entity)
        if last_frame is None:
            return False
        return bool(
            self._effective_gap(last_frame, episode.frame_start)
            <= int(self.config.max_association_gap_frames)
        )

    def _entity_within_reappearance_window(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> bool:
        last_frame = self._entity_last_frame(entity)
        if last_frame is None:
            return False
        gap = self._effective_gap(last_frame, episode.frame_start)
        return bool(
            gap > int(self.config.max_association_gap_frames)
            and gap <= int(self.deferred_association_config.max_reappearance_gap_frames)
        )

    def _should_split(
        self,
        episode: TrackEpisode,
        *,
        frame_id: int,
        team_id: str | None,
        team_confidence: float,
        appearance: tuple[float, ...] | None,
        active_track_ids: set[int] | None = None,
    ) -> bool:
        frozen_overlap = self._frozen_overlap_frames(episode.frame_end, frame_id)
        if frozen_overlap <= 0:
            return super()._should_split(
                episode,
                frame_id=frame_id,
                team_id=team_id,
                team_confidence=team_confidence,
                appearance=appearance,
                active_track_ids=active_track_ids,
            )

        # Parent Gate-D layers use frame_id for current-frame cohort/global-shift logic,
        # so keep the real frame ID and adjust only the stored endpoint while split
        # evaluation runs. The base observe() then writes the real frame_end afterwards.
        original_end = int(episode.frame_end)
        episode.frame_end = min(int(frame_id), original_end + int(frozen_overlap))
        try:
            return super()._should_split(
                episode,
                frame_id=frame_id,
                team_id=team_id,
                team_confidence=team_confidence,
                appearance=appearance,
                active_track_ids=active_track_ids,
            )
        finally:
            episode.frame_end = original_end

    def stoppage_diagnostics(self) -> dict[str, object]:
        active = self._gate_d_active_frozen_range
        return {
            "stoppage_gap_suspension": True,
            "completed_frozen_ranges": len(self._gate_d_frozen_ranges),
            "active_frozen_range": None if active is None else list(active),
            "completed_frozen_frames": sum(
                int(end) - int(start) + 1
                for start, end in self._gate_d_frozen_ranges
            ),
        }
