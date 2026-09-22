from __future__ import annotations

from typing import Iterable

from services.vision.game_player_resolver_v1 import (
    GamePlayerEntity,
    GamePlayerId,
    GamePlayerResolverV1,
    TrackEpisode,
)


class GateDGamePlayerResolverV1(GamePlayerResolverV1):
    """Runtime-safe Gate-D wrapper around the pure V1 resolver.

    The base resolver already enforces overlap using observed episode intervals. During a
    real frame, however, players are observed sequentially. This wrapper also provides the
    complete active-track set for that frame so simultaneity is a hard constraint even
    before every track has extended its episode to the current frame.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gate_d_active_track_ids: set[int] = set()

    def _entity_has_other_active_track(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> bool:
        if not self._gate_d_active_track_ids:
            return False
        for key in entity.episode_keys:
            other = self.episodes.get(key)
            if other is None or other.track_id == episode.track_id:
                continue
            if other.track_id not in self._gate_d_active_track_ids:
                continue
            if self.current_episode_by_track.get(other.track_id) == key:
                return True
        return False

    def _hard_reject_entity(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
        active_track_ids: set[int] | None = None,
    ) -> bool:
        if self._entity_has_other_active_track(entity, episode):
            return True
        return super()._hard_reject_entity(entity, episode)

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
        if episode.game_player_id is not None:
            entity = self.entities.get(episode.game_player_id)
            if entity is not None and self._entity_has_other_active_track(entity, episode):
                return True
        return super()._should_split(
            episode,
            frame_id=frame_id,
            team_id=team_id,
            team_confidence=team_confidence,
            appearance=appearance,
        )

    def observe(
        self,
        *,
        frame_id: int,
        track_id: int,
        appearance: Iterable[float] | None = None,
        team_id: str | None = None,
        team_confidence: float = 0.0,
        jersey_number: str | None = None,
        jersey_confidence: float = 0.0,
        simultaneous_track_ids: Iterable[int] | None = None,
    ) -> GamePlayerId | None:
        previous = self._gate_d_active_track_ids
        self._gate_d_active_track_ids = (
            {int(value) for value in simultaneous_track_ids if int(value) >= 0}
            if simultaneous_track_ids is not None
            else set()
        )
        try:
            return super().observe(
                frame_id=frame_id,
                track_id=track_id,
                appearance=appearance,
                team_id=team_id,
                team_confidence=team_confidence,
                jersey_number=jersey_number,
                jersey_confidence=jersey_confidence,
            )
        finally:
            self._gate_d_active_track_ids = previous

    def resolution(self, track_id: int, frame_id: int) -> dict[str, object] | None:
        track_id = int(track_id)
        frame_id = int(frame_id)
        matches = [
            episode
            for episode in self.episodes.values()
            if episode.track_id == track_id
            and episode.frame_start <= frame_id <= episode.frame_end
        ]
        if len(matches) != 1:
            return None
        episode = matches[0]
        return {
            "track_id": episode.track_id,
            "episode_index": episode.episode_index,
            "frame_start": episode.frame_start,
            "frame_end": episode.frame_end,
            "game_player_id": episode.game_player_id,
            "team_id": episode.team_id,
            "team_confidence": episode.team_confidence,
            "jersey_number": episode.jersey_number,
            "jersey_confidence": episode.jersey_confidence,
            "appearance_count": episode.appearance_count,
        }
