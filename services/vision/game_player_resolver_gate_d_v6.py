from __future__ import annotations

from typing import Iterable

from services.vision.game_player_resolver_gate_d_v5 import (
    GateDDeferredAssociationConfig,
)
from services.vision.game_player_resolver_gate_d_v5_stoppage import (
    GateDStoppageAwareGamePlayerResolverV5,
)
from services.vision.game_player_resolver_v1 import (
    GamePlayerEntity,
    TrackEpisode,
)


class GateDGamePlayerResolverV6(GateDStoppageAwareGamePlayerResolverV5):
    """V6 anonymous game-local identity recovery.

    Gate D answers only whether observations belong to the same anonymous player inside
    one game. Jersey number, OCR output, roster context and jersey-trained features belong
    to P5 and are deliberately ignored here, even when a caller supplies them.

    V6 retains the validated stoppage/simultaneity safeguards and the useful suppression
    of team-only splits: a confirmed team reversal may open a new episode only when the
    existing non-semantic appearance descriptor also corroborates a visual change.
    """

    TEAM_SPLIT_RAW_SIMILARITY_MAX = 0.76
    TEAM_SPLIT_COARSE_SIMILARITY_MAX = 0.84

    def __init__(
        self,
        *args,
        deferred_association_config: GateDDeferredAssociationConfig | None = None,
        **kwargs,
    ) -> None:
        if deferred_association_config is None:
            # V6 previously widened this window to 12000 frames only because a stable
            # jersey reading was required to support the long reappearance. With jersey
            # outside Gate D, keep the conservative V5 runtime ceiling. This makes the
            # local resolver explicitly abstain rather than guess across long absences;
            # long-range continuity is the job of the anonymous global-association work.
            deferred_association_config = GateDDeferredAssociationConfig(
                min_episode_observations=12,
                min_consistent_observations=8,
                min_best_similarity=0.94,
                min_margin=0.025,
                max_reappearance_gap_frames=180,
                max_reappearance_probe_observations=24,
            )
        super().__init__(
            *args,
            deferred_association_config=deferred_association_config,
            **kwargs,
        )

    def _association_score_without_gap(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> float | None:
        """Anonymous appearance score; jersey metadata is never consulted."""
        return self._cosine_similarity(entity.appearance, episode.appearance)

    def _association_score(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> float | None:
        """Anonymous appearance score with the normal effective temporal window."""
        score = self._association_score_without_gap(entity, episode)
        if score is None:
            return None
        last_frame = self._entity_last_frame(entity)
        if last_frame is not None:
            gap = self._effective_gap(last_frame, episode.frame_start)
            if gap > int(self.config.max_association_gap_frames):
                return None
        return score

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
        split = super()._should_split(
            episode,
            frame_id=frame_id,
            team_id=team_id,
            team_confidence=team_confidence,
            appearance=appearance,
            active_track_ids=active_track_ids,
        )
        if not split:
            return False

        pending = self._gate_d_pending_episode_open.get(int(episode.track_id), {})
        if str(pending.get("reason") or "") != "team_conflict":
            return True

        raw_similarity = self._cosine_similarity(episode.appearance, appearance)
        split_similarity = self._cosine_similarity(
            self._split_prototype(episode),
            self._split_descriptor(appearance),
        )
        appearance_corroborates = bool(
            raw_similarity is not None
            and split_similarity is not None
            and raw_similarity < float(self.TEAM_SPLIT_RAW_SIMILARITY_MAX)
            and split_similarity < float(self.TEAM_SPLIT_COARSE_SIMILARITY_MAX)
        )
        if appearance_corroborates:
            return True

        self._gate_d_pending_episode_open.pop(int(episode.track_id), None)
        episode.team_conflict_streak = 0
        return False

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
    ):
        # Boundary guardrail: keep the public resolver signature compatible with the older
        # detector contract, but intentionally discard jersey evidence before it can enter
        # TrackEpisode/GamePlayerEntity state or any association score.
        _ = jersey_number, jersey_confidence
        return super().observe(
            frame_id=frame_id,
            track_id=int(track_id),
            appearance=appearance,
            team_id=team_id,
            team_confidence=team_confidence,
            jersey_number=None,
            jersey_confidence=0.0,
            simultaneous_track_ids=simultaneous_track_ids,
        )


__all__ = ["GateDGamePlayerResolverV6"]
