from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from services.vision.game_player_resolver_gate_d_v4 import GateDGamePlayerResolverV4
from services.vision.game_player_resolver_v1 import (
    AssociationDecision,
    EpisodeKey,
    GamePlayerId,
    TrackEpisode,
)


@dataclass(frozen=True, slots=True)
class GateDDeferredAssociationConfig:
    """Conservative late-resolution policy for persistent ambiguous episodes.

    This does not lower the normal resolver thresholds. It only gives an episode that
    remains UNKNOWN because the best two candidates are too close a second chance after
    the episode has accumulated stable appearance evidence for the same best candidate.
    """

    min_episode_observations: int = 12
    min_consistent_observations: int = 8
    min_best_similarity: float = 0.94
    min_margin: float = 0.025


class GateDGamePlayerResolverV5(GateDGamePlayerResolverV4):
    """Gate-D v4 plus persistent-evidence recovery from ambiguous UNKNOWN.

    V3 development evidence showed a long-lived episode that stayed UNKNOWN because its
    best and second-best historical identities were separated by less than the normal
    one-shot association margin. Re-evaluating the same one-shot rule cannot recover such
    an episode, even after its appearance estimate becomes much more stable.

    V5 therefore adds one narrow late-association path:
    * only ``unknown_ambiguous_match`` episodes are eligible;
    * the normal match threshold remains unchanged;
    * the best candidate must be very strong and retain a positive safety margin;
    * the same candidate must win for several consecutive observations;
    * candidates that share this raw track are excluded, so this path cannot undo a
      deliberate within-track identity split;
    * all existing simultaneity, temporal-window, and team hard rejects remain active.
    """

    def __init__(
        self,
        *args,
        deferred_association_config: GateDDeferredAssociationConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.deferred_association_config = (
            deferred_association_config or GateDDeferredAssociationConfig()
        )
        self._gate_d_deferred_candidate: dict[EpisodeKey, GamePlayerId] = {}
        self._gate_d_deferred_streak: dict[EpisodeKey, int] = {}
        self._gate_d_deferred_best_sum: dict[EpisodeKey, float] = {}
        self._gate_d_deferred_margin_sum: dict[EpisodeKey, float] = {}

    def _reset_deferred_candidate(self, episode_key: EpisodeKey) -> None:
        self._gate_d_deferred_candidate.pop(episode_key, None)
        self._gate_d_deferred_streak.pop(episode_key, None)
        self._gate_d_deferred_best_sum.pop(episode_key, None)
        self._gate_d_deferred_margin_sum.pop(episode_key, None)

    def _entity_reuses_track(self, game_player_id: GamePlayerId, track_id: int) -> bool:
        entity = self.entities.get(game_player_id)
        if entity is None:
            return False
        return any(
            self.episodes[key].track_id == int(track_id)
            for key in entity.episode_keys
            if key in self.episodes
        )

    def _rank_deferred_candidates(
        self,
        episode: TrackEpisode,
    ) -> list[tuple[float, GamePlayerId]]:
        scored: list[tuple[float, GamePlayerId]] = []
        for game_player_id, entity in self.entities.items():
            # Deferred recovery is cross-track only. A later episode from the same raw
            # track may represent a real tracker identity switch and must not be glued
            # back to the old identity by this relaxed temporal evidence path.
            if self._entity_reuses_track(game_player_id, episode.track_id):
                continue
            if self._hard_reject_entity(entity, episode):
                continue
            if not self._entity_within_association_window(entity, episode):
                continue
            score = self._association_score(entity, episode)
            if score is not None:
                scored.append((float(score), game_player_id))
        scored.sort(reverse=True)
        return scored

    def _refresh_assignment(self, episode: TrackEpisode) -> None:
        if episode.game_player_id is not None:
            self._reset_deferred_candidate(episode.key)
            return

        # Preserve the canonical resolver path first. V5 is only a fallback for a
        # persistent ambiguity that the normal policy explicitly abstains on.
        super()._refresh_assignment(episode)
        if episode.game_player_id is not None:
            self._reset_deferred_candidate(episode.key)
            return

        decision = self._gate_d_decisions.get(episode.key)
        if decision is None or decision.reason != "unknown_ambiguous_match":
            self._reset_deferred_candidate(episode.key)
            return

        ranked = self._rank_deferred_candidates(episode)
        if len(ranked) < 2:
            self._reset_deferred_candidate(episode.key)
            return

        best_score, best_id = ranked[0]
        second_score = ranked[1][0]
        margin = best_score - second_score
        config = self.deferred_association_config
        if (
            best_score < float(config.min_best_similarity)
            or margin < float(config.min_margin)
        ):
            self._reset_deferred_candidate(episode.key)
            return

        previous = self._gate_d_deferred_candidate.get(episode.key)
        if previous != best_id:
            self._gate_d_deferred_candidate[episode.key] = best_id
            self._gate_d_deferred_streak[episode.key] = 1
            self._gate_d_deferred_best_sum[episode.key] = best_score
            self._gate_d_deferred_margin_sum[episode.key] = margin
            return

        streak = int(self._gate_d_deferred_streak.get(episode.key, 0) or 0) + 1
        self._gate_d_deferred_streak[episode.key] = streak
        self._gate_d_deferred_best_sum[episode.key] = (
            float(self._gate_d_deferred_best_sum.get(episode.key, 0.0) or 0.0)
            + best_score
        )
        self._gate_d_deferred_margin_sum[episode.key] = (
            float(self._gate_d_deferred_margin_sum.get(episode.key, 0.0) or 0.0)
            + margin
        )

        if (
            episode.observation_count < int(config.min_episode_observations)
            or streak < int(config.min_consistent_observations)
        ):
            return

        mean_best = self._gate_d_deferred_best_sum[episode.key] / streak
        mean_margin = self._gate_d_deferred_margin_sum[episode.key] / streak
        if (
            mean_best < float(config.min_best_similarity)
            or mean_margin < float(config.min_margin)
        ):
            return

        # Re-check every hard condition immediately before promotion because another
        # track from the candidate entity may have become active earlier in this frame.
        entity = self.entities.get(best_id)
        if (
            entity is None
            or self._entity_reuses_track(best_id, episode.track_id)
            or self._hard_reject_entity(entity, episode)
            or not self._entity_within_association_window(entity, episode)
        ):
            self._reset_deferred_candidate(episode.key)
            return

        self._gate_d_decisions[episode.key] = AssociationDecision(
            best_id,
            "matched_existing_entity_deferred",
            best_score=best_score,
            second_score=second_score,
            candidate_count=len(ranked),
        )
        self._attach_episode(episode, best_id)
        self._reset_deferred_candidate(episode.key)

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
        track_id = int(track_id)
        previous_episode_key = self.current_episode_by_track.get(track_id)
        result = super().observe(
            frame_id=frame_id,
            track_id=track_id,
            appearance=appearance,
            team_id=team_id,
            team_confidence=team_confidence,
            jersey_number=jersey_number,
            jersey_confidence=jersey_confidence,
            simultaneous_track_ids=simultaneous_track_ids,
        )
        current_episode_key = self.current_episode_by_track.get(track_id)
        if (
            previous_episode_key is not None
            and current_episode_key is not None
            and current_episode_key != previous_episode_key
        ):
            self._reset_deferred_candidate(previous_episode_key)
        return result
