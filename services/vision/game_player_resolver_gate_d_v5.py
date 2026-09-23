from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from services.vision.game_player_resolver_gate_d_v4 import GateDGamePlayerResolverV4
from services.vision.game_player_resolver_v1 import (
    AssociationDecision,
    EpisodeKey,
    GamePlayerEntity,
    GamePlayerId,
    TrackEpisode,
)


@dataclass(frozen=True, slots=True)
class GateDDeferredAssociationConfig:
    """Conservative late-resolution policy for persistent ambiguous episodes.

    Normal association thresholds and the normal temporal window are unchanged. The
    deferred path can recover either a persistent within-window ambiguity or a strong
    reappearance just beyond the normal temporal window after several consistent frames.
    """

    min_episode_observations: int = 12
    min_consistent_observations: int = 8
    min_best_similarity: float = 0.94
    min_margin: float = 0.025
    max_reappearance_gap_frames: int = 270


class GateDGamePlayerResolverV5(GateDGamePlayerResolverV4):
    """Gate-D v4 plus conservative persistent-evidence identity recovery.

    V3 development evidence exposed two related UNKNOWN modes:
    * a long-lived episode can stay ambiguous even after its appearance estimate becomes
      stable enough to prefer one historical identity;
    * after a short scene/camera absence, the same player can reappear just beyond the
      normal association window and be treated as having no compatible history at all.

    V5 keeps the normal thresholds and normal association window untouched. It adds only
    deferred recovery paths with stronger temporal evidence:
    * the same candidate must win for several consecutive observations;
    * the best candidate must be very strong and keep a positive safety margin;
    * long-gap reappearance is bounded by a second, modest temporal ceiling;
    * candidates that share the raw track are excluded, so deliberate within-track
      identity splits cannot be glued back together by this path;
    * simultaneity and strong team conflicts remain hard rejects.
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

    def _association_score_without_gap(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> float | None:
        similarity = self._cosine_similarity(entity.appearance, episode.appearance)
        if similarity is None:
            return None
        score = similarity
        if (
            entity.jersey_number
            and episode.jersey_number
            and entity.jersey_confidence >= 0.50
            and episode.jersey_confidence >= 0.50
        ):
            score += (
                self.config.jersey_positive_bonus
                if entity.jersey_number == episode.jersey_number
                else -self.config.jersey_negative_penalty
            )
        return score

    def _entity_within_reappearance_window(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> bool:
        last_frame = self._entity_last_frame(entity)
        if last_frame is None:
            return False
        gap = max(0, int(episode.frame_start) - int(last_frame))
        return bool(
            gap > int(self.config.max_association_gap_frames)
            and gap <= int(self.deferred_association_config.max_reappearance_gap_frames)
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

    def _rank_reappearance_candidates(
        self,
        episode: TrackEpisode,
    ) -> list[tuple[float, GamePlayerId]]:
        scored: list[tuple[float, GamePlayerId]] = []
        for game_player_id, entity in self.entities.items():
            if self._entity_reuses_track(game_player_id, episode.track_id):
                continue
            if self._hard_reject_entity(entity, episode):
                continue
            if not self._entity_within_reappearance_window(entity, episode):
                continue
            score = self._association_score_without_gap(entity, episode)
            if score is not None:
                scored.append((float(score), game_player_id))
        scored.sort(reverse=True)
        return scored

    def _safe_ranked_head(
        self,
        ranked: list[tuple[float, GamePlayerId]],
    ) -> tuple[float, GamePlayerId, float | None, float] | None:
        if not ranked:
            return None
        best_score, best_id = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else None
        margin = (
            best_score - second_score
            if second_score is not None
            else 1.0
        )
        config = self.deferred_association_config
        if (
            best_score < float(config.min_best_similarity)
            or margin < float(config.min_margin)
        ):
            return None
        return best_score, best_id, second_score, margin

    def _choose_entity(self, episode: TrackEpisode) -> AssociationDecision:
        decision = super()._choose_entity(episode)
        if decision.reason != "new_entity_no_compatible_history":
            return decision

        # Do not widen the normal association window globally. If a very strong stale
        # candidate exists just outside it, hold this new episode UNKNOWN long enough to
        # accumulate evidence instead of immediately minting a new identity.
        ranked = self._rank_reappearance_candidates(episode)
        head = self._safe_ranked_head(ranked)
        if head is None:
            return decision
        best_score, _, second_score, _ = head
        held = AssociationDecision(
            None,
            "unknown_stale_reappearance_candidate",
            best_score=best_score,
            second_score=second_score,
            candidate_count=len(ranked),
        )
        self._gate_d_decisions[episode.key] = held
        return held

    def _refresh_assignment(self, episode: TrackEpisode) -> None:
        if episode.game_player_id is not None:
            self._reset_deferred_candidate(episode.key)
            return

        # Preserve the canonical resolver path first. V5 is only a fallback for a
        # persistent ambiguity or a bounded strong reappearance that the normal policy
        # deliberately abstains on.
        super()._refresh_assignment(episode)
        if episode.game_player_id is not None:
            self._reset_deferred_candidate(episode.key)
            return

        decision = self._gate_d_decisions.get(episode.key)
        if decision is None or decision.reason not in {
            "unknown_ambiguous_match",
            "unknown_stale_reappearance_candidate",
        }:
            self._reset_deferred_candidate(episode.key)
            return

        reappearance = decision.reason == "unknown_stale_reappearance_candidate"
        ranked = (
            self._rank_reappearance_candidates(episode)
            if reappearance
            else self._rank_deferred_candidates(episode)
        )
        head = self._safe_ranked_head(ranked)
        if head is None:
            self._reset_deferred_candidate(episode.key)
            return

        best_score, best_id, second_score, margin = head
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

        config = self.deferred_association_config
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
        within_window = bool(
            entity is not None
            and (
                self._entity_within_reappearance_window(entity, episode)
                if reappearance
                else self._entity_within_association_window(entity, episode)
            )
        )
        if (
            entity is None
            or self._entity_reuses_track(best_id, episode.track_id)
            or self._hard_reject_entity(entity, episode)
            or not within_window
        ):
            self._reset_deferred_candidate(episode.key)
            return

        reason = (
            "matched_existing_entity_deferred_reappearance"
            if reappearance
            else "matched_existing_entity_deferred"
        )
        self._gate_d_decisions[episode.key] = AssociationDecision(
            best_id,
            reason,
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
