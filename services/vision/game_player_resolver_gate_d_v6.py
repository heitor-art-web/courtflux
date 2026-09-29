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
    GamePlayerId,
    TrackEpisode,
)


class GateDGamePlayerResolverV6(GateDStoppageAwareGamePlayerResolverV5):
    """V6 identity recovery with conservative rosterless-jersey support.

    This layer is intentionally isolated to Gate-D V6. It keeps the V5 overlap,
    simultaneity, stoppage and ambiguity safeguards, while addressing three failure modes
    exposed by the frozen V6 human ball-carrier ground truth:

    * confirmed-but-spurious team reversals must not split an otherwise visually coherent
      raw track by themselves;
    * a sustained high-confidence visual jersey number may split a raw track when the
      tracker silently transfers to another same-team player;
    * the same soft jersey evidence may support bounded long-term reappearance, but only
      after the existing V5 persistent-candidate confirmation and never as a hard key.

    Jersey evidence remains soft. It changes scores and can corroborate a split, but
    simultaneity and temporal-overlap constraints remain hard rejects.
    """

    SOFT_JERSEY_MIN_CONFIDENCE = 0.70
    SOFT_JERSEY_SPLIT_OBSERVATIONS = 2
    SOFT_JERSEY_MATCH_BONUS = 0.14
    SOFT_JERSEY_MISMATCH_PENALTY = 0.18

    APPEARANCE_ONLY_REAPPEARANCE_MAX_GAP_FRAMES = 600

    TEAM_SPLIT_RAW_SIMILARITY_MAX = 0.76
    TEAM_SPLIT_COARSE_SIMILARITY_MAX = 0.84

    def __init__(
        self,
        *args,
        deferred_association_config: GateDDeferredAssociationConfig | None = None,
        **kwargs,
    ) -> None:
        if deferred_association_config is None:
            deferred_association_config = GateDDeferredAssociationConfig(
                min_episode_observations=12,
                min_consistent_observations=8,
                min_best_similarity=0.94,
                min_margin=0.025,
                max_reappearance_gap_frames=12000,
                max_reappearance_probe_observations=24,
            )
        super().__init__(
            *args,
            deferred_association_config=deferred_association_config,
            **kwargs,
        )
        self._v6_jersey_conflict_candidate: dict[tuple[int, int], str] = {}
        self._v6_jersey_conflict_streak: dict[tuple[int, int], int] = {}
        self._v6_force_jersey_split_tracks: set[int] = set()

    @classmethod
    def _strong_episode_jersey(cls, episode: TrackEpisode) -> str | None:
        if (
            episode.jersey_number is None
            or float(episode.jersey_confidence) < cls.SOFT_JERSEY_MIN_CONFIDENCE
        ):
            return None
        return str(episode.jersey_number)

    @classmethod
    def _strong_entity_jersey(cls, entity: GamePlayerEntity) -> str | None:
        if (
            entity.jersey_number is None
            or float(entity.jersey_confidence) < cls.SOFT_JERSEY_MIN_CONFIDENCE
        ):
            return None
        return str(entity.jersey_number)

    @classmethod
    def _strong_jersey_relation(
        cls,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> str | None:
        entity_number = cls._strong_entity_jersey(entity)
        episode_number = cls._strong_episode_jersey(episode)
        if entity_number is None or episode_number is None:
            return None
        return "same" if entity_number == episode_number else "different"

    def _association_score_without_gap(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> float | None:
        score = super()._association_score_without_gap(entity, episode)
        if score is None:
            return None
        relation = self._strong_jersey_relation(entity, episode)
        if relation == "same":
            score += float(self.SOFT_JERSEY_MATCH_BONUS)
        elif relation == "different":
            score -= float(self.SOFT_JERSEY_MISMATCH_PENALTY)
        return score

    def _v6_reappearance_hard_reject(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> bool:
        if self._entity_has_other_active_track(entity, episode):
            return True
        if self._entity_is_simultaneous(entity, episode):
            return True

        if self._team_incompatible(
            entity.team_id,
            entity.team_confidence,
            episode.team_id,
            episode.team_confidence,
        ):
            return self._strong_jersey_relation(entity, episode) != "same"
        return False

    def _rank_reappearance_candidates(
        self,
        episode: TrackEpisode,
    ) -> list[tuple[float, GamePlayerId]]:
        scored: list[tuple[float, GamePlayerId]] = []
        for game_player_id, entity in self.entities.items():
            if self._entity_reuses_track(game_player_id, episode.track_id):
                continue
            if self._v6_reappearance_hard_reject(entity, episode):
                continue
            if not self._entity_within_reappearance_window(entity, episode):
                continue

            last_frame = self._entity_last_frame(entity)
            if last_frame is None:
                continue
            gap = self._effective_gap(last_frame, episode.frame_start)
            relation = self._strong_jersey_relation(entity, episode)
            if (
                gap > int(self.APPEARANCE_ONLY_REAPPEARANCE_MAX_GAP_FRAMES)
                and relation != "same"
            ):
                continue

            score = self._association_score_without_gap(entity, episode)
            if score is not None:
                scored.append((float(score), game_player_id))
        scored.sort(reverse=True)
        return scored

    def _rank_deferred_candidates(
        self,
        episode: TrackEpisode,
    ) -> list[tuple[float, GamePlayerId]]:
        scored: list[tuple[float, GamePlayerId]] = []
        for game_player_id, entity in self.entities.items():
            if self._entity_reuses_track(game_player_id, episode.track_id):
                continue
            if self._v6_reappearance_hard_reject(entity, episode):
                continue
            if not self._entity_within_association_window(entity, episode):
                continue
            score = self._association_score(entity, episode)
            if score is not None:
                scored.append((float(score), game_player_id))
        scored.sort(reverse=True)
        return scored

    def _choose_entity(self, episode: TrackEpisode):
        decision = super()._choose_entity(episode)
        if decision.game_player_id is not None:
            return decision
        if str(decision.reason).startswith("unknown_stale_reappearance"):
            return decision

        if (
            decision.reason == "new_entity_strong_visual_separation"
            and self._strong_episode_jersey(episode) is not None
        ):
            ranked = self._rank_reappearance_candidates(episode)
            held = self._hold_reappearance_probe(episode, ranked)
            if held is not None:
                return held
        return decision

    def _reset_v6_jersey_conflict(self, episode: TrackEpisode) -> None:
        self._v6_jersey_conflict_candidate.pop(episode.key, None)
        self._v6_jersey_conflict_streak.pop(episode.key, None)

    def _observe_v6_jersey_conflict(
        self,
        episode: TrackEpisode,
        *,
        jersey_number: str | None,
        jersey_confidence: float,
    ) -> bool:
        established = self._strong_episode_jersey(episode)
        incoming = (
            str(jersey_number)
            if jersey_number is not None
            and float(jersey_confidence) >= self.SOFT_JERSEY_MIN_CONFIDENCE
            else None
        )
        if established is None or incoming is None or incoming == established:
            self._reset_v6_jersey_conflict(episode)
            return False

        previous = self._v6_jersey_conflict_candidate.get(episode.key)
        if previous == incoming:
            streak = int(self._v6_jersey_conflict_streak.get(episode.key, 0)) + 1
        else:
            self._v6_jersey_conflict_candidate[episode.key] = incoming
            streak = 1
        self._v6_jersey_conflict_streak[episode.key] = streak
        return streak >= int(self.SOFT_JERSEY_SPLIT_OBSERVATIONS)

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
        if int(episode.track_id) in self._v6_force_jersey_split_tracks:
            raw_similarity = self._cosine_similarity(episode.appearance, appearance)
            split_similarity = self._cosine_similarity(
                self._split_prototype(episode),
                self._split_descriptor(appearance),
            )
            self._remember_episode_open(
                episode,
                reason="jersey_conflict_soft_v6",
                raw_similarity=raw_similarity,
                split_similarity=split_similarity,
            )
            return True

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
        track_id = int(track_id)
        episode = self._episode_for_track(track_id)
        parent_jersey_number = jersey_number
        parent_jersey_confidence = jersey_confidence

        if episode is not None:
            force_split = self._observe_v6_jersey_conflict(
                episode,
                jersey_number=jersey_number,
                jersey_confidence=jersey_confidence,
            )
            if force_split:
                self._v6_force_jersey_split_tracks.add(track_id)
            elif (
                self._strong_episode_jersey(episode) is not None
                and jersey_number is not None
                and str(jersey_number) != str(episode.jersey_number)
                and float(jersey_confidence) >= self.SOFT_JERSEY_MIN_CONFIDENCE
            ):
                parent_jersey_number = episode.jersey_number
                parent_jersey_confidence = episode.jersey_confidence

        previous_episode_key = self.current_episode_by_track.get(track_id)
        try:
            result = super().observe(
                frame_id=frame_id,
                track_id=track_id,
                appearance=appearance,
                team_id=team_id,
                team_confidence=team_confidence,
                jersey_number=parent_jersey_number,
                jersey_confidence=parent_jersey_confidence,
                simultaneous_track_ids=simultaneous_track_ids,
            )
        finally:
            self._v6_force_jersey_split_tracks.discard(track_id)

        current_episode_key = self.current_episode_by_track.get(track_id)
        if (
            previous_episode_key is not None
            and current_episode_key is not None
            and current_episode_key != previous_episode_key
        ):
            self._v6_jersey_conflict_candidate.pop(previous_episode_key, None)
            self._v6_jersey_conflict_streak.pop(previous_episode_key, None)
        return result


__all__ = ["GateDGamePlayerResolverV6"]
