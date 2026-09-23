from __future__ import annotations

from typing import Iterable

from services.vision.game_player_resolver_v1 import (
    AssociationDecision,
    EpisodeKey,
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

    Gate-D runtime also treats entities outside ``max_association_gap_frames`` as
    temporally ineligible rather than as compatible-but-unresolvable candidates. Without
    that distinction, old entities could force a genuinely new player to remain UNKNOWN
    forever even when appearance evidence was available.

    Within-track split detection deliberately uses a coarser body descriptor than
    cross-track association. The full 6x3 appearance grid remains available for player
    discrimination, while split detection pools each row across columns so horizontal pose
    changes do not masquerade as a human-identity switch.
    """

    _SPLIT_GRID_ROWS = 6
    _SPLIT_GRID_COLS = 3
    _SPLIT_CELL_FEATURES = 6

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gate_d_active_track_ids: set[int] = set()
        self._gate_d_decisions: dict[EpisodeKey, AssociationDecision] = {}
        self._gate_d_episode_open: dict[EpisodeKey, dict[str, object]] = {}
        self._gate_d_pending_episode_open: dict[int, dict[str, object]] = {}
        self._gate_d_split_sum: dict[EpisodeKey, list[float]] = {}
        self._gate_d_split_count: dict[EpisodeKey, int] = {}
        self._gate_d_split_candidate: dict[EpisodeKey, tuple[float, ...]] = {}

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

    def _entity_within_association_window(
        self,
        entity: GamePlayerEntity,
        episode: TrackEpisode,
    ) -> bool:
        last_frame = self._entity_last_frame(entity)
        if last_frame is None:
            return False
        gap = max(0, int(episode.frame_start) - int(last_frame))
        return gap <= self.config.max_association_gap_frames

    def _remember_decision(
        self,
        episode: TrackEpisode,
        decision: AssociationDecision,
    ) -> AssociationDecision:
        self._gate_d_decisions[episode.key] = decision
        return decision

    def _choose_entity(self, episode: TrackEpisode) -> AssociationDecision:
        """Choose an entity without letting stale history force UNKNOWN."""
        scored: list[tuple[float, GamePlayerId]] = []
        eligible_entities = 0
        comparable_entities = 0
        for game_player_id, entity in self.entities.items():
            if self._hard_reject_entity(entity, episode):
                continue
            if not self._entity_within_association_window(entity, episode):
                continue
            eligible_entities += 1
            score = self._association_score(entity, episode)
            if score is None:
                continue
            comparable_entities += 1
            if score >= self.config.appearance_plausible_similarity:
                scored.append((score, game_player_id))

        if not scored:
            if eligible_entities == 0:
                return self._remember_decision(
                    episode,
                    AssociationDecision(None, "new_entity_no_compatible_history"),
                )
            if comparable_entities == eligible_entities:
                return self._remember_decision(
                    episode,
                    AssociationDecision(
                        None,
                        "new_entity_strong_visual_separation",
                        candidate_count=eligible_entities,
                    ),
                )
            return self._remember_decision(
                episode,
                AssociationDecision(
                    None,
                    "unknown_insufficient_association_evidence",
                    candidate_count=eligible_entities,
                ),
            )

        scored.sort(reverse=True)
        best_score, best_id = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else None
        if best_score < self.config.appearance_match_similarity:
            return self._remember_decision(
                episode,
                AssociationDecision(
                    None,
                    "unknown_weak_best_match",
                    best_score=best_score,
                    second_score=second_score,
                    candidate_count=len(scored),
                ),
            )
        if (
            second_score is not None
            and best_score - second_score < self.config.appearance_match_margin
        ):
            return self._remember_decision(
                episode,
                AssociationDecision(
                    None,
                    "unknown_ambiguous_match",
                    best_score=best_score,
                    second_score=second_score,
                    candidate_count=len(scored),
                ),
            )
        return self._remember_decision(
            episode,
            AssociationDecision(
                best_id,
                "matched_existing_entity",
                best_score=best_score,
                second_score=second_score,
                candidate_count=len(scored),
            ),
        )

    @classmethod
    def _split_descriptor(
        cls,
        appearance: tuple[float, ...] | None,
    ) -> tuple[float, ...] | None:
        if appearance is None:
            return None
        expected = cls._SPLIT_GRID_ROWS * cls._SPLIT_GRID_COLS * cls._SPLIT_CELL_FEATURES
        if len(appearance) != expected:
            # Synthetic/unit descriptors and future descriptor variants retain the
            # conservative base behavior rather than silently dropping evidence.
            return appearance

        pooled: list[float] = []
        for row in range(cls._SPLIT_GRID_ROWS):
            for feature in range(cls._SPLIT_CELL_FEATURES):
                values = [
                    appearance[
                        ((row * cls._SPLIT_GRID_COLS + col) * cls._SPLIT_CELL_FEATURES)
                        + feature
                    ]
                    for col in range(cls._SPLIT_GRID_COLS)
                ]
                pooled.append(sum(values) / cls._SPLIT_GRID_COLS)
        return tuple(pooled)

    def _split_prototype(self, episode: TrackEpisode) -> tuple[float, ...] | None:
        total = self._gate_d_split_sum.get(episode.key)
        count = int(self._gate_d_split_count.get(episode.key, 0) or 0)
        if total is not None and count > 0:
            return tuple(value / count for value in total)
        return self._split_descriptor(episode.appearance)

    def _seed_split_prototype(
        self,
        episode_key: EpisodeKey,
        appearance: tuple[float, ...] | None,
    ) -> None:
        split = self._split_descriptor(appearance)
        self._gate_d_split_candidate.pop(episode_key, None)
        if split is None:
            return
        self._gate_d_split_sum[episode_key] = [float(value) for value in split]
        self._gate_d_split_count[episode_key] = 1

    def _update_split_prototype(
        self,
        episode_key: EpisodeKey,
        split: tuple[float, ...] | None,
    ) -> None:
        if split is None:
            return
        total = self._gate_d_split_sum.get(episode_key)
        if total is None or len(total) != len(split):
            self._gate_d_split_sum[episode_key] = [float(value) for value in split]
            self._gate_d_split_count[episode_key] = 1
            return
        for index, value in enumerate(split):
            total[index] += float(value)
        self._gate_d_split_count[episode_key] = (
            int(self._gate_d_split_count.get(episode_key, 0) or 0) + 1
        )

    def _remember_episode_open(
        self,
        episode: TrackEpisode,
        *,
        reason: str,
        raw_similarity: float | None,
        split_similarity: float | None,
        candidate_similarity: float | None = None,
    ) -> None:
        self._gate_d_pending_episode_open[episode.track_id] = {
            "reason": reason,
            "appearance_similarity": raw_similarity,
            "split_similarity": split_similarity,
            "candidate_similarity": candidate_similarity,
        }

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
        gap = int(frame_id) - int(episode.frame_end)
        raw_similarity = self._cosine_similarity(episode.appearance, appearance)
        split_appearance = self._split_descriptor(appearance)
        split_similarity = self._cosine_similarity(
            self._split_prototype(episode),
            split_appearance,
        )

        if gap > self.config.max_track_gap_frames:
            self._remember_episode_open(
                episode,
                reason="track_gap",
                raw_similarity=raw_similarity,
                split_similarity=split_similarity,
            )
            return True

        if episode.game_player_id is not None:
            entity = self.entities.get(episode.game_player_id)
            if entity is not None and self._entity_has_other_active_track(entity, episode):
                self._remember_episode_open(
                    episode,
                    reason="active_track_identity_conflict",
                    raw_similarity=raw_similarity,
                    split_similarity=split_similarity,
                )
                return True

        if self._team_incompatible(
            episode.team_id,
            episode.team_confidence,
            team_id,
            team_confidence,
        ):
            episode.team_conflict_streak += 1
        else:
            episode.team_conflict_streak = 0
        if episode.team_conflict_streak >= self.config.split_conflict_observations:
            self._remember_episode_open(
                episode,
                reason="team_conflict",
                raw_similarity=raw_similarity,
                split_similarity=split_similarity,
            )
            return True

        if (
            split_similarity is not None
            and split_similarity < self.config.appearance_split_similarity
        ):
            candidate = self._gate_d_split_candidate.get(episode.key)
            candidate_similarity = self._cosine_similarity(candidate, split_appearance)
            if (
                candidate is None
                or candidate_similarity is None
                or candidate_similarity < self.config.appearance_plausible_similarity
            ):
                self._gate_d_split_candidate[episode.key] = split_appearance
                episode.appearance_conflict_streak = 1
            else:
                self._gate_d_split_candidate[episode.key] = split_appearance
                episode.appearance_conflict_streak += 1

            if (
                episode.appearance_conflict_streak
                >= self.config.split_conflict_observations
            ):
                self._remember_episode_open(
                    episode,
                    reason="appearance_conflict",
                    raw_similarity=raw_similarity,
                    split_similarity=split_similarity,
                    candidate_similarity=candidate_similarity,
                )
                return True
            return False

        episode.appearance_conflict_streak = 0
        self._gate_d_split_candidate.pop(episode.key, None)
        self._update_split_prototype(episode.key, split_appearance)
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
    ) -> GamePlayerId | None:
        track_id = int(track_id)
        descriptor = self._descriptor(appearance)
        previous_episode_key = self.current_episode_by_track.get(track_id)
        previous = self._gate_d_active_track_ids
        self._gate_d_active_track_ids = (
            {int(value) for value in simultaneous_track_ids if int(value) >= 0}
            if simultaneous_track_ids is not None
            else set()
        )
        try:
            result = super().observe(
                frame_id=frame_id,
                track_id=track_id,
                appearance=descriptor,
                team_id=team_id,
                team_confidence=team_confidence,
                jersey_number=jersey_number,
                jersey_confidence=jersey_confidence,
            )
            current_episode_key = self.current_episode_by_track.get(track_id)
            if current_episode_key is not None and current_episode_key != previous_episode_key:
                opening = self._gate_d_pending_episode_open.pop(track_id, None)
                if opening is None:
                    opening = {
                        "reason": "initial_track_observation",
                        "appearance_similarity": None,
                        "split_similarity": None,
                        "candidate_similarity": None,
                    }
                self._gate_d_episode_open[current_episode_key] = opening
                self._seed_split_prototype(current_episode_key, descriptor)
                if previous_episode_key is not None:
                    self._gate_d_split_candidate.pop(previous_episode_key, None)
            return result
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
        decision = self._gate_d_decisions.get(episode.key)
        opening = self._gate_d_episode_open.get(episode.key, {})
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
            "association_reason": None if decision is None else decision.reason,
            "association_best_score": None if decision is None else decision.best_score,
            "association_second_score": None if decision is None else decision.second_score,
            "association_candidate_count": 0 if decision is None else decision.candidate_count,
            "episode_open_reason": opening.get("reason"),
            "episode_open_appearance_similarity": opening.get("appearance_similarity"),
            "episode_open_split_similarity": opening.get("split_similarity"),
            "episode_open_candidate_similarity": opening.get("candidate_similarity"),
        }
