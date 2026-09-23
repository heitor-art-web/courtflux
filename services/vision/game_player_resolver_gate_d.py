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
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gate_d_active_track_ids: set[int] = set()
        self._gate_d_decisions: dict[EpisodeKey, AssociationDecision] = {}
        self._gate_d_episode_open: dict[EpisodeKey, dict[str, object]] = {}
        self._gate_d_pending_episode_open: dict[int, dict[str, object]] = {}

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
        """Choose an entity without letting stale history force UNKNOWN.

        Hard-rejected and temporally stale entities are not candidates. Among recent
        compatible entities the conservative V1 semantics are preserved: a unique strong
        match may merge, clear visual separation creates a new entity, and genuinely weak
        or ambiguous evidence remains UNKNOWN.
        """
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
                    AssociationDecision(None, "new_entity_no_recent_compatible_history"),
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
        similarity = self._cosine_similarity(episode.appearance, appearance)

        if episode.game_player_id is not None:
            entity = self.entities.get(episode.game_player_id)
            if entity is not None and self._entity_has_other_active_track(entity, episode):
                self._gate_d_pending_episode_open[episode.track_id] = {
                    "reason": "active_track_identity_conflict",
                    "appearance_similarity": similarity,
                }
                return True

        split = super()._should_split(
            episode,
            frame_id=frame_id,
            team_id=team_id,
            team_confidence=team_confidence,
            appearance=appearance,
        )
        if not split:
            return False

        if gap > self.config.max_track_gap_frames:
            reason = "track_gap"
        elif episode.team_conflict_streak >= self.config.split_conflict_observations:
            reason = "team_conflict"
        elif episode.appearance_conflict_streak >= self.config.split_conflict_observations:
            reason = "appearance_conflict"
        else:
            reason = "resolver_split"
        self._gate_d_pending_episode_open[episode.track_id] = {
            "reason": reason,
            "appearance_similarity": similarity,
        }
        return True

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
                appearance=appearance,
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
                    }
                self._gate_d_episode_open[current_episode_key] = opening
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
        }
