from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable


GamePlayerId = str
EpisodeKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class GamePlayerResolverConfig:
    """Conservative defaults for temporal game-player association.

    The resolver deliberately prefers UNKNOWN over a speculative merge. These values
    are implementation defaults, not Gate-D thresholds; they are expected to be
    validated/tuned only on the development corpus (V1/V3/V4/V5).
    """

    max_track_gap_frames: int = 24
    team_hard_confidence: float = 0.70
    appearance_match_similarity: float = 0.88
    appearance_plausible_similarity: float = 0.78
    appearance_match_margin: float = 0.06
    appearance_split_similarity: float = 0.58
    split_conflict_observations: int = 2
    jersey_positive_bonus: float = 0.04
    jersey_negative_penalty: float = 0.06
    max_association_gap_frames: int = 180


@dataclass(slots=True)
class TrackEpisode:
    track_id: int
    episode_index: int
    frame_start: int
    frame_end: int
    observation_count: int = 0
    game_player_id: GamePlayerId | None = None
    team_id: str | None = None
    team_confidence: float = 0.0
    jersey_number: str | None = None
    jersey_confidence: float = 0.0
    appearance_sum: list[float] | None = None
    appearance_count: int = 0
    appearance_conflict_streak: int = 0
    team_conflict_streak: int = 0

    @property
    def key(self) -> EpisodeKey:
        return (self.track_id, self.episode_index)

    @property
    def appearance(self) -> tuple[float, ...] | None:
        if not self.appearance_sum or self.appearance_count <= 0:
            return None
        return tuple(value / self.appearance_count for value in self.appearance_sum)

    def add_appearance(self, descriptor: tuple[float, ...] | None) -> None:
        if descriptor is None:
            return
        if self.appearance_sum is None:
            self.appearance_sum = [0.0] * len(descriptor)
        if len(self.appearance_sum) != len(descriptor):
            raise ValueError("appearance descriptor width changed within an episode")
        for index, value in enumerate(descriptor):
            self.appearance_sum[index] += float(value)
        self.appearance_count += 1


@dataclass(slots=True)
class GamePlayerEntity:
    game_player_id: GamePlayerId
    episode_keys: list[EpisodeKey] = field(default_factory=list)
    team_id: str | None = None
    team_confidence: float = 0.0
    jersey_number: str | None = None
    jersey_confidence: float = 0.0
    appearance_sum: list[float] | None = None
    appearance_count: int = 0

    @property
    def appearance(self) -> tuple[float, ...] | None:
        if not self.appearance_sum or self.appearance_count <= 0:
            return None
        return tuple(value / self.appearance_count for value in self.appearance_sum)

    def absorb_episode(self, episode: TrackEpisode) -> None:
        if episode.key not in self.episode_keys:
            self.episode_keys.append(episode.key)
        if episode.team_id and episode.team_confidence >= self.team_confidence:
            self.team_id = episode.team_id
            self.team_confidence = episode.team_confidence
        if episode.jersey_number and episode.jersey_confidence >= self.jersey_confidence:
            self.jersey_number = episode.jersey_number
            self.jersey_confidence = episode.jersey_confidence
        appearance = episode.appearance
        if appearance is not None:
            if self.appearance_sum is None:
                self.appearance_sum = [0.0] * len(appearance)
            if len(self.appearance_sum) != len(appearance):
                raise ValueError("appearance descriptor width changed between episodes")
            weight = max(1, episode.appearance_count)
            for index, value in enumerate(appearance):
                self.appearance_sum[index] += float(value) * weight
            self.appearance_count += weight


@dataclass(frozen=True, slots=True)
class AssociationDecision:
    game_player_id: GamePlayerId | None
    reason: str
    best_score: float | None = None
    second_score: float | None = None
    candidate_count: int = 0


class GamePlayerResolverV1:
    """Resolve `(track_id, temporal_episode)` into stable game-player entities.

    Raw tracker IDs and jersey OCR are evidence only. A track may split into multiple
    episodes after a sustained strong contradiction. Cross-track association is
    conservative: simultaneous episodes and strong team conflicts are hard rejects;
    ambiguous matches abstain to UNKNOWN.
    """

    def __init__(self, config: GamePlayerResolverConfig | None = None) -> None:
        self.config = config or GamePlayerResolverConfig()
        self.episodes: dict[EpisodeKey, TrackEpisode] = {}
        self.entities: dict[GamePlayerId, GamePlayerEntity] = {}
        self.current_episode_by_track: dict[int, EpisodeKey] = {}
        self._next_entity_index = 1

    @staticmethod
    def _descriptor(values: Iterable[float] | None) -> tuple[float, ...] | None:
        if values is None:
            return None
        descriptor = tuple(float(value) for value in values)
        return descriptor if descriptor else None

    @staticmethod
    def _cosine_similarity(
        left: tuple[float, ...] | None,
        right: tuple[float, ...] | None,
    ) -> float | None:
        if left is None or right is None:
            return None
        if len(left) != len(right) or not left:
            return None
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            return None
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    def _new_entity_id(self) -> GamePlayerId:
        value = f"game_player_{self._next_entity_index:03d}"
        self._next_entity_index += 1
        return value

    def _episode_for_track(self, track_id: int) -> TrackEpisode | None:
        key = self.current_episode_by_track.get(int(track_id))
        return self.episodes.get(key) if key is not None else None

    def _episodes_overlap(self, left: TrackEpisode, right: TrackEpisode) -> bool:
        return left.frame_start <= right.frame_end and right.frame_start <= left.frame_end

    def _entity_is_simultaneous(self, entity: GamePlayerEntity, episode: TrackEpisode) -> bool:
        for key in entity.episode_keys:
            other = self.episodes.get(key)
            if other is not None and self._episodes_overlap(other, episode):
                return True
        return False

    def _team_incompatible(
        self,
        left_team: str | None,
        left_confidence: float,
        right_team: str | None,
        right_confidence: float,
    ) -> bool:
        return bool(
            left_team
            and right_team
            and left_team != right_team
            and left_confidence >= self.config.team_hard_confidence
            and right_confidence >= self.config.team_hard_confidence
        )

    def _hard_reject_entity(self, entity: GamePlayerEntity, episode: TrackEpisode) -> bool:
        if self._entity_is_simultaneous(entity, episode):
            return True
        return self._team_incompatible(
            entity.team_id,
            entity.team_confidence,
            episode.team_id,
            episode.team_confidence,
        )

    def _entity_last_frame(self, entity: GamePlayerEntity) -> int | None:
        frames = [
            self.episodes[key].frame_end
            for key in entity.episode_keys
            if key in self.episodes
        ]
        return max(frames) if frames else None

    def _association_score(
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
        last_frame = self._entity_last_frame(entity)
        if last_frame is not None:
            gap = max(0, episode.frame_start - last_frame)
            if gap > self.config.max_association_gap_frames:
                return None
        return score

    def _choose_entity(self, episode: TrackEpisode) -> AssociationDecision:
        scored: list[tuple[float, GamePlayerId]] = []
        compatible_entities = 0
        comparable_entities = 0
        for game_player_id, entity in self.entities.items():
            if self._hard_reject_entity(entity, episode):
                continue
            compatible_entities += 1
            score = self._association_score(entity, episode)
            if score is not None:
                comparable_entities += 1
                if score >= self.config.appearance_plausible_similarity:
                    scored.append((score, game_player_id))

        if not scored:
            if compatible_entities == 0:
                return AssociationDecision(None, "new_entity_no_compatible_history")
            if comparable_entities == compatible_entities:
                return AssociationDecision(
                    None,
                    "new_entity_strong_visual_separation",
                    candidate_count=compatible_entities,
                )
            return AssociationDecision(
                None,
                "unknown_insufficient_association_evidence",
                candidate_count=compatible_entities,
            )

        scored.sort(reverse=True)
        best_score, best_id = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else None
        if best_score < self.config.appearance_match_similarity:
            return AssociationDecision(
                None,
                "unknown_weak_best_match",
                best_score=best_score,
                second_score=second_score,
                candidate_count=len(scored),
            )
        if (
            second_score is not None
            and best_score - second_score < self.config.appearance_match_margin
        ):
            return AssociationDecision(
                None,
                "unknown_ambiguous_match",
                best_score=best_score,
                second_score=second_score,
                candidate_count=len(scored),
            )
        return AssociationDecision(
            best_id,
            "matched_existing_entity",
            best_score=best_score,
            second_score=second_score,
            candidate_count=len(scored),
        )

    def _attach_episode(self, episode: TrackEpisode, game_player_id: GamePlayerId) -> None:
        entity = self.entities.setdefault(
            game_player_id,
            GamePlayerEntity(game_player_id=game_player_id),
        )
        episode.game_player_id = game_player_id
        if episode.key not in entity.episode_keys:
            entity.episode_keys.append(episode.key)
        self._rebuild_entity(game_player_id)

    def _rebuild_entity(self, game_player_id: GamePlayerId) -> None:
        entity = self.entities[game_player_id]
        keys = list(entity.episode_keys)
        entity.team_id = None
        entity.team_confidence = 0.0
        entity.jersey_number = None
        entity.jersey_confidence = 0.0
        entity.appearance_sum = None
        entity.appearance_count = 0
        entity.episode_keys = []
        for key in keys:
            episode = self.episodes.get(key)
            if episode is not None:
                entity.absorb_episode(episode)

    def _create_episode(
        self,
        *,
        track_id: int,
        frame_id: int,
        team_id: str | None,
        team_confidence: float,
        jersey_number: str | None,
        jersey_confidence: float,
        appearance: tuple[float, ...] | None,
    ) -> TrackEpisode:
        previous = [key[1] for key in self.episodes if key[0] == track_id]
        episode_index = max(previous, default=-1) + 1
        episode = TrackEpisode(
            track_id=track_id,
            episode_index=episode_index,
            frame_start=frame_id,
            frame_end=frame_id,
            observation_count=1,
            team_id=team_id,
            team_confidence=max(0.0, min(float(team_confidence), 1.0)),
            jersey_number=str(jersey_number) if jersey_number is not None else None,
            jersey_confidence=max(0.0, min(float(jersey_confidence), 1.0)),
        )
        episode.add_appearance(appearance)
        self.episodes[episode.key] = episode
        self.current_episode_by_track[track_id] = episode.key

        decision = self._choose_entity(episode)
        if decision.game_player_id is not None:
            self._attach_episode(episode, decision.game_player_id)
        elif decision.reason in {
            "new_entity_no_compatible_history",
            "new_entity_strong_visual_separation",
        }:
            self._attach_episode(episode, self._new_entity_id())
        return episode

    def _should_split(
        self,
        episode: TrackEpisode,
        *,
        frame_id: int,
        team_id: str | None,
        team_confidence: float,
        appearance: tuple[float, ...] | None,
    ) -> bool:
        if frame_id - episode.frame_end > self.config.max_track_gap_frames:
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

        similarity = self._cosine_similarity(episode.appearance, appearance)
        if similarity is not None and similarity < self.config.appearance_split_similarity:
            episode.appearance_conflict_streak += 1
        else:
            episode.appearance_conflict_streak = 0

        return bool(
            episode.team_conflict_streak >= self.config.split_conflict_observations
            or episode.appearance_conflict_streak >= self.config.split_conflict_observations
        )

    def _refresh_assignment(self, episode: TrackEpisode) -> None:
        if episode.game_player_id is not None:
            return
        decision = self._choose_entity(episode)
        if decision.game_player_id is not None:
            self._attach_episode(episode, decision.game_player_id)

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
    ) -> GamePlayerId | None:
        """Observe one tracked player and return the currently resolved game-player ID."""
        frame_id = int(frame_id)
        track_id = int(track_id)
        if track_id < 0:
            return None
        descriptor = self._descriptor(appearance)
        team_confidence = max(0.0, min(float(team_confidence), 1.0))
        jersey_confidence = max(0.0, min(float(jersey_confidence), 1.0))

        episode = self._episode_for_track(track_id)
        if episode is None or self._should_split(
            episode,
            frame_id=frame_id,
            team_id=team_id,
            team_confidence=team_confidence,
            appearance=descriptor,
        ):
            episode = self._create_episode(
                track_id=track_id,
                frame_id=frame_id,
                team_id=team_id,
                team_confidence=team_confidence,
                jersey_number=jersey_number,
                jersey_confidence=jersey_confidence,
                appearance=descriptor,
            )
            return episode.game_player_id

        episode.frame_end = max(episode.frame_end, frame_id)
        episode.observation_count += 1
        if team_id and team_confidence >= episode.team_confidence:
            episode.team_id = str(team_id)
            episode.team_confidence = team_confidence
        if jersey_number is not None and jersey_confidence >= episode.jersey_confidence:
            episode.jersey_number = str(jersey_number)
            episode.jersey_confidence = jersey_confidence
        episode.add_appearance(descriptor)
        self._refresh_assignment(episode)
        if episode.game_player_id is not None:
            self._rebuild_entity(episode.game_player_id)
        return episode.game_player_id

    def resolve(self, track_id: int, frame_id: int) -> GamePlayerId | None:
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
        return matches[0].game_player_id

    def resolve_interval(
        self,
        track_id: int,
        frame_start: int,
        frame_end: int,
    ) -> GamePlayerId | None:
        track_id = int(track_id)
        frame_start = int(frame_start)
        frame_end = max(frame_start, int(frame_end))
        overlapping = [
            episode
            for episode in self.episodes.values()
            if episode.track_id == track_id
            and episode.frame_start <= frame_end
            and frame_start <= episode.frame_end
        ]
        if not overlapping or any(episode.game_player_id is None for episode in overlapping):
            return None
        identities = {episode.game_player_id for episode in overlapping}
        return next(iter(identities)) if len(identities) == 1 else None

    def snapshot(self) -> dict[str, object]:
        return {
            "episodes": [
                {
                    "track_id": episode.track_id,
                    "episode_index": episode.episode_index,
                    "frame_start": episode.frame_start,
                    "frame_end": episode.frame_end,
                    "game_player_id": episode.game_player_id,
                    "team_id": episode.team_id,
                    "jersey_number": episode.jersey_number,
                    "appearance_count": episode.appearance_count,
                }
                for episode in sorted(
                    self.episodes.values(),
                    key=lambda value: (value.track_id, value.episode_index),
                )
            ],
            "entities": [
                {
                    "game_player_id": entity.game_player_id,
                    "episode_keys": [list(key) for key in entity.episode_keys],
                    "team_id": entity.team_id,
                    "jersey_number": entity.jersey_number,
                    "appearance_count": entity.appearance_count,
                }
                for entity in self.entities.values()
            ],
        }
