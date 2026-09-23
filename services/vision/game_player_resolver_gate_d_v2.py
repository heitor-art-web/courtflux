from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from services.vision.game_player_resolver_gate_d import GateDGamePlayerResolverV1
from services.vision.game_player_resolver_v1 import EpisodeKey, TrackEpisode


@dataclass(frozen=True, slots=True)
class GateDGlobalVisualShiftConfig:
    """Gate-D-only safeguards for coherent frame-wide appearance shocks."""

    min_track_observations: int = 6
    min_comparable_tracks: int = 4
    min_conflicting_tracks: int = 3
    min_conflict_ratio: float = 0.35
    raw_split_coarse_similarity_ceiling: float = 0.82


class GateDGamePlayerResolverV2(GateDGamePlayerResolverV1):
    """V1 Gate-D resolver plus frame-cohort visual-shift suppression.

    V4 showed two distinct signals that must not be conflated:
    * an isolated same-track visual discontinuity can represent a real tracker identity
      switch and must still be allowed to open a new temporal episode;
    * several established tracks changing appearance together is usually a global visual
      transition (camera/pose/crop regime), and must not fragment every player identity.

    Cross-track association and all V1 hard constraints remain unchanged. The extra logic
    affects only within-track appearance splitting in the dedicated Gate-D validation path.
    """

    def __init__(
        self,
        *args,
        global_shift_config: GateDGlobalVisualShiftConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.global_shift_config = global_shift_config or GateDGlobalVisualShiftConfig()
        self._gate_d_frame_context: dict[str, object] = {
            "frame_id": -1,
            "global_visual_shift": False,
            "comparable_tracks": 0,
            "conflicting_tracks": 0,
            "conflict_ratio": 0.0,
            "conflict_track_ids": [],
        }
        self._gate_d_global_suppressed_track_ids: set[int] = set()
        self._gate_d_raw_sum: dict[EpisodeKey, list[float]] = {}
        self._gate_d_raw_count: dict[EpisodeKey, int] = {}
        self._gate_d_raw_candidate: dict[EpisodeKey, tuple[float, ...]] = {}
        self._gate_d_raw_streak: dict[EpisodeKey, int] = {}

    def _raw_prototype(self, episode: TrackEpisode) -> tuple[float, ...] | None:
        total = self._gate_d_raw_sum.get(episode.key)
        count = int(self._gate_d_raw_count.get(episode.key, 0) or 0)
        if total is not None and count > 0:
            return tuple(value / count for value in total)
        return episode.appearance

    def _seed_raw_prototype(
        self,
        episode_key: EpisodeKey,
        appearance: tuple[float, ...] | None,
    ) -> None:
        self._gate_d_raw_candidate.pop(episode_key, None)
        self._gate_d_raw_streak.pop(episode_key, None)
        if appearance is None:
            self._gate_d_raw_sum.pop(episode_key, None)
            self._gate_d_raw_count.pop(episode_key, None)
            return
        self._gate_d_raw_sum[episode_key] = [float(value) for value in appearance]
        self._gate_d_raw_count[episode_key] = 1

    def _update_raw_prototype(
        self,
        episode_key: EpisodeKey,
        appearance: tuple[float, ...] | None,
    ) -> None:
        if appearance is None:
            return
        total = self._gate_d_raw_sum.get(episode_key)
        if total is None or len(total) != len(appearance):
            self._seed_raw_prototype(episode_key, appearance)
            return
        for index, value in enumerate(appearance):
            total[index] += float(value)
        self._gate_d_raw_count[episode_key] = (
            int(self._gate_d_raw_count.get(episode_key, 0) or 0) + 1
        )

    def _reset_raw_conflict(self, episode_key: EpisodeKey) -> None:
        self._gate_d_raw_candidate.pop(episode_key, None)
        self._gate_d_raw_streak.pop(episode_key, None)

    def prepare_frame(
        self,
        *,
        frame_id: int,
        observations: Iterable[tuple[int, Iterable[float] | None]],
    ) -> dict[str, object]:
        """Inspect all established tracks before any sequential observe() mutates state."""
        frame_id = int(frame_id)
        comparable_track_ids: list[int] = []
        conflict_track_ids: list[int] = []

        for raw_track_id, values in observations:
            track_id = int(raw_track_id)
            if track_id < 0:
                continue
            episode = self._episode_for_track(track_id)
            if (
                episode is None
                or episode.observation_count < self.global_shift_config.min_track_observations
            ):
                continue
            descriptor = self._descriptor(values)
            split_descriptor = self._split_descriptor(descriptor)
            similarity = self._cosine_similarity(
                self._split_prototype(episode),
                split_descriptor,
            )
            if similarity is None:
                continue
            comparable_track_ids.append(track_id)
            if similarity < self.config.appearance_split_similarity:
                conflict_track_ids.append(track_id)

        comparable = len(comparable_track_ids)
        conflicting = len(conflict_track_ids)
        ratio = (conflicting / comparable) if comparable else 0.0
        global_shift = bool(
            comparable >= self.global_shift_config.min_comparable_tracks
            and conflicting >= self.global_shift_config.min_conflicting_tracks
            and ratio >= self.global_shift_config.min_conflict_ratio
        )

        self._gate_d_global_suppressed_track_ids = (
            set(comparable_track_ids) if global_shift else set()
        )
        self._gate_d_frame_context = {
            "frame_id": frame_id,
            "global_visual_shift": global_shift,
            "comparable_tracks": comparable,
            "conflicting_tracks": conflicting,
            "conflict_ratio": round(ratio, 6),
            "conflict_track_ids": sorted(conflict_track_ids),
        }
        return dict(self._gate_d_frame_context)

    def frame_context(self, frame_id: int) -> dict[str, object]:
        if int(self._gate_d_frame_context.get("frame_id", -1)) != int(frame_id):
            return {
                "frame_id": int(frame_id),
                "global_visual_shift": False,
                "comparable_tracks": 0,
                "conflicting_tracks": 0,
                "conflict_ratio": 0.0,
                "conflict_track_ids": [],
            }
        return dict(self._gate_d_frame_context)

    def _global_shift_suppresses(self, episode: TrackEpisode, frame_id: int) -> bool:
        return bool(
            int(self._gate_d_frame_context.get("frame_id", -1)) == int(frame_id)
            and bool(self._gate_d_frame_context.get("global_visual_shift", False))
            and episode.track_id in self._gate_d_global_suppressed_track_ids
        )

    def _rebase_after_global_shift(
        self,
        episode: TrackEpisode,
        appearance: tuple[float, ...] | None,
    ) -> None:
        episode.appearance_conflict_streak = 0
        self._gate_d_split_candidate.pop(episode.key, None)
        self._seed_split_prototype(episode.key, appearance)
        self._seed_raw_prototype(episode.key, appearance)

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
        raw_prototype = self._raw_prototype(episode)
        raw_similarity = self._cosine_similarity(raw_prototype, appearance)
        split_appearance = self._split_descriptor(appearance)
        raw_coarse_similarity = self._cosine_similarity(
            self._split_descriptor(raw_prototype),
            split_appearance,
        )

        base_split = super()._should_split(
            episode,
            frame_id=frame_id,
            team_id=team_id,
            team_confidence=team_confidence,
            appearance=appearance,
            active_track_ids=active_track_ids,
        )

        if self._global_shift_suppresses(episode, frame_id):
            pending = self._gate_d_pending_episode_open.get(episode.track_id, {})
            reason = pending.get("reason")
            if base_split and reason != "appearance_conflict":
                self._reset_raw_conflict(episode.key)
                return True
            self._gate_d_pending_episode_open.pop(episode.track_id, None)
            self._rebase_after_global_shift(episode, appearance)
            return False

        if base_split:
            self._reset_raw_conflict(episode.key)
            return True

        relaxed_coarse_ceiling = max(
            self.config.appearance_split_similarity,
            self.global_shift_config.raw_split_coarse_similarity_ceiling,
        )
        raw_conflict = bool(
            raw_similarity is not None
            and raw_coarse_similarity is not None
            and raw_similarity < self.config.appearance_split_similarity
            and raw_coarse_similarity < relaxed_coarse_ceiling
        )
        if raw_conflict:
            candidate = self._gate_d_raw_candidate.get(episode.key)
            candidate_similarity = self._cosine_similarity(candidate, appearance)
            if (
                candidate is None
                or candidate_similarity is None
                or candidate_similarity < self.config.appearance_plausible_similarity
            ):
                self._gate_d_raw_candidate[episode.key] = appearance
                self._gate_d_raw_streak[episode.key] = 1
            else:
                self._gate_d_raw_candidate[episode.key] = appearance
                self._gate_d_raw_streak[episode.key] = (
                    int(self._gate_d_raw_streak.get(episode.key, 0) or 0) + 1
                )

            if (
                int(self._gate_d_raw_streak.get(episode.key, 0) or 0)
                >= self.config.split_conflict_observations
            ):
                self._remember_episode_open(
                    episode,
                    reason="appearance_conflict_raw",
                    raw_similarity=raw_similarity,
                    split_similarity=raw_coarse_similarity,
                    candidate_similarity=candidate_similarity,
                )
                return True
            return False

        self._reset_raw_conflict(episode.key)
        self._update_raw_prototype(episode.key, appearance)
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
        descriptor = self._descriptor(appearance)
        previous_episode_key = self.current_episode_by_track.get(track_id)
        result = super().observe(
            frame_id=frame_id,
            track_id=track_id,
            appearance=descriptor,
            team_id=team_id,
            team_confidence=team_confidence,
            jersey_number=jersey_number,
            jersey_confidence=jersey_confidence,
            simultaneous_track_ids=simultaneous_track_ids,
        )
        current_episode_key = self.current_episode_by_track.get(track_id)
        if current_episode_key is not None and current_episode_key != previous_episode_key:
            self._seed_raw_prototype(current_episode_key, descriptor)
            if previous_episode_key is not None:
                self._reset_raw_conflict(previous_episode_key)
        return result
