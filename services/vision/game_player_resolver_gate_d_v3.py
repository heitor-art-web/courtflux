from __future__ import annotations

from typing import Iterable

from services.vision.game_player_resolver_gate_d_v2 import GateDGamePlayerResolverV2


class GateDGamePlayerResolverV3(GateDGamePlayerResolverV2):
    """Gate-D v2 with global-shift preflight aligned to both split pathways.

    V2 added a relaxed raw+coarse within-track split pathway, but its frame preflight
    still counted only strict coarse conflicts. Real V4 therefore produced many
    ``appearance_conflict_raw`` episode openings while reporting zero global visual
    shifts. V3 changes only that preflight classification: a track contributes to a
    frame-wide visual shock when either the strict coarse rule or the same raw+coarse
    rule used by the v2 split path is active.
    """

    def prepare_frame(
        self,
        *,
        frame_id: int,
        observations: Iterable[tuple[int, Iterable[float] | None]],
    ) -> dict[str, object]:
        frame_id = int(frame_id)
        comparable_track_ids: list[int] = []
        conflict_track_ids: list[int] = []
        coarse_conflict_track_ids: list[int] = []
        raw_conflict_track_ids: list[int] = []
        relaxed_coarse_ceiling = max(
            self.config.appearance_split_similarity,
            self.global_shift_config.raw_split_coarse_similarity_ceiling,
        )

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
            split_similarity = self._cosine_similarity(
                self._split_prototype(episode),
                split_descriptor,
            )
            if split_similarity is None:
                continue

            raw_prototype = self._raw_prototype(episode)
            raw_similarity = self._cosine_similarity(raw_prototype, descriptor)
            raw_coarse_similarity = self._cosine_similarity(
                self._split_descriptor(raw_prototype),
                split_descriptor,
            )
            comparable_track_ids.append(track_id)

            coarse_conflict = bool(
                split_similarity < self.config.appearance_split_similarity
            )
            raw_conflict = bool(
                raw_similarity is not None
                and raw_coarse_similarity is not None
                and raw_similarity < self.config.appearance_split_similarity
                and raw_coarse_similarity < relaxed_coarse_ceiling
            )
            if coarse_conflict:
                coarse_conflict_track_ids.append(track_id)
            if raw_conflict:
                raw_conflict_track_ids.append(track_id)
            if coarse_conflict or raw_conflict:
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
            "coarse_conflicting_tracks": len(coarse_conflict_track_ids),
            "raw_conflicting_tracks": len(raw_conflict_track_ids),
        }
        return dict(self._gate_d_frame_context)
