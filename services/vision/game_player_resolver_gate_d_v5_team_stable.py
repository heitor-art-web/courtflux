from __future__ import annotations

from typing import Iterable

from services.vision.game_player_resolver_gate_d_v5 import GateDGamePlayerResolverV5
from services.vision.game_player_resolver_v1 import EpisodeKey, TrackEpisode


class GateDTeamStableGamePlayerResolverV5(GateDGamePlayerResolverV5):
    """V5 resolver with Identity-v2-confirmed team changes for within-track splitting.

    Identity-v2 already has a conservative team-reversal policy based on consecutive
    accepted opposing rescores. Gate D must not independently promote an unconfirmed
    single-frame/cached opposite-team prediction into a hard identity split.

    This wrapper changes only *within-track* team-change handling:
    * a hard opposite-team observation is held at the established episode team until
      Identity-v2 reports ``recent_reversal_confirmed``;
    * once confirmed, the opposite team is latched long enough for the existing Gate-D
      ``split_conflict_observations`` rule to do its normal work;
    * the established episode team is not overwritten before that split decision completes;
    * appearance conflicts, track gaps and active simultaneous-identity conflicts remain
      independent split causes and keep the raw incoming team on a newly opened episode;
    * cross-track hard team incompatibility is unchanged;
    * callers that do not provide team-stability metadata retain legacy V5 behavior.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gate_d_team_change_confirmation: dict[int, bool] | None = None
        self._gate_d_pending_unconfirmed_team: dict[int, str] = {}
        self._gate_d_confirmed_team_latch: dict[int, str] = {}
        self._gate_d_restore_stable_team_observation: dict[
            int, tuple[EpisodeKey, str | None, float]
        ] = {}

    def prepare_team_stability(self, confirmations: dict[int, bool]) -> None:
        """Provide Identity-v2 reversal confirmations for one detector observation pass."""
        self._gate_d_team_change_confirmation = {
            int(track_id): bool(confirmed)
            for track_id, confirmed in confirmations.items()
            if int(track_id) >= 0
        }
        self._gate_d_restore_stable_team_observation.clear()

    def clear_team_stability(self) -> None:
        """End the current detector observation pass without dropping cross-frame memory."""
        self._gate_d_team_change_confirmation = None
        self._gate_d_restore_stable_team_observation.clear()

    def _team_change_signal(self, track_id: int) -> bool | None:
        current = self._gate_d_team_change_confirmation
        if current is None or int(track_id) not in current:
            return None
        return bool(current[int(track_id)])

    def _team_observation_for_split(
        self,
        episode: TrackEpisode,
        *,
        team_id: str | None,
        team_confidence: float,
    ) -> tuple[str | None, float, bool]:
        """Return team evidence for split evaluation and whether stable team must restore."""
        track_id = int(episode.track_id)
        incoming_team = None if team_id is None else str(team_id)

        if incoming_team is None or not self._team_incompatible(
            episode.team_id,
            episode.team_confidence,
            incoming_team,
            team_confidence,
        ):
            if incoming_team is not None and incoming_team == episode.team_id:
                self._gate_d_pending_unconfirmed_team.pop(track_id, None)
                self._gate_d_confirmed_team_latch.pop(track_id, None)
            return team_id, team_confidence, False

        # No Identity-v2 metadata means legacy behavior. This keeps pure resolver callers,
        # existing DEV tests and non-Identity-v2 integrations backward compatible.
        if self._gate_d_team_change_confirmation is None:
            return team_id, team_confidence, False

        signal = self._team_change_signal(track_id)
        if signal is True:
            self._gate_d_confirmed_team_latch[track_id] = incoming_team
            self._gate_d_pending_unconfirmed_team.pop(track_id, None)
            # Let the existing team-conflict streak see the confirmed opposite team, but
            # restore the stable episode team if this observation alone is not enough to
            # open the new episode. Otherwise the base update would overwrite the old team
            # and erase the conflict before the next observation.
            return team_id, team_confidence, True

        # Confirmation is a one-frame event in Identity-v2 diagnostics. Once observed,
        # keep accepting that same opposite team until the existing Gate-D conflict
        # streak either opens an episode or the classifier returns to the established team.
        if self._gate_d_confirmed_team_latch.get(track_id) == incoming_team:
            return team_id, team_confidence, True

        if signal is False:
            self._gate_d_pending_unconfirmed_team[track_id] = incoming_team
        elif self._gate_d_pending_unconfirmed_team.get(track_id) != incoming_team:
            # Metadata absent for this track and no preceding explicit unconfirmed change:
            # preserve legacy behavior rather than silently inventing a confirmation state.
            return team_id, team_confidence, False

        # Hold the episode's established team for split evaluation. observe() also restores
        # that stable team after the base resolver's normal update if no independent split
        # (appearance/gap/active identity) occurred.
        return episode.team_id, episode.team_confidence, True

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
        split_team, split_confidence, restore_stable = self._team_observation_for_split(
            episode,
            team_id=team_id,
            team_confidence=team_confidence,
        )
        if restore_stable:
            self._gate_d_restore_stable_team_observation[int(episode.track_id)] = (
                episode.key,
                episode.team_id,
                float(episode.team_confidence),
            )
        return super()._should_split(
            episode,
            frame_id=frame_id,
            team_id=split_team,
            team_confidence=split_confidence,
            appearance=appearance,
            active_track_ids=active_track_ids,
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
        restore = self._gate_d_restore_stable_team_observation.pop(track_id, None)

        if (
            restore is not None
            and previous_episode_key is not None
            and current_episode_key == previous_episode_key == restore[0]
        ):
            episode = self.episodes.get(current_episode_key)
            if episode is not None:
                episode.team_id = restore[1]
                episode.team_confidence = float(restore[2])
                if episode.game_player_id is not None:
                    self._rebuild_entity(episode.game_player_id)

        if (
            previous_episode_key is not None
            and current_episode_key is not None
            and current_episode_key != previous_episode_key
        ):
            self._gate_d_pending_unconfirmed_team.pop(track_id, None)
            self._gate_d_confirmed_team_latch.pop(track_id, None)

        return result
