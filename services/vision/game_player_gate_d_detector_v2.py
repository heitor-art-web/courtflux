from __future__ import annotations

from time import perf_counter
from typing import Any

from services.vision.game_player_gate_d_detector import PerformanceDetector as GateDDetectorV1
from services.vision.game_player_resolver_gate_d_v2 import GateDGamePlayerResolverV2


class PerformanceDetector(GateDDetectorV1):
    """Gate-D detector with frame-cohort visual-shift suppression.

    The parent detector remains the canonical Gate-D bridge. This subclass changes only
    the validation resolver and evaluates all player descriptors before sequentially
    observing tracks, so a frame-wide visual shift can be detected without order effects.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.game_player_resolver = GateDGamePlayerResolverV2()
        print(
            "🧍 Gate D global visual-shift suppression: on",
            flush=True,
        )

    def _observe_game_players(self, frame_state: Any) -> None:
        started = perf_counter()
        frame_id = int(getattr(frame_state, "frame_id", 0))
        frame = getattr(frame_state, "frame", None)
        players = self._identity_eligible_players(frame_state)
        active_track_ids = {
            int(player.get("track_id", -1))
            for player in players
            if int(player.get("track_id", -1)) >= 0
        }

        prepared: list[dict[str, Any]] = []
        descriptors_available = 0
        for player in players:
            track_id = int(player.get("track_id", -1))
            if track_id < 0:
                continue
            crop = self._crop_player(frame, player.get("box"))
            appearance = self.game_player_appearance.describe(crop)
            if appearance is not None:
                descriptors_available += 1
            team_id = player.get("team")
            team_confidence = float(player.get("team_confidence", 0.0) or 0.0)
            jersey_number, jersey_confidence = self._jersey_evidence(player)
            prepared.append(
                {
                    "player": player,
                    "track_id": track_id,
                    "appearance": appearance,
                    "team_id": team_id,
                    "team_confidence": team_confidence,
                    "jersey_number": jersey_number,
                    "jersey_confidence": jersey_confidence,
                }
            )

        frame_context = self.game_player_resolver.prepare_frame(
            frame_id=frame_id,
            observations=[
                (int(item["track_id"]), item["appearance"])
                for item in prepared
            ],
        )

        predictions: list[dict[str, Any]] = []
        resolved = 0
        for item in prepared:
            player = item["player"]
            track_id = int(item["track_id"])
            appearance = item["appearance"]
            team_id = item["team_id"]
            team_confidence = float(item["team_confidence"])
            jersey_number = item["jersey_number"]
            jersey_confidence = float(item["jersey_confidence"])

            game_player_id = self.game_player_resolver.observe(
                frame_id=frame_id,
                track_id=track_id,
                appearance=appearance,
                team_id=(None if team_id is None else str(team_id)),
                team_confidence=team_confidence,
                jersey_number=jersey_number,
                jersey_confidence=jersey_confidence,
                simultaneous_track_ids=active_track_ids,
            )
            resolution = self.game_player_resolver.resolution(track_id, frame_id) or {}
            episode_index = resolution.get("episode_index")

            player["game_player_id"] = game_player_id
            player["game_player_episode"] = episode_index
            if game_player_id is not None:
                resolved += 1

            predictions.append(
                {
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "game_player_id": game_player_id,
                    "episode_index": episode_index,
                    "team": team_id,
                    "team_confidence": round(team_confidence, 6),
                    "jersey_number": jersey_number,
                    "jersey_confidence": round(jersey_confidence, 6),
                    "appearance_available": appearance is not None,
                    "appearance_dimensions": 0 if appearance is None else len(appearance),
                    "association_reason": resolution.get("association_reason"),
                    "association_best_score": self._rounded_optional(
                        resolution.get("association_best_score")
                    ),
                    "association_second_score": self._rounded_optional(
                        resolution.get("association_second_score")
                    ),
                    "association_candidate_count": int(
                        resolution.get("association_candidate_count", 0) or 0
                    ),
                    "episode_open_reason": resolution.get("episode_open_reason"),
                    "episode_open_appearance_similarity": self._rounded_optional(
                        resolution.get("episode_open_appearance_similarity")
                    ),
                    "episode_open_split_similarity": self._rounded_optional(
                        resolution.get("episode_open_split_similarity")
                    ),
                    "episode_open_candidate_similarity": self._rounded_optional(
                        resolution.get("episode_open_candidate_similarity")
                    ),
                    "global_visual_shift": bool(
                        frame_context.get("global_visual_shift", False)
                    ),
                    "global_visual_shift_comparable_tracks": int(
                        frame_context.get("comparable_tracks", 0) or 0
                    ),
                    "global_visual_shift_conflicting_tracks": int(
                        frame_context.get("conflicting_tracks", 0) or 0
                    ),
                    "global_visual_shift_conflict_ratio": self._rounded_optional(
                        frame_context.get("conflict_ratio")
                    ),
                    "box": list(player.get("box", []) or []),
                }
            )

        diag = frame_state.diagnostics.setdefault("detector", {})
        diag["game_player_predictions"] = predictions
        diag["game_player_runtime"] = {
            "version": "game_player_resolver_v2_gate_d",
            "identity_key": "track_id+temporal_episode",
            "unknown_preferred": True,
            "global_visual_shift": bool(
                frame_context.get("global_visual_shift", False)
            ),
            "global_visual_shift_comparable_tracks": int(
                frame_context.get("comparable_tracks", 0) or 0
            ),
            "global_visual_shift_conflicting_tracks": int(
                frame_context.get("conflicting_tracks", 0) or 0
            ),
            "global_visual_shift_conflict_ratio": self._rounded_optional(
                frame_context.get("conflict_ratio")
            ),
            "eligible_players": len(players),
            "tracked_players": len(predictions),
            "appearance_available": descriptors_available,
            "resolved_players": resolved,
            "unknown_players": len(predictions) - resolved,
            "runtime_ms": round((perf_counter() - started) * 1000.0, 4),
        }
