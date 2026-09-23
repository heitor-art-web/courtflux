from __future__ import annotations

from time import perf_counter
from typing import Any

from services.vision.game_player_appearance_v1 import GamePlayerAppearanceV1
from services.vision.game_player_resolver_gate_d import GateDGamePlayerResolverV1
from services.vision.identity_v2_detector import PerformanceDetector as IdentityV2Detector


class PerformanceDetector(IdentityV2Detector):
    """Gate-D validation detector that adds temporal game-player identity diagnostics.

    The canonical Identity-v2 detector remains untouched. This subclass is selected only
    by the dedicated Gate-D probe, so normal P1/P2 runtime behavior is unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.game_player_appearance = GamePlayerAppearanceV1()
        self.game_player_resolver = GateDGamePlayerResolverV1()
        print(
            "🧍 Gate D game-player resolver: on "
            "identity=(track_id, temporal_episode) appearance=full_player_v1",
            flush=True,
        )

    @staticmethod
    def _crop_player(frame: Any, box: Any) -> Any | None:
        if frame is None or box is None:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in list(box)[:4]]
            height, width = int(frame.shape[0]), int(frame.shape[1])
        except (TypeError, ValueError, IndexError, AttributeError):
            return None
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _jersey_evidence(player: dict[str, Any]) -> tuple[str | None, float]:
        number = player.get("shirt_num")
        if number is None:
            return None, 0.0
        # Identity-v2 exposes shirt_num only from its reversible temporal profile. It is
        # deliberately still soft evidence here; it never becomes a game-player key.
        confidence = 0.75 if bool(player.get("identity_locked", False)) else 0.0
        return str(number), confidence

    @staticmethod
    def _rounded_optional(value: object) -> float | None:
        if value is None:
            return None
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return None

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

        predictions: list[dict[str, Any]] = []
        descriptors_available = 0
        resolved = 0
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
                    "box": list(player.get("box", []) or []),
                }
            )

        diag = frame_state.diagnostics.setdefault("detector", {})
        diag["game_player_predictions"] = predictions
        diag["game_player_runtime"] = {
            "version": "game_player_resolver_v1",
            "identity_key": "track_id+temporal_episode",
            "unknown_preferred": True,
            "eligible_players": len(players),
            "tracked_players": len(predictions),
            "appearance_available": descriptors_available,
            "resolved_players": resolved,
            "unknown_players": len(predictions) - resolved,
            "runtime_ms": round((perf_counter() - started) * 1000.0, 4),
        }

    def process(self, frame_state: Any):
        result = super().process(frame_state)
        if getattr(result, "frame", None) is not None:
            self._observe_game_players(result)
        return result
