from __future__ import annotations

from typing import Any

LIMITED_BOOTSTRAP_MAX_PLAYERS = 8

# Existing/resolved raw tracks may continue through a LIMITED view with weaker image-space
# court evidence: this does not invent a cross-track identity, it preserves an already
# established temporal episode.
LIMITED_BOOTSTRAP_KNOWN_MIN_PLAYER_CONFIDENCE = 0.72
LIMITED_BOOTSTRAP_KNOWN_MIN_SURFACE_RATIO = 0.08

# New raw tracks remain conservative. They must be high-confidence detections with at least
# plausible court surface; the detector layer additionally requires temporal persistence.
LIMITED_BOOTSTRAP_NEW_MIN_PLAYER_CONFIDENCE = 0.88
LIMITED_BOOTSTRAP_NEW_MIN_SURFACE_RATIO = 0.14
LIMITED_BOOTSTRAP_NEW_STRONG_SURFACE_RATIO = 0.16
LIMITED_BOOTSTRAP_MIN_TEAM_CONFIDENCE = 0.80

ALLOWED_COURT_SOURCES = {
    "image_space_court_surface",
    "image_space_ambiguous_surface",
}


def select_limited_bootstrap_candidates(
    players: list[dict[str, Any]],
    *,
    known_track_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Select bounded identity observations when LIMITED explicit scope is empty.

    Two evidence tiers are intentionally separate:

    * ``known_track_ids`` contains tracks whose current resolver episode already has a
      stable game-player ID. They may continue with weaker but still plausible court
      evidence because no new cross-track identity is being invented.
    * unseen tracks require stronger detector/court evidence. The detector wrapper applies
      an additional consecutive-frame persistence gate before those rows are exposed.

    Explicit ``active_on_court=False`` and ``identity_eligible=False`` remain hard rejects.
    """

    known = {int(track_id) for track_id in (known_track_ids or set())}
    ranked: list[tuple[int, float, float, float, dict[str, Any]]] = []

    for player in list(players or []):
        if not isinstance(player, dict):
            continue
        if player.get("active_on_court") is False:
            continue
        if player.get("identity_eligible") is False:
            continue

        try:
            track_id = int(player.get("track_id", -1))
            confidence = float(player.get("confidence", 0.0) or 0.0)
            surface_ratio = float(player.get("local_court_surface_ratio", 0.0) or 0.0)
            team_confidence = float(player.get("team_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if track_id < 0:
            continue

        source = str(player.get("court_eligibility_source") or "")
        if source not in ALLOWED_COURT_SOURCES:
            continue

        is_known = track_id in known
        has_strong_team = bool(
            player.get("team") is not None
            and team_confidence >= LIMITED_BOOTSTRAP_MIN_TEAM_CONFIDENCE
        )

        if is_known:
            if confidence < LIMITED_BOOTSTRAP_KNOWN_MIN_PLAYER_CONFIDENCE:
                continue
            if surface_ratio < LIMITED_BOOTSTRAP_KNOWN_MIN_SURFACE_RATIO:
                continue
            tier = "known_resolved_track"
            tier_rank = 0
        else:
            if confidence < LIMITED_BOOTSTRAP_NEW_MIN_PLAYER_CONFIDENCE:
                continue
            if surface_ratio < LIMITED_BOOTSTRAP_NEW_MIN_SURFACE_RATIO:
                continue
            if (
                not has_strong_team
                and surface_ratio < LIMITED_BOOTSTRAP_NEW_STRONG_SURFACE_RATIO
            ):
                continue
            tier = "persistent_new_track_candidate"
            tier_rank = 1

        copy = dict(player)
        copy["active_on_court"] = True
        copy["gate_d_limited_bootstrap"] = True
        copy["gate_d_limited_bootstrap_tier"] = tier
        copy["gate_d_limited_bootstrap_reason"] = (
            "continue_resolved_track_with_plausible_court_evidence"
            if is_known
            else (
                "strong_team_and_court_evidence"
                if has_strong_team
                else "high_confidence_strong_surface_candidate"
            )
        )
        ranked.append(
            (tier_rank, -surface_ratio, -confidence, -team_confidence, copy)
        )

    ranked.sort(key=lambda row: row[:4])
    return [row[4] for row in ranked[:LIMITED_BOOTSTRAP_MAX_PLAYERS]]


__all__ = [
    "LIMITED_BOOTSTRAP_MAX_PLAYERS",
    "LIMITED_BOOTSTRAP_KNOWN_MIN_PLAYER_CONFIDENCE",
    "LIMITED_BOOTSTRAP_KNOWN_MIN_SURFACE_RATIO",
    "LIMITED_BOOTSTRAP_NEW_MIN_PLAYER_CONFIDENCE",
    "LIMITED_BOOTSTRAP_NEW_MIN_SURFACE_RATIO",
    "LIMITED_BOOTSTRAP_NEW_STRONG_SURFACE_RATIO",
    "LIMITED_BOOTSTRAP_MIN_TEAM_CONFIDENCE",
    "select_limited_bootstrap_candidates",
]
