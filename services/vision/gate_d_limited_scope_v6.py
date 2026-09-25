from __future__ import annotations

from typing import Any

LIMITED_BOOTSTRAP_MAX_PLAYERS = 8
LIMITED_BOOTSTRAP_MIN_PLAYER_CONFIDENCE = 0.82
LIMITED_BOOTSTRAP_MIN_SURFACE_RATIO = 0.18
LIMITED_BOOTSTRAP_STRONG_SURFACE_RATIO = 0.30
LIMITED_BOOTSTRAP_MIN_TEAM_CONFIDENCE = 0.80


def select_limited_bootstrap_candidates(
    players: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select strong fallback observations for a LIMITED frame with no explicit players.

    This helper is intentionally independent of the detector/resolver stack so its policy
    can be exercised without loading the full vision model runtime.
    """
    ranked: list[tuple[float, float, float, dict[str, Any]]] = []
    for player in list(players or []):
        if not isinstance(player, dict):
            continue
        if player.get("active_on_court") is False:
            continue
        if player.get("identity_eligible") is False:
            continue

        try:
            confidence = float(player.get("confidence", 0.0) or 0.0)
            surface_ratio = float(player.get("local_court_surface_ratio", 0.0) or 0.0)
            team_confidence = float(player.get("team_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < LIMITED_BOOTSTRAP_MIN_PLAYER_CONFIDENCE:
            continue
        if surface_ratio < LIMITED_BOOTSTRAP_MIN_SURFACE_RATIO:
            continue

        source = str(player.get("court_eligibility_source") or "")
        strong_court = bool(
            source == "image_space_court_surface"
            or surface_ratio >= LIMITED_BOOTSTRAP_STRONG_SURFACE_RATIO
        )
        if not strong_court:
            continue

        has_strong_team = bool(
            player.get("team") is not None
            and team_confidence >= LIMITED_BOOTSTRAP_MIN_TEAM_CONFIDENCE
        )
        if not has_strong_team and surface_ratio < LIMITED_BOOTSTRAP_STRONG_SURFACE_RATIO:
            continue

        copy = dict(player)
        copy["active_on_court"] = True
        copy["gate_d_limited_bootstrap"] = True
        copy["gate_d_limited_bootstrap_reason"] = (
            "strong_team_and_court_evidence"
            if has_strong_team
            else "strong_court_surface_evidence"
        )
        ranked.append((surface_ratio, confidence, team_confidence, copy))

    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2]))
    return [row[3] for row in ranked[:LIMITED_BOOTSTRAP_MAX_PLAYERS]]


__all__ = [
    "LIMITED_BOOTSTRAP_MAX_PLAYERS",
    "LIMITED_BOOTSTRAP_MIN_PLAYER_CONFIDENCE",
    "LIMITED_BOOTSTRAP_MIN_SURFACE_RATIO",
    "LIMITED_BOOTSTRAP_STRONG_SURFACE_RATIO",
    "LIMITED_BOOTSTRAP_MIN_TEAM_CONFIDENCE",
    "select_limited_bootstrap_candidates",
]
