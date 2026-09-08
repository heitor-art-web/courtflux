from __future__ import annotations

from typing import Any


def filter_smart_candidates(
    candidates: list[dict[str, Any]],
    *,
    emit_refinements: bool = False,
    zoom_min_geometry: float = 0.60,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Conservatively prune smart-only proposals while never touching full-frame candidates."""
    kept: list[dict[str, Any]] = []
    dropped_refinements = 0
    dropped_zoom_geometry = 0

    for candidate in candidates:
        source = str(candidate.get("source") or "full_frame")
        if source == "zoom_candidate_refine" and not emit_refinements:
            dropped_refinements += 1
            continue
        if source == "zoom_temporal":
            geometry = float(candidate.get("geometry_score", 0.0) or 0.0)
            if geometry < float(zoom_min_geometry):
                dropped_zoom_geometry += 1
                continue
        kept.append(candidate)

    return kept, {
        "dropped_refinement_candidates": dropped_refinements,
        "dropped_zoom_geometry_candidates": dropped_zoom_geometry,
    }
