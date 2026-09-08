from __future__ import annotations

from typing import Any


def filter_smart_candidates(
    candidates: list[dict[str, Any]],
    *,
    emit_refinements: bool = False,
    zoom_min_geometry: float = 0.60,
    temporal_rescue_min_score: float = 0.70,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Conservatively prune smart-only proposals while never touching full-frame candidates.

    Temporal zoom proposals normally need enough ball-like geometry. A narrow rescue is
    allowed when the crop itself lands on a temporally plausible candidate with a strong
    temporal score after the full-frame stream had no temporally plausible candidate.
    This protects short recovery chains without broadly reopening low-geometry proposals.
    """
    kept: list[dict[str, Any]] = []
    dropped_refinements = 0
    dropped_zoom_geometry = 0
    rescued_zoom_temporal = 0

    for candidate in candidates:
        source = str(candidate.get("source") or "full_frame")
        if source == "zoom_candidate_refine" and not emit_refinements:
            dropped_refinements += 1
            continue

        if source == "zoom_temporal":
            geometry = float(candidate.get("geometry_score", 0.0) or 0.0)
            if geometry < float(zoom_min_geometry):
                rescue = (
                    str(candidate.get("zoom_reason") or "")
                    == "no_temporally_plausible_candidate"
                    and candidate.get("temporal_plausible") is True
                    and float(candidate.get("temporal_score", 0.0) or 0.0)
                    >= float(temporal_rescue_min_score)
                )
                if rescue:
                    rescued_zoom_temporal += 1
                else:
                    dropped_zoom_geometry += 1
                    continue

        kept.append(candidate)

    return kept, {
        "dropped_refinement_candidates": dropped_refinements,
        "dropped_zoom_geometry_candidates": dropped_zoom_geometry,
        "rescued_zoom_temporal_candidates": rescued_zoom_temporal,
    }
