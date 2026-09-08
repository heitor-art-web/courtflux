from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.ball_candidate_filter import filter_smart_candidates


def test_preserves_full_frame_candidates() -> None:
    raw = [{"source": "full_frame", "geometry_score": 0.01, "confidence": 0.2}]
    kept, diag = filter_smart_candidates(raw)
    assert kept == raw
    assert diag["dropped_refinement_candidates"] == 0
    assert diag["dropped_zoom_geometry_candidates"] == 0
    assert diag["rescued_zoom_temporal_candidates"] == 0


def test_refinements_are_not_emitted_by_default() -> None:
    candidates = [{"source": "zoom_candidate_refine", "geometry_score": 1.0}]
    kept, diag = filter_smart_candidates(candidates)
    assert kept == []
    assert diag["dropped_refinement_candidates"] == 1


def test_temporal_zoom_rejects_only_weak_geometry() -> None:
    candidates = [
        {"source": "zoom_temporal", "geometry_score": 0.59, "id": "weak"},
        {"source": "zoom_temporal", "geometry_score": 0.60, "id": "boundary"},
        {"source": "zoom_temporal", "geometry_score": 0.90, "id": "strong"},
    ]
    kept, diag = filter_smart_candidates(candidates, zoom_min_geometry=0.60)
    assert [item["id"] for item in kept] == ["boundary", "strong"]
    assert diag["dropped_zoom_geometry_candidates"] == 1
    assert diag["rescued_zoom_temporal_candidates"] == 0


def test_temporal_rescue_requires_strong_plausible_recovery() -> None:
    candidates = [
        {
            "source": "zoom_temporal",
            "geometry_score": 0.51,
            "temporal_plausible": True,
            "temporal_score": 0.79,
            "zoom_reason": "no_temporally_plausible_candidate",
            "id": "rescued",
        },
        {
            "source": "zoom_temporal",
            "geometry_score": 0.51,
            "temporal_plausible": True,
            "temporal_score": 0.69,
            "zoom_reason": "no_temporally_plausible_candidate",
            "id": "weak_temporal",
        },
        {
            "source": "zoom_temporal",
            "geometry_score": 0.51,
            "temporal_plausible": True,
            "temporal_score": 0.90,
            "zoom_reason": "no_full_frame_candidate",
            "id": "wrong_reason",
        },
    ]
    kept, diag = filter_smart_candidates(candidates)
    assert [item["id"] for item in kept] == ["rescued"]
    assert diag["rescued_zoom_temporal_candidates"] == 1
    assert diag["dropped_zoom_geometry_candidates"] == 2


def test_refinement_emission_remains_available_for_ab() -> None:
    candidate = {"source": "zoom_candidate_refine", "geometry_score": 0.9}
    kept, _ = filter_smart_candidates([candidate], emit_refinements=True)
    assert kept == [candidate]


def main() -> None:
    test_preserves_full_frame_candidates()
    test_refinements_are_not_emitted_by_default()
    test_temporal_zoom_rejects_only_weak_geometry()
    test_temporal_rescue_requires_strong_plausible_recovery()
    test_refinement_emission_remains_available_for_ab()
    print("[PASS] smart candidate filter")


if __name__ == "__main__":
    main()
