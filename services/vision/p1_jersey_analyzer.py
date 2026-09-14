from __future__ import annotations

from typing import Optional

from services.vision.performance_jersey_analyzer import (
    PerformanceJerseyAnalyzer as BasePerformanceJerseyAnalyzer,
)


class PerformanceJerseyAnalyzer(BasePerformanceJerseyAnalyzer):
    """P1 correctness wrapper for the runtime Jersey recognizer.

    The underlying model is unchanged. This wrapper only tightens acceptance so roster
    conditioning cannot coerce a confident raw digit prediction into a different known
    roster number. In that case we abstain and keep the track unidentified.
    """

    def predict_jersey(self, track_id: int, team_name: Optional[str] = None) -> dict:
        prediction = dict(super().predict_jersey(track_id=track_id, team_name=team_name) or {})
        prediction["p1_raw_roster_consistency_gate"] = True

        if not prediction.get("accepted"):
            return prediction

        candidate = prediction.get("candidate_number")
        raw_canonical = prediction.get("raw_canonical")
        if candidate is None or raw_canonical is None:
            return prediction

        if str(candidate) != str(raw_canonical):
            prediction["accepted"] = False
            prediction["rejection_reason"] = "raw_roster_disagreement"
            prediction["rejected_candidate_number"] = str(candidate)

        return prediction
