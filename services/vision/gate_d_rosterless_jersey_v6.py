from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GateDRosterlessJerseyEvidenceConfigV6:
    """Conservative visual-number aggregation used only by Gate-D V6."""

    window_frames: int = 96
    min_observations: int = 3
    min_candidate_posterior: float = 0.80
    min_crop_quality: float = 0.40
    min_winner_share: float = 0.70
    min_winner_margin: float = 0.30


class GateDRosterlessJerseyEvidenceV6:
    """Aggregate valid raw OCR rejected only because no roster was supplied.

    The normal P1 identity contract remains unchanged: rosterless OCR is still *not*
    accepted as a canonical jersey identity. Gate D consumes it only as soft visual
    evidence after repeated, high-confidence agreement over a short temporal window.
    """

    def __init__(
        self,
        config: GateDRosterlessJerseyEvidenceConfigV6 | None = None,
    ) -> None:
        self.config = config or GateDRosterlessJerseyEvidenceConfigV6()
        self._history: dict[int, list[tuple[int, str, float]]] = {}
        self.accepted_raw_observations = 0
        self.stable_evidence_emissions = 0

    @staticmethod
    def _number(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or not text.isdigit():
            return None
        number = int(text)
        if not 0 <= number <= 99:
            return None
        return "00" if text == "00" else str(number)

    @staticmethod
    def _float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _prune(self, track_id: int, frame_id: int) -> list[tuple[int, str, float]]:
        cutoff = int(frame_id) - int(self.config.window_frames)
        history = [
            item
            for item in self._history.get(int(track_id), [])
            if int(item[0]) >= cutoff
        ]
        if history:
            self._history[int(track_id)] = history
        else:
            self._history.pop(int(track_id), None)
        return history

    def observe_predictions(
        self,
        *,
        frame_id: int,
        predictions: list[dict[str, Any]],
    ) -> None:
        frame_id = int(frame_id)
        for row in predictions:
            if not isinstance(row, dict):
                continue
            if str(row.get("rejection_reason") or "") != "no_known_roster_numbers":
                continue
            try:
                track_id = int(row.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if track_id < 0:
                continue

            number = self._number(
                row.get("candidate_number")
                or row.get("raw_canonical")
                or row.get("raw_number")
            )
            posterior = self._float(
                row.get("candidate_posterior", row.get("candidate_probability", 0.0))
            )
            quality = self._float(row.get("mean_crop_quality", 0.0))
            if (
                number is None
                or posterior < float(self.config.min_candidate_posterior)
                or quality < float(self.config.min_crop_quality)
            ):
                continue

            weight = posterior * (0.70 + 0.30 * min(1.0, max(0.0, quality)))
            history = self._prune(track_id, frame_id)
            history.append((frame_id, number, weight))
            self._history[track_id] = history[-16:]
            self.accepted_raw_observations += 1

        stale_before = frame_id - 4 * int(self.config.window_frames)
        self._history = {
            track_id: [item for item in history if int(item[0]) >= stale_before]
            for track_id, history in self._history.items()
            if any(int(item[0]) >= stale_before for item in history)
        }

    def evidence(
        self,
        *,
        track_id: int,
        frame_id: int,
    ) -> tuple[str | None, float, dict[str, Any]]:
        history = self._prune(int(track_id), int(frame_id))
        if not history:
            return None, 0.0, {
                "observations": 0,
                "winner": None,
                "winner_count": 0,
                "winner_share": 0.0,
                "winner_margin": 0.0,
            }

        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for _seen_frame, number, weight in history:
            totals[number] += float(weight)
            counts[number] += 1

        ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        winner, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        total = sum(totals.values()) or 1e-12
        share = top / total
        margin = max(0.0, (top - second) / total)
        winner_count = int(counts[winner])

        stable = bool(
            winner_count >= int(self.config.min_observations)
            and share >= float(self.config.min_winner_share)
            and margin >= float(self.config.min_winner_margin)
        )
        diagnostics = {
            "observations": len(history),
            "winner": winner,
            "winner_count": winner_count,
            "winner_share": round(share, 6),
            "winner_margin": round(margin, 6),
        }
        if not stable:
            return None, 0.0, diagnostics

        confidence = min(
            0.95,
            0.62
            + 0.22 * share
            + 0.08 * min(1.0, winner_count / max(int(self.config.min_observations), 1)),
        )
        self.stable_evidence_emissions += 1
        return winner, confidence, diagnostics


__all__ = [
    "GateDRosterlessJerseyEvidenceConfigV6",
    "GateDRosterlessJerseyEvidenceV6",
]
