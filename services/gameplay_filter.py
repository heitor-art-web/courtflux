from collections import deque

import cv2
import numpy as np


class GameplayFilter:
    """Cheap broadcast prefilter with a high-recall canonical runtime policy.

    The image-only heuristics remain useful as evidence, but they are not reliable enough
    to veto detector/tracker inference across courts, lighting conditions, camera cuts,
    zooms, and painted areas. By default the filter therefore fails open for every valid
    frame and exposes its historical gameplay decision as ``likely_gameplay`` diagnostics.

    ``hard_reject=True`` retains the historical gating behaviour for explicit experiments.
    """

    def __init__(
        self,
        min_court_ratio=0.15,
        min_motion_ratio=0.003,
        max_grace_frames=15,
        processing_scale=0.25,
        hard_reject=False,
    ):
        self.min_court_ratio = min_court_ratio
        self.min_motion_ratio = min_motion_ratio
        self.max_grace_frames = max_grace_frames
        self.processing_scale = processing_scale
        self.hard_reject = bool(hard_reject)
        self.grace_counter = 0

        self.prev_gray_small = None
        self.prev_hist = None
        self.gameplay_votes = deque(maxlen=8)
        self.last_diagnostics = {}

    def is_gameplay(self, frame) -> bool:
        if frame is None:
            self.last_diagnostics = {
                "result": False,
                "should_infer": False,
                "likely_gameplay": False,
                "prefilter_mode": "high_recall_fail_open",
                "reason": "empty_frame",
            }
            return False

        if self.processing_scale != 1.0:
            small = cv2.resize(
                frame,
                (0, 0),
                fx=self.processing_scale,
                fy=self.processing_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame

        h, w = small.shape[:2]
        roi = small[int(h * 0.2):int(h * 0.8), int(w * 0.15):int(w * 0.85)]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        similarity = 1.0
        if self.prev_hist is not None:
            similarity = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)

        scene_cut_score = max(0.0, min(1.0, (0.60 - similarity) / 0.60))
        scene_cut = scene_cut_score > 0.5
        self.prev_hist = hist

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([5, 50, 80])
        upper = np.array([25, 250, 255])
        mask = cv2.inRange(hsv, lower, upper)
        court_ratio = cv2.countNonZero(mask) / mask.size
        court_score = min(1.0, court_ratio / max(self.min_court_ratio, 1e-6))

        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        motion_ratio = None
        motion_score = 1.0
        if self.prev_gray_small is not None:
            diff = cv2.absdiff(self.prev_gray_small, blur)
            _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_ratio = cv2.countNonZero(th) / th.size
            motion_score = min(1.0, motion_ratio / max(self.min_motion_ratio, 1e-6))
        self.prev_gray_small = blur

        edges = cv2.Canny(blur, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / edges.size
        edge_score = max(0.0, 1.0 - edge_ratio / 0.12)

        confidence = 0.45 * court_score + 0.35 * motion_score + 0.20 * edge_score
        confidence = min(confidence, 1.0)

        # Historical image-only vote. Keep it as evidence, but do not let it create
        # tracker/Identity gaps in the canonical runtime.
        court_vote_floor = max(0.01, self.min_court_ratio * 0.60)
        court_grace_floor = max(0.01, self.min_court_ratio * 0.40)
        has_court_evidence = court_ratio >= court_vote_floor
        vote = confidence > 0.70 and has_court_evidence and not scene_cut

        if scene_cut:
            self.gameplay_votes.clear()
            self.grace_counter = 0

        self.gameplay_votes.append(vote)
        voted_gameplay = sum(self.gameplay_votes) >= 5

        used_grace = False
        if scene_cut:
            likely_gameplay = False
        elif voted_gameplay:
            self.grace_counter = self.max_grace_frames
            likely_gameplay = True
        elif self.grace_counter > 0 and court_ratio >= court_grace_floor:
            self.grace_counter -= 1
            used_grace = True
            likely_gameplay = True
        else:
            self.grace_counter = 0
            likely_gameplay = False

        if scene_cut:
            evidence_reason = "scene_cut_reset"
        elif voted_gameplay:
            evidence_reason = "temporal_vote"
        elif used_grace:
            evidence_reason = "court_supported_grace"
        elif not has_court_evidence:
            evidence_reason = "insufficient_court_evidence"
        else:
            evidence_reason = "insufficient_temporal_votes"

        # Canonical policy: valid frames continue to detector/tracker/Identity. Richer
        # post-detector CourtState evidence decides gameplay presence and analysis quality.
        should_infer = bool(likely_gameplay) if self.hard_reject else True

        self.last_diagnostics = {
            "scene_similarity": round(float(similarity), 4),
            "scene_cut_score": round(float(scene_cut_score), 4),
            "scene_cut": bool(scene_cut),
            "court_ratio": round(float(court_ratio), 5),
            "court_score": round(float(court_score), 4),
            "court_vote_floor": round(float(court_vote_floor), 5),
            "court_grace_floor": round(float(court_grace_floor), 5),
            "has_court_evidence": bool(has_court_evidence),
            "motion_ratio": round(float(motion_ratio), 5) if motion_ratio is not None else None,
            "motion_score": round(float(motion_score), 4),
            "edge_ratio": round(float(edge_ratio), 5),
            "edge_score": round(float(edge_score), 4),
            "confidence": round(float(confidence), 4),
            "vote": bool(vote),
            "votes_true": int(sum(self.gameplay_votes)),
            "votes_window": int(len(self.gameplay_votes)),
            "used_grace": bool(used_grace),
            "likely_gameplay": bool(likely_gameplay),
            "should_infer": bool(should_infer),
            "hard_reject_enabled": bool(self.hard_reject),
            "prefilter_mode": (
                "legacy_hard_reject" if self.hard_reject else "high_recall_fail_open"
            ),
            "evidence_reason": evidence_reason,
            "reason": (
                evidence_reason
                if self.hard_reject
                else (
                    "prefilter_likely_gameplay"
                    if likely_gameplay
                    else "prefilter_uncertain_fail_open"
                )
            ),
            "result": bool(should_infer),
        }
        return should_infer
