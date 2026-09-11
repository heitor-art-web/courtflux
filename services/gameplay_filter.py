from collections import deque
import cv2
import numpy as np


class GameplayFilter:

    def __init__(
        self,
        min_court_ratio=0.15,
        min_motion_ratio=0.003,
        max_grace_frames=15,
        processing_scale=0.25,
    ):
        self.min_court_ratio = min_court_ratio
        self.min_motion_ratio = min_motion_ratio
        self.max_grace_frames = max_grace_frames
        self.processing_scale = processing_scale
        self.grace_counter = 0

        self.prev_gray_small = None
        self.prev_hist = None
        self.gameplay_votes = deque(maxlen=8)
        self.last_diagnostics = {}

    def is_gameplay(self, frame) -> bool:
        if frame is None:
            self.last_diagnostics = {"result": False, "reason": "empty_frame"}
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

        # Gameplay needs some actual court evidence. This prevents close-ups, crowd shots,
        # benches, score graphics, and other moving broadcast content from winning on motion alone.
        court_vote_floor = max(0.01, self.min_court_ratio * 0.60)
        court_grace_floor = max(0.01, self.min_court_ratio * 0.40)
        has_court_evidence = court_ratio >= court_vote_floor
        vote = confidence > 0.70 and has_court_evidence and not scene_cut

        # A scene cut is a boundary, not positive gameplay evidence. Reset temporal state so
        # grace from the previous camera shot cannot bleed into replays/close-ups/graphics.
        if scene_cut:
            self.gameplay_votes.clear()
            self.grace_counter = 0

        self.gameplay_votes.append(vote)
        voted_gameplay = sum(self.gameplay_votes) >= 5

        used_grace = False
        if scene_cut:
            result = False
        elif voted_gameplay:
            self.grace_counter = self.max_grace_frames
            result = True
        elif self.grace_counter > 0 and court_ratio >= court_grace_floor:
            self.grace_counter -= 1
            used_grace = True
            result = True
        else:
            self.grace_counter = 0
            result = False

        if scene_cut:
            reason = "scene_cut_reset"
        elif voted_gameplay:
            reason = "temporal_vote"
        elif used_grace:
            reason = "court_supported_grace"
        elif not has_court_evidence:
            reason = "insufficient_court_evidence"
        else:
            reason = "insufficient_temporal_votes"

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
            "reason": reason,
            "result": bool(result),
        }
        return result
