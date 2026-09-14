from __future__ import annotations

import os
from time import perf_counter
from typing import Any

from services.vision.ball_hypothesis_tracker import OCSortBallHypothesisTracker
from services.vision.court_state_v1 import CourtStateEstimatorV1
from services.vision.p0_association_detector import PerformanceDetector as P0PerformanceDetector
from services.vision.smart_p0_detector import PerformanceDetector as SmartPerformanceDetector


class PerformanceDetector(SmartPerformanceDetector):
    """Smart P0 detector with optional OC-SORT hypothesis/identity layer.

    Default behaviour is unchanged. Set ``BASKETBALL_BALL_HYPOTHESIS_TRACKER=1`` to
    run OC-SORT+CIoU in parallel and expose lifecycle/identity diagnostics. Set
    ``BASKETBALL_BALL_HYPOTHESIS_OUTPUT=1`` only for an explicit experimental run; in
    that mode concrete ball output is emitted exclusively from an observed TRACKED
    hypothesis approved by the frozen temporal seed model. LOST/TENTATIVE hypotheses
    are never emitted.

    Player and referee detections share YOLO/BoT-SORT inference, but referees are parsed
    into ``FrameState.referees`` and never enter PlayerState/team/jersey identity.

    Before team/Jersey evidence is accumulated, CourtState evaluates the whole frame and
    per-entity court eligibility. Identity evidence is accumulated only on analysis-
    eligible frames and never from explicit ``active_on_court=False`` detections.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.referee_conf_thresh = self._env_conf("BASKETBALL_REFEREE_CONF", 0.50)
        self.identity_court_estimator = CourtStateEstimatorV1()
        self.team_stable_min_observations = self._env_int(
            "BASKETBALL_TEAM_STABLE_MIN_OBSERVATIONS",
            6,
            1,
            120,
        )
        self.team_stable_cadence_frames = self._env_int(
            "BASKETBALL_TEAM_STABLE_CADENCE_FRAMES",
            5,
            1,
            120,
        )
        self.team_stable_confidence = self._env_float(
            "BASKETBALL_TEAM_STABLE_CONFIDENCE",
            0.60,
            0.0,
            1.0,
        )
        self.team_stable_margin = self._env_float(
            "BASKETBALL_TEAM_STABLE_MARGIN",
            0.02,
            0.0,
            1.0,
        )
        self.ball_hypothesis_enabled = self._env_bool(
            "BASKETBALL_BALL_HYPOTHESIS_TRACKER",
            False,
        )
        self.ball_hypothesis_output_enabled = self._env_bool(
            "BASKETBALL_BALL_HYPOTHESIS_OUTPUT",
            False,
        )
        identity_model_path = os.getenv("BASKETBALL_BALL_IDENTITY_MODEL", "").strip()
        frame_rate = self._env_float(
            "BASKETBALL_BALL_HYPOTHESIS_FPS",
            30.0,
            1.0,
            240.0,
        )
        self.ball_hypothesis_tracker: OCSortBallHypothesisTracker | None = None
        if self.ball_hypothesis_enabled:
            self.ball_hypothesis_tracker = OCSortBallHypothesisTracker(
                frame_rate=frame_rate,
                identity_model_path=identity_model_path or None,
            )
            status = (
                "available"
                if self.ball_hypothesis_tracker.available
                else f"unavailable:{self.ball_hypothesis_tracker.unavailable_reason}"
            )
            model_status = (
                self.ball_hypothesis_tracker.identity_model.version
                if self.ball_hypothesis_tracker.identity_model is not None
                else "none"
            )
            print(
                "🏀 OC-SORT hypothesis tracker: on "
                f"status={status} identity_model={model_status} "
                f"output={'on' if self.ball_hypothesis_output_enabled else 'diagnostic-only'}",
                flush=True,
            )
        else:
            print("🏀 OC-SORT hypothesis tracker: off", flush=True)
        print(
            f"🟨 Referee tracking: on conf={self.referee_conf_thresh:.2f} "
            "(separate from PlayerState/team/jersey)",
            flush=True,
        )
        print(
            "👕 Team temporal cadence: "
            f"stable_after={self.team_stable_min_observations}obs "
            f"conf>={self.team_stable_confidence:.2f} "
            f"margin>={self.team_stable_margin:.3f} "
            f"rescore_every={self.team_stable_cadence_frames}f",
            flush=True,
        )

    @staticmethod
    def _is_referee_class(name: str) -> bool:
        value = str(name).lower().replace("-", "_").replace(" ", "_")
        return "referee" in value or value in {"ref", "official", "officials"}

    def _tracked_entity_class_ids(self) -> list[int]:
        names = getattr(self.model, "names", {}) or {}
        items = names.items() if isinstance(names, dict) else enumerate(names)
        class_ids = []
        for class_id, name in items:
            normalized = str(name).lower()
            if self._is_player_class(normalized) or self._is_referee_class(normalized):
                class_ids.append(int(class_id))
        return sorted(set(class_ids))

    def _track(self, frame: Any):
        """Track players and referees while keeping split ball detection independent."""
        if not bool(getattr(self, "split_ball_detection", False)):
            return super()._track(frame)

        kwargs: dict[str, Any] = {
            "persist": True,
            "tracker": self.tracker,
            "conf": self.player_conf_thresh,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "device": self.device,
            "verbose": False,
        }
        class_ids = self._tracked_entity_class_ids()
        if class_ids:
            kwargs["classes"] = class_ids
        return self.model.track(frame, **kwargs)

    def _parse(
        self,
        frame_state: Any,
        tracked_results: Any,
        ball_results: Any = None,
    ) -> None:
        """Preserve split player/ball parsing and expose referees independently."""
        super()._parse(frame_state, tracked_results, ball_results)
        referees: list[dict[str, Any]] = []
        raw_referees = 0
        if tracked_results and tracked_results[0].boxes is not None:
            names = getattr(self.model, "names", {}) or {}
            for box in tracked_results[0].boxes:
                cls_id = int(box.cls[0])
                if isinstance(names, dict):
                    cls_name = str(names.get(cls_id, cls_id)).lower()
                else:
                    cls_name = str(
                        names[cls_id] if 0 <= cls_id < len(names) else cls_id
                    ).lower()
                if not self._is_referee_class(cls_name):
                    continue
                raw_referees += 1
                conf = float(box.conf[0])
                if conf < self.referee_conf_thresh:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x2 <= x1 or y2 <= y1:
                    continue
                track_id = int(box.id[0]) if box.id is not None else -1
                referees.append(
                    {
                        "track_id": track_id,
                        "box": [x1, y1, x2, y2],
                        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        "confidence": conf,
                        "class_name": cls_name,
                    }
                )

        frame_state.referees = referees
        diag = frame_state.diagnostics.setdefault("detector", {})
        diag["raw_referee_candidates"] = int(raw_referees)
        diag["accepted_referee_candidates"] = len(referees)
        diag["referee_conf"] = round(float(self.referee_conf_thresh), 6)
        diag["referee_predictions"] = [
            {
                "track_id": int(referee.get("track_id", -1)),
                "box": list(referee.get("box", [])),
                "confidence": round(float(referee.get("confidence", 0.0)), 6),
            }
            for referee in referees
        ]

    def _prepare_identity_court_gate(self, frame_state: Any) -> None:
        """Build CourtState before identity evidence can mutate track profiles."""
        diag = frame_state.diagnostics.setdefault("detector", {})
        if bool(diag.get("identity_court_gate_prepared", False)):
            return

        court_state = self.identity_court_estimator.update(
            frame_state,
            gameplay_diagnostics=(
                frame_state.diagnostics.get("gameplay_filter", {}) or {}
            ),
        )
        frame_state.court_state = court_state
        frame_state.diagnostics["court_state"] = court_state.as_dict()

        players = [
            player
            for player in list(getattr(frame_state, "players", []) or [])
            if isinstance(player, dict)
        ]
        explicit_off_court = sum(
            1 for player in players if player.get("active_on_court") is False
        )
        eligible_entities = sum(
            1 for player in players if player.get("active_on_court") is not False
        )
        eligible_for_identity = (
            eligible_entities if court_state.analysis_eligible else 0
        )

        diag["identity_court_gate_prepared"] = True
        diag["identity_frame_analysis_eligible"] = bool(court_state.analysis_eligible)
        diag["identity_frame_eligibility_reason"] = court_state.eligibility_reason
        diag["identity_players_total"] = len(players)
        diag["identity_players_explicit_off_court"] = explicit_off_court
        diag["identity_players_eligible"] = eligible_for_identity
        diag["identity_court_gate_policy"] = (
            "require_analysis_eligible_frame_and_exclude_explicit_off_court"
        )

    @staticmethod
    def _identity_eligible_players(frame_state: Any) -> list[dict[str, Any]]:
        court_state = getattr(frame_state, "court_state", None)
        if court_state is not None and not bool(
            getattr(court_state, "analysis_eligible", False)
        ):
            return []
        return [
            player
            for player in list(getattr(frame_state, "players", []) or [])
            if isinstance(player, dict) and player.get("active_on_court") is not False
        ]

    def _team_score_due(self, profile: dict[str, Any], frame_id: int) -> bool:
        if profile.get("identity_locked") and profile.get("identity_team"):
            return False
        observations = int(profile.get("team_observations", 0) or 0)
        stable = bool(
            profile.get("team")
            and observations >= self.team_stable_min_observations
            and float(profile.get("team_confidence", 0.0) or 0.0) >= self.team_stable_confidence
            and float(profile.get("team_margin", 0.0) or 0.0) >= self.team_stable_margin
        )
        if not stable:
            return True
        last_frame = int(profile.get("last_team_observation_frame", -10**9) or -10**9)
        return frame_id - last_frame >= self.team_stable_cadence_frames

    @staticmethod
    def _reuse_team_profile(player: dict[str, Any], profile: dict[str, Any]) -> None:
        if profile.get("identity_locked") and profile.get("identity_team"):
            profile["team"] = profile["identity_team"]
        player["team"] = profile.get("team")
        player["team_confidence"] = round(float(profile.get("team_confidence", 0.0)), 4)
        player["team_margin"] = round(float(profile.get("team_margin", 0.0)), 6)
        player["shirt_num"] = profile.get("shirt_num")
        player["player_name"] = profile.get("player_name")

    def _assign_teams(self, frame_state: Any):
        """Score team evidence only when needed; never charge Jersey buffering to team time."""
        self._prepare_identity_court_gate(frame_state)
        original_players = frame_state.players
        eligible_players = self._identity_eligible_players(frame_state)
        frame_id = int(getattr(frame_state, "frame_id", 0))
        players_to_score: list[dict[str, Any]] = []
        reused_players: list[dict[str, Any]] = []

        for player in eligible_players:
            track_id = int(player.get("track_id", -1))
            if track_id < 0:
                players_to_score.append(player)
                continue
            profile = self._profile(track_id)
            if self._team_score_due(profile, frame_id):
                players_to_score.append(player)
            else:
                self._reuse_team_profile(player, profile)
                reused_players.append(player)

        # Base _assign_teams historically also collected Jersey crops. Disable only that
        # side effect here; _jerseys() below owns Jersey buffering and its timing.
        previous_enable_jersey = self.enable_jersey
        frame_state.players = players_to_score
        self.enable_jersey = False
        try:
            result = super()._assign_teams(frame_state)
        finally:
            self.enable_jersey = previous_enable_jersey
            frame_state.players = original_players

        for player in players_to_score:
            track_id = int(player.get("track_id", -1))
            if track_id >= 0:
                self._profile(track_id)["last_team_observation_frame"] = frame_id

        diag = frame_state.diagnostics.setdefault("detector", {})
        predictions = list(diag.get("team_predictions", []) or [])
        for item in predictions:
            if isinstance(item, dict):
                item["rescored"] = True
        predictions.extend(
            {
                "track_id": int(player.get("track_id", -1)),
                "team": player.get("team"),
                "confidence": player.get("team_confidence"),
                "margin": player.get("team_margin"),
                "scores": {},
                "rescored": False,
                "source": "persistent_team_profile",
            }
            for player in reused_players
        )
        diag["team_predictions"] = predictions
        diag["team_runtime"] = {
            "eligible_players": len(eligible_players),
            "rescored_players": len(players_to_score),
            "reused_players": len(reused_players),
            "stable_min_observations": self.team_stable_min_observations,
            "stable_confidence": round(self.team_stable_confidence, 6),
            "stable_margin": round(self.team_stable_margin, 6),
            "stable_cadence_frames": self.team_stable_cadence_frames,
        }
        return result

    def _jerseys(self, frame_state: Any):
        """Keep Jersey ON, but own its crop-buffer work inside Jersey timing."""
        self._prepare_identity_court_gate(frame_state)
        original_players = frame_state.players
        eligible_players = self._identity_eligible_players(frame_state)
        frame_state.players = eligible_players
        buffer_started = perf_counter()
        buffer_observations = 0
        try:
            if self.enable_jersey and hasattr(self.visual, "observe_track"):
                frame = frame_state.frame
                for player in eligible_players:
                    track_id = int(player.get("track_id", -1))
                    if track_id < 0:
                        continue
                    profile = self._profile(track_id)
                    if profile.get("identity_locked"):
                        continue
                    x1, y1, x2, y2 = player["box"]
                    crop = frame[
                        max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2),
                    ]
                    self.visual.observe_track(crop, track_id)
                    buffer_observations += 1
            buffer_ms = (perf_counter() - buffer_started) * 1000.0
            inference_started = perf_counter()
            result = super()._jerseys(frame_state)
            inference_ms = (perf_counter() - inference_started) * 1000.0
            diag = frame_state.diagnostics.setdefault("detector", {})
            diag["jersey_buffer_ms"] = round(buffer_ms, 4)
            diag["jersey_buffer_observations"] = int(buffer_observations)
            diag["jersey_inference_ms"] = round(inference_ms, 4)
            return result
        finally:
            frame_state.players = original_players

    def _tracker_config(self) -> dict[str, Any]:
        config = super()._tracker_config()
        tracker = getattr(self, "ball_hypothesis_tracker", None)
        config["hypothesis_tracker"] = {
            "enabled": bool(getattr(self, "ball_hypothesis_enabled", False)),
            "output_enabled": bool(
                getattr(self, "ball_hypothesis_output_enabled", False)
            ),
            "rollback_env": "BASKETBALL_BALL_HYPOTHESIS_TRACKER=0",
            "output_rollback_env": "BASKETBALL_BALL_HYPOTHESIS_OUTPUT=0",
            "available": bool(tracker.available) if tracker is not None else False,
            "profile": tracker.profile if tracker is not None else None,
            "identity_model": (
                tracker._identity_model_diagnostics()
                if tracker is not None
                else {"loaded": False}
            ),
            "safety": {
                "lost_can_emit_concrete_output": False,
                "tentative_can_emit_concrete_output": False,
                "hard_reacquisition_required": False,
            },
        }
        return config

    def _hypothesis_update(
        self,
        candidates: list[dict[str, Any]],
        frame_state: Any,
    ) -> dict[str, Any]:
        tracker = getattr(self, "ball_hypothesis_tracker", None)
        if not bool(getattr(self, "ball_hypothesis_enabled", False)) or tracker is None:
            return {
                "enabled": False,
                "available": False,
                "hypotheses": [],
                "selected_seed": None,
            }
        return tracker.update(
            candidates,
            frame_id=int(getattr(frame_state, "frame_id", 0)),
        )

    def _seed_association(
        self,
        *,
        enhanced: list[dict[str, Any]],
        association_candidates: list[dict[str, Any]],
        hypothesis_diag: dict[str, Any],
        smart_diag: dict[str, Any],
        frame_state: Any,
    ):
        traces = self._candidate_trace(association_candidates, frame_state)
        selected_seed = hypothesis_diag.get("selected_seed")
        selected_index = None
        if isinstance(selected_seed, dict):
            try:
                candidate_index = int(selected_seed.get("candidate_index"))
                if 0 <= candidate_index < len(enhanced):
                    selected_index = candidate_index
            except (TypeError, ValueError):
                selected_index = None

        ball = None
        source = "missing"
        score = None
        track_state = "ABSTAIN"
        if selected_index is not None:
            selected = dict(enhanced[selected_index])
            candidate_source = str(selected.get("source", "full_frame"))
            selected["candidate_source"] = candidate_source
            selected["association_confidence"] = float(
                association_candidates[selected_index].get("confidence", 0.0)
            )
            selected["source"] = "ocsort_identity_seed"
            selected["interpolated"] = False
            ball = selected
            source = "ocsort_identity_seed"
            score = float(selected_seed.get("identity_probability", 0.0))
            track_state = "TRACKED"

        self._mark_candidate_decisions(traces, selected_index)
        association = {
            "source": source,
            "selected_index": selected_index,
            "score": None if score is None else round(score, 6),
            "distance_norm": None,
            "gate_norm": None,
            "rejected_candidates": 0,
            "hold_age_frames": 0,
            "track_state": track_state,
            "reacquired": False,
            "challenger_streak": 0,
            "challenger_confidence": None,
            "candidate_count": len(enhanced),
            "candidates": traces,
            "config": self._tracker_config(),
            "ball_hypotheses": hypothesis_diag,
            "smart_pipeline": smart_diag,
            "raw_candidate_count": int(smart_diag.get("raw_candidate_count", 0)),
            "enhanced_candidate_count": len(enhanced),
            **self._trajectory_payload(ball),
        }
        return ball, association

    def _select_ball(self, candidates: list[dict[str, Any]], frame_state: Any):
        # Exact legacy path when the new layer is disabled.
        if not bool(getattr(self, "ball_hypothesis_enabled", False)):
            return super()._select_ball(candidates, frame_state)

        pipeline = getattr(self, "smart_ball_pipeline", None)
        if pipeline is None:
            hypothesis_diag = self._hypothesis_update(candidates, frame_state)
            if bool(getattr(self, "ball_hypothesis_output_enabled", False)):
                smart_diag = {
                    "enabled": False,
                    "raw_candidate_count": len(candidates),
                }
                return self._seed_association(
                    enhanced=candidates,
                    association_candidates=candidates,
                    hypothesis_diag=hypothesis_diag,
                    smart_diag=smart_diag,
                    frame_state=frame_state,
                )
            ball, association = P0PerformanceDetector._select_ball(
                self,
                candidates,
                frame_state,
            )
            association["ball_hypotheses"] = hypothesis_diag
            return ball, association

        frame_id = int(getattr(frame_state, "frame_id", 0))
        enhanced, smart_diag = pipeline.enhance(
            frame_state.frame,
            candidates,
            frame_id=frame_id,
            tracker=self.ball_tracker,
        )
        from services.vision.ball_candidate_filter import filter_smart_candidates

        enhanced, filter_diag = filter_smart_candidates(
            enhanced,
            emit_refinements=self.smart_emit_refinements,
            zoom_min_geometry=self.smart_zoom_min_geometry,
            temporal_rescue_min_score=self.smart_temporal_rescue_min_score,
        )
        smart_diag.update(filter_diag)
        smart_diag["post_filter_candidate_count"] = len(enhanced)

        enhanced, restored = self._preserve_full_frame_candidates(
            candidates,
            enhanced,
            frame_state=frame_state,
        )
        smart_diag["full_frame_candidates_restored"] = int(restored)
        smart_diag["output_candidate_count"] = len(enhanced)
        smart_diag["raw_candidate_count"] = len(candidates)

        association_candidates = self._association_candidates(enhanced)
        hypothesis_diag = self._hypothesis_update(enhanced, frame_state)

        if bool(getattr(self, "ball_hypothesis_output_enabled", False)):
            return self._seed_association(
                enhanced=enhanced,
                association_candidates=association_candidates,
                hypothesis_diag=hypothesis_diag,
                smart_diag=smart_diag,
                frame_state=frame_state,
            )

        # Diagnostic-only mode: preserve the current Smart-v1 selection exactly.
        ball, association = P0PerformanceDetector._select_ball(
            self,
            association_candidates,
            frame_state,
        )
        selected_index = association.get("selected_index")
        if (
            selected_index is not None
            and 0 <= int(selected_index) < len(enhanced)
            and ball is not None
            and not bool(ball.get("interpolated", False))
        ):
            selected = dict(enhanced[int(selected_index)])
            candidate_source = str(selected.get("source", "full_frame"))
            selected["candidate_source"] = candidate_source
            selected["association_confidence"] = float(
                association_candidates[int(selected_index)].get("confidence", 0.0)
            )
            selected["source"] = str(association.get("source") or "detector")
            selected["interpolated"] = False
            ball = selected

        association["smart_pipeline"] = smart_diag
        association["raw_candidate_count"] = len(candidates)
        association["enhanced_candidate_count"] = len(enhanced)
        association["candidate_count"] = len(enhanced)
        association["ball_hypotheses"] = hypothesis_diag
        return ball, association
