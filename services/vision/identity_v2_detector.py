from __future__ import annotations

import math
from time import perf_counter
from typing import Any

from services.vision.hypothesis_p0_detector import PerformanceDetector as LegacyPerformanceDetector


class PerformanceDetector(LegacyPerformanceDetector):
    """Reversible P1 Identity v2 layer; detector/tracker/ball behavior is inherited unchanged."""

    _TEAM_VECTOR_KEYS = ("Black", "Grey", "White", "Blue", "Navy", "Red", "Green", "Yellow")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.team_evidence_window_frames = self._env_int("BASKETBALL_IDENTITY_TEAM_WINDOW_FRAMES", 120, 24, 600)
        self.team_evidence_half_life_frames = self._env_float("BASKETBALL_IDENTITY_TEAM_HALF_LIFE_FRAMES", 45.0, 8.0, 300.0)
        self.team_prototype_min_observations = self._env_int("BASKETBALL_IDENTITY_TEAM_PROTOTYPE_MIN_OBS", 4, 2, 40)
        self.team_prototype_alpha = self._env_float("BASKETBALL_IDENTITY_TEAM_PROTOTYPE_ALPHA", 0.18, 0.02, 0.80)
        self.team_reversal_min_observations = self._env_int("BASKETBALL_IDENTITY_TEAM_REVERSAL_MIN_OBS", 3, 2, 20)
        self.jersey_evidence_window_frames = self._env_int("BASKETBALL_IDENTITY_JERSEY_WINDOW_FRAMES", 180, 36, 900)
        self.jersey_evidence_half_life_frames = self._env_float("BASKETBALL_IDENTITY_JERSEY_HALF_LIFE_FRAMES", 72.0, 12.0, 450.0)
        # Jersey v3 uses joint digit-pair posterior probabilities. Two minimum-strength
        # accepted observations (posterior=0.18, team confidence~=0.58) separated by
        # the normal 24-frame cadence contribute about 0.142 after temporal decay.
        # Keeping the old 0.18 lock floor therefore made the second-stage gate stricter
        # than its own accepted inputs. Retain the two-observation/share/margin gates,
        # but calibrate absolute evidence to the v3 probability scale.
        self.identity_lock_min_evidence = self._env_float(
            "BASKETBALL_IDENTITY_LOCK_MIN_EVIDENCE", 0.14, 0.05, 1.0
        )
        self.identity_locked_jersey_cadence_multiplier = self._env_int(
            "BASKETBALL_IDENTITY_LOCKED_JERSEY_CADENCE_MULTIPLIER", 3, 1, 12
        )
        self.team_prototypes: dict[str, dict[str, Any]] = {}
        print(
            "🪪 Identity v2: on "
            f"team_window={self.team_evidence_window_frames}f "
            f"jersey_window={self.jersey_evidence_window_frames}f "
            f"lock_evidence={self.identity_lock_min_evidence:.2f} "
            "lock=reversible roster=validator-only",
            flush=True,
        )

    def _profile(self, track_id: int) -> dict[str, Any]:
        profile = super()._profile(track_id)
        defaults = {
            "team_evidence": [],
            "jersey_evidence": [],
            "team_corrections": 0,
            "team_reversal_candidate": None,
            "team_reversal_streak": 0,
            "identity_revision": 0,
            "identity_lock_frame": None,
            "identity_status": "unknown",
            "identity_conflict_streak": 0,
        }
        for key, value in defaults.items():
            profile.setdefault(key, value.copy() if isinstance(value, list) else value)
        return profile

    @staticmethod
    def _normalized(scores: dict[str, float], names: list[str]) -> dict[str, float]:
        values = {name: max(0.0, float(scores.get(name, 0.0))) for name in names}
        total = sum(values.values())
        return ({name: value / total for name, value in values.items()} if total > 1e-12 else {name: 0.0 for name in names})

    @classmethod
    def _coverage_vector(cls, result: dict[str, Any]) -> list[float]:
        coverage = dict(result.get("color_coverage", {}) or {})
        return [max(0.0, float(coverage.get(key, 0.0))) for key in cls._TEAM_VECTOR_KEYS]

    def _prototype_distribution(self, vector: list[float], teams: list[str]) -> tuple[dict[str, float], bool]:
        ready = bool(teams) and all(
            int((self.team_prototypes.get(team) or {}).get("observations", 0)) >= self.team_prototype_min_observations
            for team in teams
        )
        if not ready:
            return {team: 0.0 for team in teams}, False
        similarities = {}
        for team in teams:
            prototype = list(self.team_prototypes[team]["vector"])
            distance = sum(abs(a - b) for a, b in zip(vector, prototype)) / max(len(vector), 1)
            similarities[team] = math.exp(-distance / 0.10)
        return self._normalized(similarities, teams), True

    def _update_team_prototype(self, team: str | None, vector: list[float], result: dict[str, Any]) -> None:
        if (
            not team
            or not bool(result.get("evidence_accepted"))
            or str(result.get("best_team") or "") != str(team)
            or float(result.get("best_score", 0.0) or 0.0) < 0.045
            or float(result.get("margin", 0.0) or 0.0) < 0.018
        ):
            return
        current = self.team_prototypes.get(team)
        if not current:
            self.team_prototypes[team] = {"vector": list(vector), "observations": 1}
            return
        alpha = self.team_prototype_alpha
        current["vector"] = [
            (1.0 - alpha) * old + alpha * new for old, new in zip(current["vector"], vector)
        ]
        current["observations"] = int(current.get("observations", 0)) + 1

    def _team_distribution(self, result: dict[str, Any]) -> tuple[dict[str, float], bool, dict[str, Any]]:
        teams = [str(team) for team in getattr(self.visual, "rosters", {})]
        static = self._normalized(dict(result.get("raw_scores", {}) or {}), teams)
        vector = self._coverage_vector(result)
        prototype, prototype_ready = self._prototype_distribution(vector, teams)
        fused = (
            self._normalized({team: 0.78 * static[team] + 0.22 * prototype[team] for team in teams}, teams)
            if prototype_ready
            else static
        )
        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        best_team, best = ordered[0] if ordered else (None, 0.0)
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = max(0.0, best - second)
        accepted = bool(result.get("evidence_accepted")) or bool(prototype_ready and best >= 0.64 and margin >= 0.18)
        return fused, accepted, {
            "prototype_ready": prototype_ready,
            "prototype_scores": {name: round(value, 6) for name, value in prototype.items()},
            "fused_scores": {name: round(value, 6) for name, value in fused.items()},
            "fused_best_team": best_team,
            "fused_confidence": round(best, 6),
            "fused_margin": round(margin, 6),
        }

    def _aggregate_team_evidence(self, profile: dict[str, Any], frame_id: int) -> tuple[str | None, float, float, int]:
        evidence = [
            item for item in list(profile.get("team_evidence", []) or [])
            if frame_id - int(item.get("frame_id", frame_id)) <= self.team_evidence_window_frames
        ][-48:]
        profile["team_evidence"] = evidence
        totals: dict[str, float] = {}
        for item in evidence:
            age = max(0, frame_id - int(item.get("frame_id", frame_id)))
            weight = (0.5 ** (age / max(self.team_evidence_half_life_frames, 1.0))) * max(0.05, float(item.get("weight", 1.0)))
            for team, value in dict(item.get("scores", {}) or {}).items():
                totals[str(team)] = totals.get(str(team), 0.0) + weight * max(0.0, float(value))
        normalized = self._normalized(totals, list(totals))
        ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        if not ordered:
            return None, 0.0, 0.0, len(evidence)
        best_team, confidence = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        return best_team, confidence, max(0.0, confidence - second), len(evidence)

    def _observe_team_reversal_candidate(
        self,
        profile: dict[str, Any],
        *,
        candidate: str | None,
        confidence: float,
        margin: float,
        accepted: bool,
    ) -> bool:
        """Confirm a stale-profile reversal only from consecutive fresh accepted rescores.

        The confirmation thresholds intentionally match the existing reversal thresholds.
        Reused profile frames never advance this streak because this method is called only
        from the rescore path. Once confirmed, callers may discard older contradictory
        evidence so the decay window cannot outvote three consecutive strong fresh crops.
        """
        previous = profile.get("team")
        strong_conflict = bool(
            accepted
            and previous
            and candidate
            and candidate != previous
            and confidence >= 0.64
            and margin >= 0.18
        )
        if not strong_conflict:
            profile["team_reversal_candidate"] = None
            profile["team_reversal_streak"] = 0
            return False

        if str(profile.get("team_reversal_candidate") or "") == str(candidate):
            streak = int(profile.get("team_reversal_streak", 0) or 0) + 1
        else:
            profile["team_reversal_candidate"] = candidate
            streak = 1
        profile["team_reversal_streak"] = streak
        return streak >= self.team_reversal_min_observations

    def _clear_identity_number(self, profile: dict[str, Any], *, reason: str) -> None:
        had_identity = bool(profile.get("shirt_num") or profile.get("identity_locked"))
        profile.update(
            shirt_num=None,
            player_name=None,
            identity_team=None,
            identity_locked=False,
            identity_lock_frame=None,
            identity_status="unknown",
            identity_conflict_streak=0,
            jersey_evidence=[],
            identity_last_change_reason=reason,
        )
        if had_identity:
            profile["identity_revision"] = int(profile.get("identity_revision", 0)) + 1

    def _apply_team_resolution(
        self,
        profile: dict[str, Any],
        *,
        best_team: str | None,
        confidence: float,
        margin: float,
        observations: int,
    ) -> None:
        previous = profile.get("team")
        accepted = bool(best_team and confidence >= 0.58 and margin >= 0.10)
        if previous and best_team and best_team != previous:
            accepted = bool(
                observations >= self.team_reversal_min_observations and confidence >= 0.64 and margin >= 0.18
            )
        profile["team_confidence"] = confidence
        profile["team_margin"] = margin
        if not accepted:
            return
        if previous and best_team != previous:
            profile["team_corrections"] = int(profile.get("team_corrections", 0)) + 1
            self._clear_identity_number(profile, reason="team_reversal")
        profile["team"] = best_team

    def _team_score_due(self, profile: dict[str, Any], frame_id: int) -> bool:
        pending_reversal = bool(profile.get("team_reversal_candidate")) and int(
            profile.get("team_reversal_streak", 0) or 0
        ) > 0
        if pending_reversal:
            return True
        stable = bool(
            profile.get("team")
            and int(profile.get("team_observations", 0) or 0) >= self.team_stable_min_observations
            and float(profile.get("team_confidence", 0.0) or 0.0) >= self.team_stable_confidence
            and float(profile.get("team_margin", 0.0) or 0.0) >= self.team_stable_margin
        )
        if not stable:
            return True
        return frame_id - int(profile.get("last_team_observation_frame", -10**9) or -10**9) >= self.team_stable_cadence_frames

    @staticmethod
    def _reuse_team_profile(player: dict[str, Any], profile: dict[str, Any]) -> None:
        player.update(
            team=profile.get("team"),
            team_confidence=round(float(profile.get("team_confidence", 0.0)), 4),
            team_margin=round(float(profile.get("team_margin", 0.0)), 6),
            shirt_num=profile.get("shirt_num"),
            player_name=profile.get("player_name"),
            identity_locked=bool(profile.get("identity_locked")),
            identity_revision=int(profile.get("identity_revision", 0)),
        )

    @staticmethod
    def _team_eligible_players(frame_state: Any) -> list[dict[str, Any]]:
        """Use gameplay presence, but require positive on-court evidence in limited frames."""
        court_state = getattr(frame_state, "court_state", None)
        gameplay_present = None
        analysis_eligible = True
        if court_state is not None:
            gameplay_present = getattr(court_state, "gameplay_present", None)
            analysis_eligible = bool(getattr(court_state, "analysis_eligible", False))
            if gameplay_present is not None:
                if not bool(gameplay_present):
                    return []
            elif not analysis_eligible:
                return []

        limited_gameplay = bool(gameplay_present is True and not analysis_eligible)
        return [
            player
            for player in list(getattr(frame_state, "players", []) or [])
            if isinstance(player, dict)
            and (
                player.get("active_on_court") is True
                if limited_gameplay
                else player.get("active_on_court") is not False
            )
        ]

    def _assign_teams(self, frame_state: Any):
        self._prepare_identity_court_gate(frame_state)
        frame, frame_id = frame_state.frame, int(getattr(frame_state, "frame_id", 0))
        eligible = self._team_eligible_players(frame_state)
        predictions, rescored, reused = [], 0, 0
        for player in eligible:
            x1, y1, x2, y2 = player["box"]
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            track_id = int(player.get("track_id", -1))
            if track_id < 0:
                result = self.visual.score_teams(crop) if hasattr(self.visual, "score_teams") else {}
                player["team"] = result.get("best_team") if result else self.visual.get_team_color(crop)
                predictions.append({"track_id": track_id, "team": player.get("team"), "rescored": True, "source": "untracked_current_crop"})
                continue

            profile = self._profile(track_id)
            if not self._team_score_due(profile, frame_id):
                self._reuse_team_profile(player, profile)
                reused += 1
                predictions.append({
                    "track_id": track_id, "team": player.get("team"), "confidence": player.get("team_confidence"),
                    "margin": player.get("team_margin"), "rescored": False, "source": "identity_v2_team_profile",
                })
                continue

            rescored += 1
            result = self.visual.score_teams(crop) if hasattr(self.visual, "score_teams") else {}
            distribution, accepted, adaptive = self._team_distribution(result)
            current_best: str | None = None
            recent_reversal_confirmed = False
            if accepted and distribution:
                ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
                current_best = ordered[0][0]
                current_confidence = ordered[0][1]
                current_margin = max(0.0, current_confidence - (ordered[1][1] if len(ordered) > 1 else 0.0))
                selectivity = max(0.05, current_margin)
                profile["team_evidence"].append({
                    "frame_id": frame_id,
                    "scores": distribution,
                    "weight": 0.55 + min(0.45, 2.0 * selectivity),
                })
                recent_reversal_confirmed = self._observe_team_reversal_candidate(
                    profile,
                    candidate=current_best,
                    confidence=current_confidence,
                    margin=current_margin,
                    accepted=True,
                )
                if recent_reversal_confirmed:
                    # The last N accepted rescores are a consistent strong contradiction
                    # of the current profile. Older team evidence is now stale and would
                    # otherwise keep the profile pinned to the wrong team for the full
                    # decay window.
                    profile["team_evidence"] = list(profile["team_evidence"])[
                        -self.team_reversal_min_observations:
                    ]
                self._update_team_prototype(current_best, self._coverage_vector(result), result)
            else:
                self._observe_team_reversal_candidate(
                    profile,
                    candidate=None,
                    confidence=0.0,
                    margin=0.0,
                    accepted=False,
                )

            best_team, confidence, margin, observations = self._aggregate_team_evidence(profile, frame_id)
            profile["team_observations"] = observations
            self._apply_team_resolution(
                profile, best_team=best_team, confidence=confidence, margin=margin, observations=observations
            )
            if recent_reversal_confirmed and current_best and profile.get("team") == current_best:
                profile["team_reversal_candidate"] = None
                profile["team_reversal_streak"] = 0
            profile["last_team_observation_frame"] = frame_id
            self._reuse_team_profile(player, profile)
            predictions.append({
                "track_id": track_id,
                "team": player.get("team"),
                "confidence": player.get("team_confidence"),
                "margin": player.get("team_margin"),
                "scores": dict(result.get("scores", {}) or {}),
                "raw_scores": dict(result.get("raw_scores", {}) or {}),
                "rescored": True,
                "source": "identity_v2_decay_window",
                "team_evidence_count": observations,
                "team_corrections": int(profile.get("team_corrections", 0)),
                "recent_reversal_confirmed": recent_reversal_confirmed,
                "recent_reversal_candidate": profile.get("team_reversal_candidate"),
                "recent_reversal_streak": int(profile.get("team_reversal_streak", 0) or 0),
                **adaptive,
            })

        court_state = getattr(frame_state, "court_state", None)
        gameplay_present = (
            None if court_state is None else getattr(court_state, "gameplay_present", None)
        )
        analysis_eligible = bool(
            court_state is not None and getattr(court_state, "analysis_eligible", False)
        )
        diag = frame_state.diagnostics.setdefault("detector", {})
        diag["team_predictions"] = predictions
        diag["team_runtime"] = {
            "identity_version": "v2",
            "eligible_players": len(eligible),
            "rescored_players": rescored,
            "reused_players": reused,
            "evidence_window_frames": self.team_evidence_window_frames,
            "evidence_half_life_frames": self.team_evidence_half_life_frames,
            "prototype_min_observations": self.team_prototype_min_observations,
            "prototype_observations": {team: int(value.get("observations", 0)) for team, value in self.team_prototypes.items()},
            "team_reversal_policy": "consecutive_accepted_opposing_rescores_v1",
            "team_reversal_min_observations": self.team_reversal_min_observations,
            "identity_lock_blocks_team_rescore": False,
            "team_eligibility_policy": "gameplay_present;limited_requires_explicit_on_court;analysis_eligible_excludes_explicit_off_court",
            "gameplay_present": gameplay_present,
            "analysis_eligible": analysis_eligible,
            "limited_gameplay_team_evidence": bool(gameplay_present is True and not analysis_eligible),
            "limited_gameplay_requires_explicit_on_court": True,
        }

    def _aggregate_jersey_evidence(
        self, profile: dict[str, Any], frame_id: int
    ) -> tuple[str | None, float, float, int, float]:
        evidence = [
            item for item in list(profile.get("jersey_evidence", []) or [])
            if frame_id - int(item.get("frame_id", frame_id)) <= self.jersey_evidence_window_frames
        ][-32:]
        profile["jersey_evidence"] = evidence
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for item in evidence:
            number = str(item.get("number") or "")
            if not number:
                continue
            age = max(0, frame_id - int(item.get("frame_id", frame_id)))
            weight = (0.5 ** (age / max(self.jersey_evidence_half_life_frames, 1.0))) * max(0.01, float(item.get("weight", 0.0)))
            totals[number] = totals.get(number, 0.0) + weight
            counts[number] = counts.get(number, 0) + 1
        if not totals:
            return None, 0.0, 0.0, 0, 0.0
        ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        number, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        total = sum(totals.values()) or 1e-12
        return number, top / total, max(0.0, (top - second) / total), int(counts[number]), top

    def _lock_identity(self, profile: dict[str, Any], *, team: str, number: str, frame_id: int, changed: bool) -> None:
        if changed:
            profile["identity_revision"] = int(profile.get("identity_revision", 0)) + 1
        profile.update(
            shirt_num=str(number), identity_team=str(team), identity_locked=True,
            identity_lock_frame=frame_id, identity_status="locked_reversible", identity_conflict_streak=0,
        )
        if hasattr(self.visual, "resolve_player_name"):
            profile["player_name"] = self.visual.resolve_player_name(team, str(number))

    def _jerseys(self, frame_state: Any):
        self._prepare_identity_court_gate(frame_state)
        if not self.enable_jersey:
            return 0, 0, []
        frame_id, frame = int(getattr(frame_state, "frame_id", 0)), frame_state.frame
        eligible = self._identity_eligible_players(frame_state)

        buffer_started, buffered = perf_counter(), 0
        if hasattr(self.visual, "observe_track"):
            for player in eligible:
                track_id = int(player.get("track_id", -1))
                if track_id < 0:
                    continue
                x1, y1, x2, y2 = player["box"]
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                try:
                    self.visual.observe_track(crop, track_id, frame_id=frame_id)
                except TypeError:
                    self.visual.observe_track(crop, track_id)
                buffered += 1
        buffer_ms = (perf_counter() - buffer_started) * 1000.0

        inference_started, attempts, locks, predictions = perf_counter(), 0, 0, []
        for player in eligible:
            track_id = int(player.get("track_id", -1))
            if track_id < 0:
                continue
            profile = self._profile(track_id)
            self._reuse_team_profile(player, profile)
            if hasattr(self.visual, "ready") and not self.visual.ready(track_id):
                continue
            cadence = self.jersey_interval_frames * (
                self.identity_locked_jersey_cadence_multiplier if profile.get("identity_locked") else 1
            )
            if frame_id - int(profile.get("last_jersey_attempt", -10**9)) < cadence:
                continue

            profile["last_jersey_attempt"] = frame_id
            attempts += 1
            player["ocr_attempted"] = True
            prediction = dict(self.visual.predict_jersey(track_id=track_id, team_name=profile.get("team")) or {})
            prediction.update({
                "track_id": track_id, "frame_id": frame_id, "team": profile.get("team"),
                "team_confidence": round(float(profile.get("team_confidence", 0.0)), 4),
                "team_margin": round(float(profile.get("team_margin", 0.0)), 6),
                "identity_locked_before": bool(profile.get("identity_locked")),
            })

            candidate = prediction.get("candidate_number")
            if prediction.get("accepted") and candidate and profile.get("team"):
                posterior = max(0.0, float(prediction.get("candidate_posterior", 0.0) or 0.0))
                margin = max(0.0, float(prediction.get("candidate_margin", 0.0) or 0.0))
                team_conf = max(0.35, float(profile.get("team_confidence", 0.0) or 0.0))
                profile["jersey_evidence"].append({
                    "frame_id": frame_id, "number": str(candidate),
                    "weight": max(0.05, posterior) * team_conf * (0.75 + 0.25 * margin),
                })
                number, share, vote_margin, count, evidence = self._aggregate_jersey_evidence(profile, frame_id)
                prediction["identity_v2_jersey_evidence"] = {
                    "winner": number, "share": round(share, 6), "margin": round(vote_margin, 6),
                    "count": count, "evidence": round(evidence, 6),
                    "lock_min_evidence": round(self.identity_lock_min_evidence, 6),
                }
                changed = bool(
                    profile.get("shirt_num") is not None
                    and (str(profile.get("shirt_num")) != str(number) or str(profile.get("identity_team")) != str(profile.get("team")))
                )
                enough = bool(
                    number and count >= 2 and share >= (0.66 if changed else 0.60)
                    and vote_margin >= (0.24 if changed else 0.18)
                    and evidence >= self.identity_lock_min_evidence
                )
                if enough:
                    was_locked = bool(profile.get("identity_locked"))
                    self._lock_identity(
                        profile, team=str(profile["team"]), number=str(number), frame_id=frame_id, changed=changed
                    )
                    if changed or not was_locked:
                        locks += 1
                    prediction["identity_action"] = "relocked" if changed else ("confirmed" if was_locked else "locked")
                    player["ocr_success"] = True
                else:
                    prediction["identity_action"] = "evidence_accumulating"
            else:
                raw_candidate = prediction.get("raw_canonical") or prediction.get("candidate_number")
                conflict = bool(
                    profile.get("identity_locked") and raw_candidate
                    and str(raw_candidate) != str(profile.get("shirt_num"))
                    and float(prediction.get("candidate_posterior", 0.0) or 0.0) >= 0.65
                )
                if conflict:
                    profile["identity_conflict_streak"] = int(profile.get("identity_conflict_streak", 0)) + 1
                    if int(profile["identity_conflict_streak"]) >= 2:
                        self._clear_identity_number(profile, reason="sustained_jersey_conflict")
                        prediction["identity_action"] = "unlocked_to_unknown"
                    else:
                        prediction["identity_action"] = "conflict_observed"
                else:
                    profile["identity_conflict_streak"] = max(0, int(profile.get("identity_conflict_streak", 0)) - 1)
                    prediction["identity_action"] = "abstained"

            self._reuse_team_profile(player, profile)
            prediction.update(
                identity_locked=bool(profile.get("identity_locked")),
                identity_revision=int(profile.get("identity_revision", 0)),
                resolved_player_name=profile.get("player_name"),
            )
            predictions.append(prediction)

        diag = frame_state.diagnostics.setdefault("detector", {})
        diag["jersey_buffer_ms"] = round(buffer_ms, 4)
        diag["jersey_buffer_observations"] = buffered
        diag["jersey_inference_ms"] = round((perf_counter() - inference_started) * 1000.0, 4)
        diag["identity_v2"] = {
            "enabled": True, "unknown_preferred": True, "roster_policy": "validator_only",
            "identity_lock_reversible": True,
            "identity_lock_min_evidence": self.identity_lock_min_evidence,
            "locked_jersey_cadence_multiplier": self.identity_locked_jersey_cadence_multiplier,
        }
        return attempts, locks, predictions
