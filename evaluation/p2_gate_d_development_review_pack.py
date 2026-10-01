from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_CLIPS = ("v1", "v3", "v4", "v5")
MAX_TOTAL_PER_CLIP = 15
TYPE_LIMITS = {
    "raw_track_continuity": 2,
    "raw_track_switch": 3,
    "fragment_reappearance": 6,
    "covisible_negative": 4,
}


@dataclass(frozen=True, slots=True)
class Obs:
    frame_id: int
    timestamp_sec: float
    track_id: int
    box: tuple[float, float, float, float]
    team: str | None
    team_confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def height(self) -> float:
        return max(1.0, self.box[3] - self.box[1])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _video_id(source: str) -> str | None:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", source)
    return None if match is None else match.group(1)


def _valid_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _distance(a: Obs, b: Obs) -> float:
    return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])


def _norm_distance(a: Obs, b: Obs) -> float:
    return _distance(a, b) / max(1.0, (a.height + b.height) * 0.5)


def _read_states(
    path: Path,
) -> tuple[dict[int, list[Obs]], dict[int, list[Obs]], dict[str, int]]:
    by_track: dict[int, list[Obs]] = {}
    by_frame: dict[int, list[Obs]] = {}
    stats = {
        "state_rows": 0,
        "frames_with_team_predictions": 0,
        "prediction_rows": 0,
        "filtered_identity_ineligible": 0,
        "filtered_off_court": 0,
        "filtered_invalid_track": 0,
        "filtered_invalid_box": 0,
        "accepted_observations": 0,
        "team_labeled_observations": 0,
    }
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                continue
            stats["state_rows"] += 1
            try:
                frame_id = int(row.get("frame_id", -1))
                timestamp_sec = float(row.get("timestamp_sec", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            predictions = row.get("team_predictions") or []
            if predictions:
                stats["frames_with_team_predictions"] += 1
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                stats["prediction_rows"] += 1
                if pred.get("identity_eligible") is False:
                    stats["filtered_identity_ineligible"] += 1
                    continue
                if pred.get("active_on_court") is False:
                    stats["filtered_off_court"] += 1
                    continue
                try:
                    track_id = int(pred.get("track_id", -1))
                except (TypeError, ValueError):
                    stats["filtered_invalid_track"] += 1
                    continue
                if track_id < 0:
                    stats["filtered_invalid_track"] += 1
                    continue
                box = _valid_box(pred.get("box"))
                if box is None:
                    stats["filtered_invalid_box"] += 1
                    continue
                team = pred.get("team")
                try:
                    conf = float(pred.get("team_confidence", pred.get("confidence", 0.0)) or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                obs = Obs(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    track_id=track_id,
                    box=box,
                    team=None if team is None else str(team),
                    team_confidence=max(0.0, min(1.0, conf)),
                )
                by_track.setdefault(track_id, []).append(obs)
                by_frame.setdefault(frame_id, []).append(obs)
                stats["accepted_observations"] += 1
                if obs.team:
                    stats["team_labeled_observations"] += 1
    for rows in by_track.values():
        rows.sort(key=lambda x: x.frame_id)
    stats["unique_tracks"] = len(by_track)
    stats["frames_with_observations"] = len(by_frame)
    return by_track, by_frame, stats


def _candidate_row(
    kind: str,
    clip: str,
    left: Obs,
    right: Obs,
    score: float,
    note: str,
) -> dict[str, Any]:
    return {
        "id": f"{clip.upper()}-{kind}-{left.track_id}-{right.track_id}-{left.frame_id}-{right.frame_id}",
        "clip": clip,
        "kind": kind,
        "priority_score": round(float(score), 6),
        "suggested_review_question": "SAME or DIFFERENT?",
        "left": {
            "frame_id": left.frame_id,
            "timestamp_sec": left.timestamp_sec,
            "track_id": left.track_id,
            "box": list(left.box),
            "team": left.team,
            "team_confidence": left.team_confidence,
        },
        "right": {
            "frame_id": right.frame_id,
            "timestamp_sec": right.timestamp_sec,
            "track_id": right.track_id,
            "box": list(right.box),
            "team": right.team,
            "team_confidence": right.team_confidence,
        },
        "why_selected": note,
        "human_label": None,
        "human_note": None,
    }


def _select_candidates(
    clip: str,
    by_track: dict[int, list[Obs]],
    by_frame: dict[int, list[Obs]],
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    continuity_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    fragment_rows: list[dict[str, Any]] = []
    covisible_rows: list[dict[str, Any]] = []

    # Long same-raw-track spans are useful contamination probes: a raw ID may hide a
    # handoff even if no team evidence exists. Human review decides SAME / DIFFERENT.
    min_span = max(1, int(round(fps * 1.5)))
    for rows in by_track.values():
        if len(rows) < 2:
            continue
        left = rows[0]
        right = rows[-1]
        span = right.frame_id - left.frame_id
        if span < min_span:
            continue
        nd = _norm_distance(left, right)
        score = min(6.0, span / max(fps, 1.0)) + min(2.0, nd) * 0.15
        continuity_rows.append(
            _candidate_row(
                "raw_track_continuity",
                clip,
                left,
                right,
                score,
                "Same raw track persists over time. Review whether it stayed on one human or contains an identity handoff.",
            )
        )

    # Suspicious short-horizon geometric jumps within one raw track. Team-independent.
    for rows in by_track.values():
        for left, right in zip(rows, rows[1:]):
            gap = right.frame_id - left.frame_id
            if gap <= 0 or gap > max(6, int(round(fps * 0.40))):
                continue
            nd = _norm_distance(left, right)
            if nd < 0.60:
                continue
            switch_rows.append(
                _candidate_row(
                    "raw_track_switch",
                    clip,
                    left,
                    right,
                    nd,
                    "Same raw track makes a large short-horizon geometric jump. Review for a hidden identity handoff.",
                )
            )

    # Different raw-track endpoints that could be the same person after fragmentation.
    # Deliberately team-agnostic because neutral development clips commonly have team=None.
    endpoints: list[tuple[int, Obs, Obs]] = []
    for track_id, rows in by_track.items():
        if len(rows) >= 2:
            endpoints.append((track_id, rows[0], rows[-1]))

    max_gap = int(round(fps * 8.0))
    for left_id, _left_first, left_last in endpoints:
        for right_id, right_first, _right_last in endpoints:
            if left_id == right_id:
                continue
            gap = right_first.frame_id - left_last.frame_id
            if gap <= 0 or gap > max_gap:
                continue
            nd = _norm_distance(left_last, right_first)
            if nd > 2.75:
                continue
            seconds = gap / max(fps, 1.0)
            score = (1.0 / max(0.15, seconds)) + (1.0 / max(0.25, nd + 0.15))
            fragment_rows.append(
                _candidate_row(
                    "fragment_reappearance",
                    clip,
                    left_last,
                    right_first,
                    score,
                    "Different raw tracks have a plausible temporal/spatial handoff. Review whether this is the same human after fragmentation.",
                )
            )

    # Co-visible distinct raw tracks provide strong DIFFERENT candidates without team.
    # Sample about once per second to keep the pack compact and visually diverse.
    stride = max(1, int(round(fps)))
    for frame_id in sorted(by_frame)[::stride]:
        rows = by_frame[frame_id]
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if left.track_id == right.track_id:
                    continue
                nd = _norm_distance(left, right)
                if nd < 0.40 or nd > 5.0:
                    continue
                covisible_rows.append(
                    _candidate_row(
                        "covisible_negative",
                        clip,
                        left,
                        right,
                        min(5.0, nd),
                        "Two distinct raw tracks are visible in the same frame. Review as a candidate DIFFERENT cannot-link example.",
                    )
                )

    pools = {
        "raw_track_continuity": len(continuity_rows),
        "raw_track_switch": len(switch_rows),
        "fragment_reappearance": len(fragment_rows),
        "covisible_negative": len(covisible_rows),
    }

    def top_unique(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        ordered = sorted(rows, key=lambda row: (-float(row["priority_score"]), str(row["id"])))
        out: list[dict[str, Any]] = []
        used: set[tuple[Any, ...]] = set()
        for row in ordered:
            kind = str(row["kind"])
            left = row["left"]
            right = row["right"]
            if kind == "raw_track_continuity":
                key = (kind, int(left["track_id"]))
            else:
                key = (kind,) + tuple(sorted((int(left["track_id"]), int(right["track_id"]))))
            if key in used:
                continue
            used.add(key)
            out.append(row)
            if len(out) >= n:
                break
        return out

    selected = (
        top_unique(fragment_rows, TYPE_LIMITS["fragment_reappearance"])
        + top_unique(covisible_rows, TYPE_LIMITS["covisible_negative"])
        + top_unique(switch_rows, TYPE_LIMITS["raw_track_switch"])
        + top_unique(continuity_rows, TYPE_LIMITS["raw_track_continuity"])
    )
    return selected[:MAX_TOTAL_PER_CLIP], pools


def _crop_from_video(video_path: Path, timestamp_sec: float, box: list[float]):
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp_sec)) * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    pad_x = max(12.0, (x2 - x1) * 0.20)
    pad_y = max(12.0, (y2 - y1) * 0.12)
    ix1 = max(0, int(math.floor(x1 - pad_x)))
    iy1 = max(0, int(math.floor(y1 - pad_y)))
    ix2 = min(w, int(math.ceil(x2 + pad_x)))
    iy2 = min(h, int(math.ceil(y2 + pad_y)))
    crop = frame[iy1:iy2, ix1:ix2]
    return crop.copy() if crop.size else None


def _write_contact_sheet(candidate: dict[str, Any], video_path: Path, output: Path) -> bool:
    try:
        import cv2
        import numpy as np
    except Exception:
        return False
    left = _crop_from_video(video_path, candidate["left"]["timestamp_sec"], candidate["left"]["box"])
    right = _crop_from_video(video_path, candidate["right"]["timestamp_sec"], candidate["right"]["box"])
    if left is None or right is None:
        return False
    target_h = 260

    def resize(img):
        scale = target_h / max(1, img.shape[0])
        return cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), target_h))

    left = resize(left)
    right = resize(right)
    gap = np.zeros((target_h, 24, 3), dtype=np.uint8)
    sheet = np.concatenate([left, gap, right], axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output), sheet))


def _generate_missing_states(clip: str, gt: dict[str, Any], *, device: str) -> Path:
    video = gt.get("video") or {}
    source = str(video.get("source") or "")
    start = float(video.get("start_time_sec", 0.0) or 0.0)
    end = float(video.get("end_time_sec", 0.0) or 0.0)
    if not source or end <= start:
        raise RuntimeError(f"{clip.upper()}: invalid video source/window in GT")

    vid = _video_id(source)
    local_video = PROJECT_ROOT / "data" / f"{vid}.mp4" if vid else None
    video_arg = str(local_video) if local_video is not None and local_video.exists() else source
    match_config = str(
        (gt.get("baseline") or {}).get("match_config")
        or "evaluation/match_configs/unknown.json"
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "p2_real_video_probe.py"),
        "--video",
        video_arg,
        "--start",
        str(start),
        "--end",
        str(end),
        "--match-config",
        match_config,
        "--device",
        device,
        "--no-jersey",
    ]
    print(
        f"[Gate D dev review] {clip.upper()}: baseline states missing; generating Jersey-OFF development states",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    states_path: Path | None = None
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{clip.upper()} probe] {line}", end="", flush=True)
        marker = "[P2_STATES_FILE]"
        if marker in line:
            value = line.split(marker, 1)[1].strip()
            if value:
                states_path = Path(value).expanduser().resolve()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{clip.upper()}: p2_real_video_probe failed with exit code {return_code}")
    if states_path is None or not states_path.exists():
        raise RuntimeError(f"{clip.upper()}: probe completed but no P2_STATES_FILE was produced")
    return states_path


def _previous_states(output_dir: Path) -> dict[str, Path]:
    pack_path = output_dir / "review_pack.json"
    if not pack_path.exists():
        return {}
    try:
        pack = _load_json(pack_path)
    except Exception:
        return {}
    result: dict[str, Path] = {}
    for row in pack.get("clips", []) or []:
        if not isinstance(row, dict):
            continue
        clip = str(row.get("clip") or "").lower()
        raw_path = row.get("states")
        if clip not in DEVELOPMENT_CLIPS or not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        if path.exists():
            result[clip] = path.resolve()
    return result


def run(output_dir: Path, *, generate_missing: bool = False, device: str = "mps") -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reusable_states = _previous_states(output_dir)
    all_candidates: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []

    for clip in DEVELOPMENT_CLIPS:
        gt_path = PROJECT_ROOT / "evaluation" / "ground_truth" / f"{clip}.json"
        gt = _load_json(gt_path)
        baseline = gt.get("baseline") or {}
        baseline_run_id = str(baseline.get("run_id") or "")
        states = PROJECT_ROOT / "evaluation" / "runs" / baseline_run_id / "p2_states.jsonl"
        states_source = "historical_baseline"
        if not baseline_run_id or not states.exists():
            reused = reusable_states.get(clip)
            if reused is not None and reused.exists():
                states = reused
                states_source = "reused_previous_review_pack"
            elif not generate_missing:
                clip_rows.append({"clip": clip, "status": "missing_states", "states": str(states)})
                continue
            else:
                try:
                    states = _generate_missing_states(clip, gt, device=device)
                except Exception as exc:
                    clip_rows.append({
                        "clip": clip,
                        "status": "generation_failed",
                        "error": str(exc),
                        "states": str(states),
                    })
                    continue
                states_source = "generated_missing_baseline"

        video = gt.get("video") or {}
        fps = float(video.get("fps", 30.0) or 30.0)
        source = str(video.get("source") or "")
        vid = _video_id(source)
        video_path = PROJECT_ROOT / "data" / f"{vid}.mp4" if vid else None

        print(f"[Gate D dev review] {clip.upper()}: reading {states}", flush=True)
        by_track, by_frame, read_stats = _read_states(states)
        candidates, candidate_pools = _select_candidates(clip, by_track, by_frame, fps)
        clip_dir = output_dir / clip
        images = 0
        if video_path is not None and video_path.exists():
            for index, candidate in enumerate(candidates, start=1):
                image_path = clip_dir / f"{index:02d}_{candidate['kind']}.jpg"
                if _write_contact_sheet(candidate, video_path, image_path):
                    candidate["contact_sheet"] = str(image_path.relative_to(output_dir))
                    images += 1
        all_candidates.extend(candidates)
        clip_rows.append({
            "clip": clip,
            "status": "ok",
            "baseline_run_id": baseline_run_id,
            "states_source": states_source,
            "states": str(states),
            "generated_run_id": states.parent.name if states_source == "generated_missing_baseline" else None,
            "video": None if video_path is None else str(video_path),
            "video_found": bool(video_path is not None and video_path.exists()),
            "candidate_count": len(candidates),
            "contact_sheets": images,
            "state_diagnostics": read_stats,
            "candidate_pools_before_dedup": candidate_pools,
        })

    pack = {
        "version": "courtflux_gate_d_development_review_pack_v3",
        "purpose": "Human SAME/DIFFERENT review candidates for Gate D development on V1/V3/V4/V5 only.",
        "boundary": {
            "v6_used": False,
            "v7_used": False,
            "jersey_number_used": False,
            "jersey_ocr_used": False,
            "generated_runs_force_jersey_off": True,
            "game_player_id_used_for_candidate_selection": False,
            "team_required_for_candidate_selection": False,
            "human_label_is_required_before_scoring": True,
        },
        "clips": clip_rows,
        "candidates": all_candidates,
    }
    json_path = output_dir / "review_pack.json"
    json_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Gate D development review pack",
        "",
        "Review only V1/V3/V4/V5. For each candidate write `SAME`, `DIFFERENT`, or `UNKNOWN`.",
        "Do not use jersey number as the reason for the identity decision.",
        "Team labels are not required for candidate selection.",
        "",
    ]
    for candidate in all_candidates:
        lines += [
            f"## {candidate['id']}",
            f"- clip: `{candidate['clip']}`",
            f"- type: `{candidate['kind']}`",
            f"- left: t={candidate['left']['timestamp_sec']:.3f}s track={candidate['left']['track_id']}",
            f"- right: t={candidate['right']['timestamp_sec']:.3f}s track={candidate['right']['track_id']}",
            f"- why: {candidate['why_selected']}",
            f"- image: `{candidate.get('contact_sheet', 'not generated')}`",
            "- HUMAN LABEL: `________`",
            "- note:",
            "",
        ]
    md_path = output_dir / "REVIEW.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nGate D development review pack", flush=True)
    for row in clip_rows:
        diagnostics = row.get("state_diagnostics") or {}
        pools = row.get("candidate_pools_before_dedup") or {}
        print(
            f"{str(row['clip']).upper()}: status={row['status']} "
            f"candidates={row.get('candidate_count', 0)} images={row.get('contact_sheets', 0)} "
            f"obs={diagnostics.get('accepted_observations', 0)} "
            f"tracks={diagnostics.get('unique_tracks', 0)} "
            f"team_labeled={diagnostics.get('team_labeled_observations', 0)} "
            f"pools={pools}",
            flush=True,
        )
    print(f"review json: {json_path}", flush=True)
    print(f"review markdown: {md_path}", flush=True)
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gate-D human review candidates from development clips only.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "runs" / "gate_d_development_review_pack",
    )
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help=(
            "Run Jersey-OFF V1/V3/V4/V5 probes only when neither historical states nor "
            "states recorded by the previous review_pack.json are available locally."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
        help="Device used only by --generate-missing probes.",
    )
    args = parser.parse_args()
    run(
        args.output.expanduser().resolve(),
        generate_missing=bool(args.generate_missing),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
