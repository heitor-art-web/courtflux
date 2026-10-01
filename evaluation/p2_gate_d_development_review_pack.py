from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_CLIPS = ("v1", "v3", "v4", "v5")
MAX_PER_TYPE_PER_CLIP = 4


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


def _read_states(path: Path) -> tuple[dict[int, list[Obs]], dict[int, list[Obs]]]:
    by_track: dict[int, list[Obs]] = {}
    by_frame: dict[int, list[Obs]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                continue
            try:
                frame_id = int(row.get("frame_id", -1))
                timestamp_sec = float(row.get("timestamp_sec", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            predictions = row.get("team_predictions") or []
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                if pred.get("identity_eligible") is False or pred.get("active_on_court") is False:
                    continue
                try:
                    track_id = int(pred.get("track_id", -1))
                except (TypeError, ValueError):
                    continue
                if track_id < 0:
                    continue
                box = _valid_box(pred.get("box"))
                if box is None:
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
    for rows in by_track.values():
        rows.sort(key=lambda x: x.frame_id)
    return by_track, by_frame


def _candidate_row(kind: str, clip: str, left: Obs, right: Obs, score: float, note: str) -> dict[str, Any]:
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


def _select_candidates(clip: str, by_track: dict[int, list[Obs]], by_frame: dict[int, list[Obs]], fps: float) -> list[dict[str, Any]]:
    switch_rows: list[dict[str, Any]] = []
    fragment_rows: list[dict[str, Any]] = []
    covisible_rows: list[dict[str, Any]] = []

    # 1) Same raw track showing a strong team flip or a large short-horizon jump.
    for track_id, rows in by_track.items():
        for a, b in zip(rows, rows[1:]):
            gap = b.frame_id - a.frame_id
            if gap <= 0 or gap > max(6, int(round(fps * 0.35))):
                continue
            nd = _norm_distance(a, b)
            strong_team_flip = bool(
                a.team and b.team and a.team != b.team
                and a.team_confidence >= 0.85 and b.team_confidence >= 0.85
            )
            if strong_team_flip or nd >= 0.75:
                score = (2.0 if strong_team_flip else 0.0) + nd
                switch_rows.append(_candidate_row(
                    "raw_track_switch",
                    clip,
                    a,
                    b,
                    score,
                    "Same raw track changed strongly over a short interval; review for hidden identity handoff.",
                ))

    # Track endpoints for fragmentation/reappearance proposals.
    endpoints = []
    for track_id, rows in by_track.items():
        if len(rows) < 2:
            continue
        endpoints.append((track_id, rows[0], rows[-1]))

    max_gap = int(round(fps * 8.0))
    for left_id, _left_first, left_last in endpoints:
        for right_id, right_first, _right_last in endpoints:
            if left_id == right_id:
                continue
            gap = right_first.frame_id - left_last.frame_id
            if gap <= 0 or gap > max_gap:
                continue
            same_team = bool(
                left_last.team and right_first.team
                and left_last.team == right_first.team
                and left_last.team_confidence >= 0.60
                and right_first.team_confidence >= 0.60
            )
            if not same_team:
                continue
            nd = _norm_distance(left_last, right_first)
            if nd > 2.5:
                continue
            score = (1.0 / max(0.15, gap / max(fps, 1.0))) + (1.0 / max(0.25, nd + 0.15))
            fragment_rows.append(_candidate_row(
                "fragment_reappearance",
                clip,
                left_last,
                right_first,
                score,
                "Different raw tracks from the same team with a plausible temporal/spatial handoff; review whether this is the same human after fragmentation.",
            ))

    # 3) Co-visible same-team negatives: definitely DIFFERENT if both boxes are real players.
    stride = max(1, int(round(fps * 1.0)))
    for frame_id in sorted(by_frame)[::stride]:
        rows = by_frame[frame_id]
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                if not a.team or a.team != b.team:
                    continue
                if a.team_confidence < 0.65 or b.team_confidence < 0.65:
                    continue
                nd = _norm_distance(a, b)
                if nd < 0.45 or nd > 4.0:
                    continue
                score = min(4.0, nd)
                covisible_rows.append(_candidate_row(
                    "covisible_negative",
                    clip,
                    a,
                    b,
                    score,
                    "Two same-team tracks are visible in the same frame; useful DIFFERENT cannot-link ground truth.",
                ))

    def top_unique(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        rows = sorted(rows, key=lambda r: (-float(r["priority_score"]), str(r["id"])))
        out: list[dict[str, Any]] = []
        used_tracks: set[tuple[int, int]] = set()
        for row in rows:
            pair = tuple(sorted((int(row["left"]["track_id"]), int(row["right"]["track_id"]))))
            if pair in used_tracks:
                continue
            used_tracks.add(pair)
            out.append(row)
            if len(out) >= n:
                break
        return out

    return (
        top_unique(switch_rows, MAX_PER_TYPE_PER_CLIP)
        + top_unique(fragment_rows, MAX_PER_TYPE_PER_CLIP)
        + top_unique(covisible_rows, MAX_PER_TYPE_PER_CLIP)
    )


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


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_candidates: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []

    for clip in DEVELOPMENT_CLIPS:
        gt_path = PROJECT_ROOT / "evaluation" / "ground_truth" / f"{clip}.json"
        gt = _load_json(gt_path)
        baseline = gt.get("baseline") or {}
        run_id = str(baseline.get("run_id") or "")
        states = PROJECT_ROOT / "evaluation" / "runs" / run_id / "p2_states.jsonl"
        if not run_id or not states.exists():
            clip_rows.append({"clip": clip, "status": "missing_states", "states": str(states)})
            continue
        video = gt.get("video") or {}
        fps = float(video.get("fps", 30.0) or 30.0)
        source = str(video.get("source") or "")
        vid = _video_id(source)
        video_path = PROJECT_ROOT / "data" / f"{vid}.mp4" if vid else None

        print(f"[Gate D dev review] {clip.upper()}: reading {states}", flush=True)
        by_track, by_frame = _read_states(states)
        candidates = _select_candidates(clip, by_track, by_frame, fps)
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
            "run_id": run_id,
            "states": str(states),
            "video": None if video_path is None else str(video_path),
            "video_found": bool(video_path is not None and video_path.exists()),
            "candidate_count": len(candidates),
            "contact_sheets": images,
        })

    pack = {
        "version": "courtflux_gate_d_development_review_pack_v1",
        "purpose": "Human SAME/DIFFERENT review candidates for Gate D development on V1/V3/V4/V5 only.",
        "boundary": {
            "v6_used": False,
            "v7_used": False,
            "jersey_number_used": False,
            "jersey_ocr_used": False,
            "game_player_id_used_for_candidate_selection": False,
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
        print(
            f"{str(row['clip']).upper()}: status={row['status']} "
            f"candidates={row.get('candidate_count', 0)} images={row.get('contact_sheets', 0)}",
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
    args = parser.parse_args()
    run(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
