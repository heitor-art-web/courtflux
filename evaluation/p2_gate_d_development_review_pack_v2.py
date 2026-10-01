from __future__ import annotations

import argparse
from pathlib import Path

from evaluation import p2_gate_d_development_review_pack as base


MAX_PER_TYPE_PER_CLIP = 4


def _select_candidates_v2(
    clip: str,
    by_track: dict[int, list[base.Obs]],
    by_frame: dict[int, list[base.Obs]],
    fps: float,
) -> list[dict]:
    continuity_rows: list[dict] = []
    switch_rows: list[dict] = []
    fragment_rows: list[dict] = []
    covisible_rows: list[dict] = []

    # 1) Long-ish same-raw-track continuity candidates. Human review decides whether
    # the tracker really stayed on one person or silently handed the ID to someone else.
    min_span = max(1, int(round(fps * 1.5)))
    for track_id, rows in by_track.items():
        if len(rows) < 2:
            continue
        left = rows[0]
        right = rows[-1]
        span = right.frame_id - left.frame_id
        if span < min_span:
            continue
        nd = base._norm_distance(left, right)
        score = min(6.0, span / max(fps, 1.0)) + min(2.0, nd) * 0.15
        continuity_rows.append(
            base._candidate_row(
                "raw_track_continuity",
                clip,
                left,
                right,
                score,
                "Same raw track persists over time. Review whether it is truly the same human or contains an identity handoff.",
            )
        )

    # 2) Suspicious short-horizon jumps inside one raw track. No team signal required.
    for track_id, rows in by_track.items():
        for left, right in zip(rows, rows[1:]):
            gap = right.frame_id - left.frame_id
            if gap <= 0 or gap > max(6, int(round(fps * 0.40))):
                continue
            nd = base._norm_distance(left, right)
            if nd < 0.60:
                continue
            switch_rows.append(
                base._candidate_row(
                    "raw_track_switch",
                    clip,
                    left,
                    right,
                    nd,
                    "Same raw track makes a large short-horizon geometric jump. Review for an identity switch.",
                )
            )

    # 3) Different raw-track endpoints that could be a fragmentation/reappearance.
    # Deliberately team-agnostic: the human decides SAME / DIFFERENT / UNKNOWN.
    endpoints: list[tuple[int, base.Obs, base.Obs]] = []
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
            nd = base._norm_distance(left_last, right_first)
            if nd > 2.75:
                continue
            seconds = gap / max(fps, 1.0)
            score = (1.0 / max(0.15, seconds)) + (1.0 / max(0.25, nd + 0.15))
            fragment_rows.append(
                base._candidate_row(
                    "fragment_reappearance",
                    clip,
                    left_last,
                    right_first,
                    score,
                    "Different raw tracks have a plausible temporal/spatial handoff. Review whether this is the same human after fragmentation.",
                )
            )

    # 4) Any two real co-visible tracks are necessarily DIFFERENT. Team is irrelevant.
    # Sample roughly once per second to keep the human pack compact.
    stride = max(1, int(round(fps)))
    for frame_id in sorted(by_frame)[::stride]:
        rows = by_frame[frame_id]
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                if left.track_id == right.track_id:
                    continue
                nd = base._norm_distance(left, right)
                if nd < 0.40 or nd > 5.0:
                    continue
                covisible_rows.append(
                    base._candidate_row(
                        "covisible_negative",
                        clip,
                        left,
                        right,
                        min(5.0, nd),
                        "Two different raw tracks are simultaneously visible. If both boxes are real players this is a direct DIFFERENT cannot-link example.",
                    )
                )

    def top_unique(rows: list[dict], n: int) -> list[dict]:
        ordered = sorted(rows, key=lambda row: (-float(row["priority_score"]), str(row["id"])))
        out: list[dict] = []
        used: set[tuple] = set()
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

    return (
        top_unique(continuity_rows, MAX_PER_TYPE_PER_CLIP)
        + top_unique(switch_rows, MAX_PER_TYPE_PER_CLIP)
        + top_unique(fragment_rows, MAX_PER_TYPE_PER_CLIP)
        + top_unique(covisible_rows, MAX_PER_TYPE_PER_CLIP)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate team-agnostic Gate-D human review candidates on V1/V3/V4/V5."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base.PROJECT_ROOT / "evaluation" / "runs" / "gate_d_development_review_pack_v2",
    )
    args = parser.parse_args()

    # Reuse the validated I/O, states discovery and contact-sheet machinery from v1;
    # replace candidate selection only. Existing generated development states are reused.
    base._select_candidates = _select_candidates_v2
    pack = base.run(args.output.expanduser().resolve(), generate_missing=False)
    pack["version"] = "courtflux_gate_d_development_review_pack_v2"
    pack["boundary"]["team_required_for_candidate_selection"] = False
    json_path = args.output.expanduser().resolve() / "review_pack.json"
    import json
    json_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
