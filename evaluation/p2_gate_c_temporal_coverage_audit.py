from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _clean_team(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"unknown", "none", "null", "unattributed"} else text


def _best(scores: Any) -> str | None:
    if not isinstance(scores, dict) or not scores:
        return None
    values = [(str(k), float(v)) for k, v in scores.items()]
    if not values or max(v for _, v in values) <= 0.0:
        return None
    return max(values, key=lambda item: item[1])[0]


def _prediction(row: dict[str, Any], track_id: int | None) -> dict[str, Any] | None:
    if track_id is None:
        return None
    for item in list(row.get("team_predictions") or []):
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("track_id", -1)) == int(track_id):
                return item
        except (TypeError, ValueError):
            continue
    return None


def _frame_diagnostic(
    states: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    truth_team: str,
    forced_track_id: int | None = None,
) -> dict[str, Any]:
    selected = [row for row in states if start <= int(row.get("frame_id", -1)) <= end]
    current_teams: Counter[str] = Counter()
    current_tracks: Counter[str] = Counter()
    prediction_teams: Counter[str] = Counter()
    raw_best: Counter[str] = Counter()
    fused_best: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    rescored = reused = prediction_rows = raw_truth = fused_truth = 0

    for row in selected:
        team = _clean_team(row.get("current_team_id"))
        if team:
            current_teams[team] += 1
        track = row.get("current_track_id")
        if track is not None:
            current_tracks[str(track)] += 1
        target_track = forced_track_id if forced_track_id is not None else track
        pred = _prediction(row, None if target_track is None else int(target_track))
        if pred is None:
            continue
        prediction_rows += 1
        pred_team = _clean_team(pred.get("team"))
        if pred_team:
            prediction_teams[pred_team] += 1
        source = str(pred.get("source") or "unknown")
        sources[source] += 1
        if pred.get("rescored") is True:
            rescored += 1
        else:
            reused += 1
        raw = _best(pred.get("raw_scores"))
        if raw:
            raw_best[raw] += 1
            raw_truth += int(raw == truth_team)
        fused = _clean_team(pred.get("fused_best_team"))
        if fused:
            fused_best[fused] += 1
            fused_truth += int(fused == truth_team)

    return {
        "frames": len(selected),
        "current_team_votes": dict(current_teams),
        "current_track_votes": dict(current_tracks),
        "prediction_rows": prediction_rows,
        "rescored": rescored,
        "reused": reused,
        "prediction_team_votes": dict(prediction_teams),
        "raw_best_votes": dict(raw_best),
        "raw_truth_hits": raw_truth,
        "fused_best_votes": dict(fused_best),
        "fused_truth_hits": fused_truth,
        "sources": dict(sources),
    }


def analyze(scoreboard: dict[str, Any]) -> dict[str, Any]:
    run_cache: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    failures: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()

    for clip in list(scoreboard.get("clips") or []):
        if str(clip.get("clip")) == "V2":
            continue
        for sample in list(clip.get("sample_results") or []):
            if bool(sample.get("correct")):
                continue
            run_id = str(sample["run_id"])
            if run_id not in run_cache:
                run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
                run_cache[run_id] = (
                    _load_jsonl(run_dir / "p2_states.jsonl"),
                    _load_jsonl(run_dir / "p2_events.jsonl"),
                )
            states, events = run_cache[run_id]
            state = str(sample.get("state") or "")
            attributed = bool(sample.get("attributed"))
            matched_index = sample.get("matched_event_index")
            if state != "possessed" and matched_index is None:
                kind = "event_not_detected_or_not_matched"
            elif not attributed:
                kind = "team_unattributed"
            else:
                kind = "wrong_team"
            kinds[kind] += 1

            start = int(sample.get("frame_start", 0))
            end = int(sample.get("frame_end", start))
            truth = str(sample.get("truth_team_id") or "")
            event = None
            actor_track_id = None
            if state != "possessed" and matched_index is not None:
                try:
                    event = events[int(matched_index)]
                except (IndexError, TypeError, ValueError):
                    event = None
                if isinstance(event, dict):
                    try:
                        actor_track_id = int(event.get("actor_track_id")) if event.get("actor_track_id") is not None else None
                    except (TypeError, ValueError):
                        actor_track_id = None

            failures.append({
                "clip": clip.get("clip"),
                "sample_index": sample.get("sample_index"),
                "run_key": sample.get("run_key"),
                "state": state,
                "window": [start, end],
                "truth": truth,
                "predicted": sample.get("predicted_team_id"),
                "failure_kind": kind,
                "matched_event_index": matched_index,
                "event_actor_track_id": actor_track_id,
                "event_team_id": None if not isinstance(event, dict) else event.get("team_id"),
                "window_diagnostic": _frame_diagnostic(
                    states,
                    start=start,
                    end=end,
                    truth_team=truth,
                    forced_track_id=actor_track_id,
                ),
            })

    return {
        "version": "courtflux_p2_gate_c_temporal_coverage_audit_v1",
        "contract": "diagnostic-only; separates missing-event coverage from team temporal/rescore propagation failures",
        "failure_counts": dict(kinds),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoreboard", default="evaluation/gate_c_scoreboard_core_first.json")
    parser.add_argument("--output", default="evaluation/gate_c_temporal_coverage_audit.json")
    args = parser.parse_args()
    source = Path(args.scoreboard)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    result = analyze(json.loads(source.read_text(encoding="utf-8")))
    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("===== GATE C TEMPORAL / COVERAGE AUDIT =====")
    print(f"failure_counts={result['failure_counts']}")
    for row in result["failures"]:
        d = row["window_diagnostic"]
        print(
            f"{row['clip']}#{row['sample_index']} {row['state']} {row['failure_kind']} "
            f"truth={row['truth']} pred={row['predicted']} window={row['window']} "
            f"event_actor={row['event_actor_track_id']} current={d['current_team_votes']} "
            f"rescored={d['rescored']} reused={d['reused']} raw={d['raw_best_votes']} "
            f"fused={d['fused_best_votes']} profile={d['prediction_team_votes']}"
        )
    print(f"[TEMPORAL_COVERAGE_AUDIT] {out.resolve()}")


if __name__ == "__main__":
    main()
