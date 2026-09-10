from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from tools.jersey_uncertainty_ocr_benchmark import (
    MODEL_NAME,
    UncertaintyJNRModel,
    _load_checkpoint,
    _preprocess_rgb,
    _resolve_device,
    _sync_device,
)


def _base_group(row: pd.Series) -> str:
    video = str(row.get("video", "") or "")
    identity = str(row.get("identity_group", "") or "").strip()
    if identity:
        return f"{video}|identity:{identity}"
    return f"{video}|track:{row.get('track_id', '')}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def _load_ocr_gt(gt_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    required = {
        "sample_id", "subject_type", "jersey_gt", "label_kind", "dataset_role",
        "split", "player_crop", "torso_crop",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ground truth missing required columns: {missing}")
    ocr = frame[
        (frame["dataset_role"] == "ocr")
        & (frame["subject_type"] == "player")
        & frame["label_kind"].isin(["number", "unreadable"])
    ].copy()
    ocr["jersey_gt_int"] = pd.to_numeric(
        ocr["jersey_gt"].where(ocr["label_kind"] == "number", ""), errors="coerce"
    )
    ocr["benchmark_group"] = ocr.apply(_base_group, axis=1)
    return ocr


def _infer(
    frame: pd.DataFrame,
    gt_path: Path,
    model: UncertaintyJNRModel,
    device: torch.device,
    batch_size: int,
    mode: str,
) -> pd.DataFrame:
    path_col = "player_crop" if mode == "official_player" else "torso_crop"
    records: list[dict[str, Any]] = []
    tensors: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal tensors, rows, records
        if not tensors:
            return
        batch = torch.stack(tensors).to(device)
        _sync_device(device)
        with torch.inference_mode():
            out = model(batch)
        _sync_device(device)
        probs = out["number_probs"].detach().cpu().numpy()
        all_logits = out["all_logits"].detach().cpu().numpy()
        uncertainty = out["uncertainty"].detach().cpu().numpy()
        for i, src in enumerate(rows):
            p = probs[i].astype(float)
            order = np.argsort(p)[::-1]
            pred = int(order[0])
            top2 = float(p[order[1]])
            max_number_logit = float(np.max(all_logits[i, :100]))
            absent_logit = float(all_logits[i, 100])
            rec = dict(src)
            rec.update(
                {
                    "mode": mode,
                    "pred_number": pred,
                    "pred_score": float(p[pred]),
                    "top2_score": top2,
                    "number_margin": float(p[pred] - top2),
                    "uncertainty": float(uncertainty[i]),
                    "absent_argmax": bool(int(np.argmax(all_logits[i])) == 100),
                    "absent_margin": float(absent_logit - max_number_logit),
                    "_number_probs": p,
                }
            )
            records.append(rec)
        tensors = []
        rows = []

    for _, src in frame.iterrows():
        crop = gt_path.parent / str(src[path_col])
        image = cv2.imread(str(crop))
        if image is None or image.size == 0:
            raise FileNotFoundError(crop)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensors.append(_preprocess_rgb(rgb, mode=mode))
        rows.append(src.to_dict())
        if len(tensors) >= batch_size:
            flush()
    flush()
    return pd.DataFrame(records)


def _groups(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for key, group in frame.groupby("benchmark_group", sort=False):
        labels = sorted({
            int(v)
            for v in group.loc[group["label_kind"] == "number", "jersey_gt_int"].tolist()
            if pd.notna(v)
        })
        if len(labels) > 1:
            continue
        split_values = [str(v) for v in group["split"].tolist() if str(v)]
        split = Counter(split_values).most_common(1)[0][0] if split_values else ""
        out.append(
            {
                "group": key,
                "split": split,
                "kind": "numeric" if len(labels) == 1 else "unreadable",
                "gt": labels[0] if labels else None,
                "frame": group.reset_index(drop=True),
            }
        )
    return out


def _candidate_values(series: pd.Series, quantiles: tuple[float, ...]) -> list[float | None]:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if vals.empty:
        return [None]
    result: list[float | None] = [None]
    for q in quantiles:
        value = float(vals.quantile(q))
        if all(existing is None or abs(float(existing) - value) > 1e-12 for existing in result):
            result.append(value)
    return result


def _aggregate(eligible: pd.DataFrame, method: str) -> tuple[np.ndarray, int, float, int]:
    probs = np.stack(eligible["_number_probs"].to_list())
    if method == "best_uncertainty":
        idx = int(np.argmin(eligible["uncertainty"].to_numpy(dtype=float)))
        agg = probs[idx]
    elif method == "mean_probs":
        agg = probs.mean(axis=0)
    elif method == "top3_low_uncertainty":
        order = np.argsort(eligible["uncertainty"].to_numpy(dtype=float))
        agg = probs[order[: min(3, len(order))]].mean(axis=0)
    elif method == "uncertainty_weighted":
        weights = np.maximum(1e-6, 1.0 - eligible["uncertainty"].to_numpy(dtype=float))
        agg = np.average(probs, axis=0, weights=weights)
    elif method == "score_weighted":
        weights = np.maximum(1e-6, eligible["pred_score"].to_numpy(dtype=float))
        agg = np.average(probs, axis=0, weights=weights)
    else:
        raise ValueError(method)
    order = np.argsort(agg)[::-1]
    pred = int(order[0])
    margin = float(agg[order[0]] - agg[order[1]])
    votes = eligible["pred_number"].astype(int).to_numpy()
    vote_count = int((votes == pred).sum())
    return agg, pred, margin, vote_count


def _group_prediction(group: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    frame = group["frame"]
    mask = pd.Series(True, index=frame.index)
    if policy["require_not_absent"]:
        mask &= ~frame["absent_argmax"].astype(bool)
    if policy["uncertainty_max"] is not None:
        mask &= frame["uncertainty"].astype(float) <= float(policy["uncertainty_max"])
    if policy["pred_score_min"] is not None:
        mask &= frame["pred_score"].astype(float) >= float(policy["pred_score_min"])
    eligible = frame[mask]
    if len(eligible) < int(policy["min_frames"]):
        return {"accepted": False}
    _, pred, agg_margin, vote_count = _aggregate(eligible, policy["aggregation"])
    vote_fraction = float(vote_count / len(eligible))
    if agg_margin < float(policy["agg_margin_min"]):
        return {"accepted": False}
    if vote_fraction < float(policy["vote_fraction_min"]):
        return {"accepted": False}
    return {
        "accepted": True,
        "prediction": pred,
        "eligible_frames": int(len(eligible)),
        "agg_margin": agg_margin,
        "vote_fraction": vote_fraction,
    }


def _evaluate(groups: list[dict[str, Any]], policy: dict[str, Any], split: str) -> dict[str, Any]:
    subset = [g for g in groups if split == "all" or g["split"] == split]
    numeric = [g for g in subset if g["kind"] == "numeric"]
    unreadable = [g for g in subset if g["kind"] == "unreadable"]
    accepted = []
    correct = 0
    accepted_numeric = 0
    unreadable_accept = 0
    for g in subset:
        result = _group_prediction(g, policy)
        if not result.get("accepted"):
            continue
        accepted.append((g, result))
        if g["kind"] == "numeric":
            accepted_numeric += 1
            if int(result["prediction"]) == int(g["gt"]):
                correct += 1
        else:
            unreadable_accept += 1
    return {
        "groups": int(len(subset)),
        "numeric_groups": int(len(numeric)),
        "unreadable_groups": int(len(unreadable)),
        "accepted_groups": int(len(accepted)),
        "accepted_numeric_groups": int(accepted_numeric),
        "correct": int(correct),
        "track_coverage": float(accepted_numeric / len(numeric)) if numeric else None,
        "decision_precision": float(correct / len(accepted)) if accepted else None,
        "accepted_numeric_accuracy": float(correct / accepted_numeric) if accepted_numeric else None,
        "unreadable_false_accept_rate": float(unreadable_accept / len(unreadable)) if unreadable else None,
        "correct_recall": float(correct / len(numeric)) if numeric else None,
    }


def _search_policy(
    groups: list[dict[str, Any]],
    train_frame: pd.DataFrame,
    target_precision: float,
    max_unreadable_fa: float,
) -> dict[str, Any] | None:
    uncertainty_values = _candidate_values(train_frame["uncertainty"], (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70))
    score_values = _candidate_values(train_frame["pred_score"], (0.50, 0.65, 0.75, 0.85, 0.90))
    methods = (
        "best_uncertainty",
        "mean_probs",
        "top3_low_uncertainty",
        "uncertainty_weighted",
        "score_weighted",
    )
    best: tuple[tuple[float, float, float, int], dict[str, Any], dict[str, Any]] | None = None

    for require_not_absent in (False, True):
        for u_max in uncertainty_values:
            for score_min in score_values:
                for aggregation in methods:
                    base = {
                        "require_not_absent": require_not_absent,
                        "uncertainty_max": u_max,
                        "pred_score_min": score_min,
                        "aggregation": aggregation,
                    }
                    # First pass obtains train margins/vote fractions without final track gates.
                    provisional = dict(base, min_frames=1, agg_margin_min=0.0, vote_fraction_min=0.0)
                    features = []
                    for g in groups:
                        if g["split"] != "train":
                            continue
                        pred = _group_prediction(g, provisional)
                        if pred.get("accepted"):
                            features.append(pred)
                    if not features:
                        continue
                    margins = pd.Series([f["agg_margin"] for f in features], dtype=float)
                    margin_values = _candidate_values(margins, (0.25, 0.50, 0.70, 0.85))
                    margin_values = [0.0 if v is None else float(v) for v in margin_values]

                    for min_frames in (1, 2, 3):
                        for vote_fraction_min in (0.0, 0.50, 0.67, 1.0):
                            for agg_margin_min in margin_values:
                                policy = dict(
                                    base,
                                    min_frames=min_frames,
                                    agg_margin_min=float(agg_margin_min),
                                    vote_fraction_min=float(vote_fraction_min),
                                )
                                metrics = _evaluate(groups, policy, "train")
                                precision = metrics["decision_precision"]
                                fa = metrics["unreadable_false_accept_rate"]
                                if precision is None or precision < target_precision:
                                    continue
                                if fa is not None and fa > max_unreadable_fa:
                                    continue
                                # Avoid trivial 1-2 track policies.
                                if metrics["accepted_groups"] < 5:
                                    continue
                                score = (
                                    float(metrics["correct_recall"] or 0.0),
                                    float(metrics["track_coverage"] or 0.0),
                                    float(precision),
                                    int(metrics["accepted_groups"]),
                                )
                                if best is None or score > best[0]:
                                    best = (score, policy, metrics)
    if best is None:
        return None
    return {"policy": best[1], "train_metrics": best[2]}


def _json_safe_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if v is None else v) for k, v in policy.items()}


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# ViT-B jersey track-policy benchmark",
        "",
        "Track-level CourtFlux benchmark. Policy search uses **train only**; val/test are evaluation-only.",
        "Unreadable-only tracks count as false accepts when a jersey number is emitted.",
        "",
    ]
    for mode in report["modes"]:
        lines += [f"## {mode['mode']}", ""]
        for target in report["target_precisions"]:
            item = mode["policies"].get(str(target))
            lines += [f"### Target precision {_pct(target)}", ""]
            if not item:
                lines += ["No train policy satisfied the target with at least 5 accepted tracks.", ""]
                continue
            p = item["policy"]
            lines += [
                f"Policy: `{p}`",
                "",
                "| Split | Track coverage | Decision precision | Accepted-number accuracy | Unreadable FA | Correct recall | Accepted |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for split in ("train", "val", "test", "all"):
                m = item["evaluation"][split]
                lines.append(
                    f"| {split} | {_pct(m['track_coverage'])} | {_pct(m['decision_precision'])} | "
                    f"{_pct(m['accepted_numeric_accuracy'])} | {_pct(m['unreadable_false_accept_rate'])} | "
                    f"{_pct(m['correct_recall'])} | {m['accepted_groups']} |"
                )
            lines.append("")
    lines += [
        "## Interpretation",
        "",
        "A useful production policy must exceed 80% decision precision **and** retain non-trivial track coverage on held-out data. "
        "A policy that gets 100% from one accepted test track is not sufficient evidence by itself.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate track-level uncertainty-aware jersey policy")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--model", default="weights/jersey_uncertainty_vit_b.pt")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop", default="both", choices=["official_player", "torso_direct", "both"])
    parser.add_argument("--targets", default="0.80,0.85,0.90")
    parser.add_argument("--max-unreadable-fa", type=float, default=0.05)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    targets = tuple(float(v.strip()) for v in args.targets.split(",") if v.strip())
    frame = _load_ocr_gt(gt_path)
    device = _resolve_device(args.device)
    model = UncertaintyJNRModel()
    checkpoint = _load_checkpoint(model, model_path)
    model = model.to(device).eval()

    modes = ["official_player", "torso_direct"] if args.crop == "both" else [args.crop]
    mode_reports = []
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else gt_path.parent / "ocr_uncertainty_vit_b_v1" / "track_policy_v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        print(f"[TRACK POLICY] inference mode={mode}", flush=True)
        pred_frame = _infer(frame, gt_path, model, device, max(1, args.batch_size), mode)
        groups = _groups(pred_frame)
        train_frame = pred_frame[pred_frame["split"] == "train"].copy()
        policies: dict[str, Any] = {}
        for target in targets:
            print(f"[TRACK POLICY] calibrating mode={mode} target={target:.2f}", flush=True)
            found = _search_policy(groups, train_frame, target, args.max_unreadable_fa)
            if found is None:
                policies[str(target)] = None
                continue
            policy = _json_safe_policy(found["policy"])
            evaluation = {split: _evaluate(groups, policy, split) for split in ("train", "val", "test", "all")}
            policies[str(target)] = {
                "policy": policy,
                "train_metrics": found["train_metrics"],
                "evaluation": evaluation,
            }
        mode_reports.append(
            {
                "mode": mode,
                "groups": len(groups),
                "policies": policies,
            }
        )

    report = {
        "model": {
            "checkpoint": str(model_path),
            "model_name": MODEL_NAME,
            "device": device.type,
            "checkpoint_coverage": checkpoint["coverage"],
        },
        "target_precisions": list(targets),
        "max_unreadable_false_accept_rate": args.max_unreadable_fa,
        "modes": mode_reports,
    }
    (output_dir / "jersey_uncertainty_track_policy_benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report, output_dir / "jersey_uncertainty_track_policy_benchmark.md")
    print(f"benchmark={output_dir / 'jersey_uncertainty_track_policy_benchmark.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
