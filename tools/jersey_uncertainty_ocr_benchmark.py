from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


MODEL_NAME = "vit_base_patch8_224.augreg2_in21k_ft_in1k"
TARGET_SIZE = (224, 224)


class TiedDigitAwareClassifier(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        embedding_type: str = "multiplicative",
        per_digit_bias: bool = False,
    ) -> None:
        super().__init__()
        if embedding_type not in {"additive", "multiplicative"}:
            raise ValueError(f"Unsupported embedding_type={embedding_type}")
        self.embedding_type = embedding_type
        self.per_digit_bias = bool(per_digit_bias)
        self.digit_classifier = torch.nn.Linear(embed_dim, 10, bias=False)
        self.absent_classifier = torch.nn.Linear(embed_dim, 1)
        bias_width = 10 if self.per_digit_bias else 1
        self.position_biases = torch.nn.Parameter(torch.zeros(3, bias_width))
        init_value = (
            torch.randn(3, embed_dim)
            if embedding_type == "additive"
            else torch.ones(3, embed_dim)
        )
        self.position_embeddings = torch.nn.Parameter(init_value)
        self.register_buffer(
            "tens_idx",
            torch.div(torch.arange(10, 100), 10, rounding_mode="floor"),
        )
        self.register_buffer("ones_idx", torch.remainder(torch.arange(10, 100), 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding_type == "additive":
            positioned = x.unsqueeze(1) + self.position_embeddings.unsqueeze(0)
        else:
            positioned = x.unsqueeze(1) * self.position_embeddings.unsqueeze(0)
        digit_logits = self.digit_classifier(positioned)
        digit_logits = digit_logits + self.position_biases.unsqueeze(0)
        single_logits = digit_logits[:, 0]
        tens_logits = digit_logits[:, 1]
        ones_logits = digit_logits[:, 2]
        absent_logit = self.absent_classifier(x)

        number_logits = torch.empty(
            x.shape[0], 100, device=x.device, dtype=x.dtype
        )
        number_logits[:, :10] = single_logits
        number_logits[:, 10:] = (
            tens_logits[:, self.tens_idx] + ones_logits[:, self.ones_idx]
        )
        return torch.cat([number_logits, absent_logit], dim=1)


class UncertaintyJNRModel(torch.nn.Module):
    """Minimal reproduction of lukaszgrad/uncertainty-jnr TimmOCRModel."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'timm'. Install the original-compatible version with: "
                "python -m pip install 'timm==1.0.11'"
            ) from exc

        self.timm_version = getattr(timm, "__version__", "unknown")
        self.backbone = timm.create_model(MODEL_NAME, pretrained=False)
        embed_dim = int(self.backbone.embed_dim)
        self.time_embed = torch.nn.Sequential(torch.nn.Linear(1, embed_dim))
        with torch.no_grad():
            self.time_embed[0].weight.data.uniform_(-0.001, 0.001)
            self.time_embed[0].bias.data.zero_()
        self.classifier = TiedDigitAwareClassifier(
            embed_dim,
            embedding_type="multiplicative",
            per_digit_bias=False,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        all_logits = self.classifier(features)
        number_logits = all_logits[:, :100]
        alpha = torch.exp(number_logits) + 1.0
        concentration = alpha.sum(dim=1, keepdim=True)
        number_probs = alpha / concentration
        uncertainty = 100.0 / concentration.squeeze(1)
        return {
            "all_logits": all_logits,
            "number_logits": number_logits,
            "number_probs": number_probs,
            "uncertainty": uncertainty,
        }


def _resolve_device(requested: str) -> torch.device:
    requested = (requested or "cpu").lower()
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync_device(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Expected uncertainty-jnr checkpoint with top-level 'model_state_dict'."
        )
    raw_state = checkpoint["model_state_dict"]
    if not isinstance(raw_state, dict):
        raise RuntimeError("'model_state_dict' is not a state dictionary.")

    state = {}
    for key, value in raw_state.items():
        new_key = str(key).replace("._orig_mod.", ".")
        state[new_key] = value

    model_state = model.state_dict()
    shared = set(model_state).intersection(state)
    shape_mismatches = [
        (
            key,
            tuple(state[key].shape),
            tuple(model_state[key].shape),
        )
        for key in shared
        if tuple(state[key].shape) != tuple(model_state[key].shape)
    ]
    if shape_mismatches:
        raise RuntimeError(
            f"Checkpoint/model shape mismatch: {shape_mismatches[:8]}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)

    checkpoint_numel = sum(
        int(value.numel()) for value in state.values() if torch.is_tensor(value)
    )
    matched_numel = sum(
        int(state[key].numel())
        for key in shared
        if torch.is_tensor(state[key])
    )
    coverage = matched_numel / checkpoint_numel if checkpoint_numel else 0.0
    if coverage < 0.995:
        raise RuntimeError(
            "Checkpoint coverage is too low for a trustworthy benchmark: "
            f"{coverage:.2%}; missing={list(missing)[:8]}; "
            f"unexpected={list(unexpected)[:8]}"
        )

    return {
        "state_source": "root.model_state_dict",
        "checkpoint_tensor_keys": len(state),
        "matched_keys": len(shared),
        "coverage": coverage,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "val_metrics": _json_safe(checkpoint.get("val_metrics")),
        "global_step": _json_safe(checkpoint.get("global_step")),
    }


def _preprocess_rgb(
    rgb: np.ndarray,
    mode: str,
) -> torch.Tensor:
    if rgb is None or rgb.size == 0:
        raise ValueError("Empty image")
    image = rgb
    if mode == "official_player":
        height = int(image.shape[0])
        top = height // 6
        bottom = height - (height // 3)
        image = image[top:bottom, :]
        if image.size == 0:
            raise ValueError("Official jersey crop became empty")

    image = cv2.resize(
        image,
        (TARGET_SIZE[1], TARGET_SIZE[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    tensor = torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    ).float()
    return (tensor / 127.5) - 1.0


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if series.empty:
        return {
            key: None
            for key in (
                "min",
                "p10",
                "p25",
                "median",
                "p75",
                "p90",
                "max",
                "mean",
            )
        }
    return {
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def _binary_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    frame = pd.DataFrame(
        {
            "label": labels.astype(bool),
            "score": pd.to_numeric(scores, errors="coerce"),
        }
    ).dropna()
    n_pos = int(frame["label"].sum())
    n_neg = int((~frame["label"]).sum())
    if not n_pos or not n_neg:
        return None
    ranks = frame["score"].rank(method="average")
    pos_rank_sum = float(ranks[frame["label"]].sum())
    return float(
        (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def _numeric_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame[frame["label_kind"] == "number"].copy()
    if numeric.empty:
        return {
            "rows": 0,
            "correct": 0,
            "exact_accuracy": None,
            "top3_accuracy": None,
            "top5_accuracy": None,
            "median_pred_score": None,
            "median_uncertainty": None,
        }
    correct = numeric["pred_number"].astype(int) == numeric["jersey_gt_int"].astype(int)
    return {
        "rows": int(len(numeric)),
        "correct": int(correct.sum()),
        "exact_accuracy": float(correct.mean()),
        "top3_accuracy": float(numeric["top3_hit"].astype(bool).mean()),
        "top5_accuracy": float(numeric["top5_hit"].astype(bool).mean()),
        "median_pred_score": float(
            pd.to_numeric(numeric["pred_score"], errors="coerce").median()
        ),
        "median_uncertainty": float(
            pd.to_numeric(numeric["uncertainty"], errors="coerce").median()
        ),
    }


def _selective_metrics(
    frame: pd.DataFrame,
    coverages: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
) -> list[dict[str, Any]]:
    numeric = frame[frame["label_kind"] == "number"].copy()
    if numeric.empty:
        return []
    numeric = numeric.sort_values(
        ["uncertainty", "pred_score"],
        ascending=[True, False],
    )
    total = len(numeric)
    rows = []
    for target in coverages:
        keep_n = max(1, min(total, int(math.ceil(total * target))))
        kept = numeric.iloc[:keep_n]
        correct = kept["pred_number"].astype(int) == kept["jersey_gt_int"].astype(int)
        rows.append(
            {
                "target_coverage": float(target),
                "kept": int(keep_n),
                "actual_coverage": float(keep_n / total),
                "accuracy": float(correct.mean()),
                "correct_recall": float(correct.sum() / total),
                "uncertainty_max": float(kept["uncertainty"].max()),
            }
        )
    return rows


def _base_group(row: pd.Series) -> str:
    video = str(row.get("video", "") or "")
    identity = str(row.get("identity_group", "") or "").strip()
    if identity:
        return f"{video}|identity:{identity}"
    return f"{video}|track:{row.get('track_id', '')}"


def _track_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame.copy()
    work["benchmark_group"] = work.apply(_base_group, axis=1)
    group_rows = []
    skipped_ambiguous = 0

    for group_key, group in work.groupby("benchmark_group", sort=False):
        labels = sorted(
            {
                int(value)
                for value in group.loc[
                    group["label_kind"] == "number",
                    "jersey_gt_int",
                ].tolist()
                if pd.notna(value)
            }
        )
        if len(labels) != 1:
            if len(labels) > 1:
                skipped_ambiguous += 1
            continue

        gt = labels[0]
        probabilities = np.stack(group["_number_probs"].to_list())
        uncertainties = group["uncertainty"].to_numpy(dtype=float)
        order = np.argsort(uncertainties)

        all_mean = probabilities.mean(axis=0)
        best = probabilities[order[0]]
        top3 = probabilities[order[: min(3, len(order))]].mean(axis=0)

        split_values = [
            str(value)
            for value in group.get("split", pd.Series(dtype=str)).tolist()
            if str(value)
        ]
        split = Counter(split_values).most_common(1)[0][0] if split_values else ""

        group_rows.append(
            {
                "group": group_key,
                "gt": gt,
                "rows": int(len(group)),
                "numeric_rows": int((group["label_kind"] == "number").sum()),
                "split": split,
                "mean_all_pred": int(np.argmax(all_mean)),
                "best_uncertainty_pred": int(np.argmax(best)),
                "top3_uncertainty_mean_pred": int(np.argmax(top3)),
            }
        )

    groups = pd.DataFrame(group_rows)
    schemes = {}
    for name, column in (
        ("mean_all_frames", "mean_all_pred"),
        ("best_uncertainty_frame", "best_uncertainty_pred"),
        ("mean_3_least_uncertain_frames", "top3_uncertainty_mean_pred"),
    ):
        if groups.empty:
            schemes[name] = {
                "groups": 0,
                "correct": 0,
                "exact_accuracy": None,
            }
        else:
            correct = groups[column] == groups["gt"]
            schemes[name] = {
                "groups": int(len(groups)),
                "correct": int(correct.sum()),
                "exact_accuracy": float(correct.mean()),
            }

    return {
        "groups": int(len(groups)),
        "skipped_ambiguous_groups": int(skipped_ambiguous),
        "schemes": schemes,
        "rows": group_rows,
        "note": (
            "CourtFlux diagnostic aggregation. Groups must have exactly one numeric GT "
            "label; unreadable crops from that same group are retained in the aggregation."
        ),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _fmt4(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Uncertainty-aware jersey OCR benchmark",
        "",
        "Roster-free CourtFlux benchmark of `lukaszgrad/uncertainty-jnr` ViT-B.",
        "",
        "The primary input mode (`official_player`) reproduces the repository's "
        "validation preprocessing: crop top 1/6 and bottom 1/3 from the player "
        "detection, bicubic resize to 224x224, then normalize pixels to [-1, 1].",
        "",
        "## Model",
        "",
        f"- Checkpoint: `{report['model']['checkpoint']}`",
        f"- Architecture: `{report['model']['model_name']}`",
        f"- timm: `{report['model']['timm_version']}`",
        f"- Device: `{report['model']['device']}`",
        f"- Checkpoint coverage: {_pct(report['model']['checkpoint_load']['coverage'])}",
        "",
        "## Dataset",
        "",
        f"- OCR player crops: {report['dataset']['ocr_rows']}",
        f"- Readable numeric crops: {report['dataset']['numeric_rows']}",
        f"- Unreadable player crops: {report['dataset']['unreadable_rows']}",
        "",
        "## Per-crop accuracy",
        "",
        "| Input | Rows | Exact | Top-3 | Top-5 | ms/crop |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for mode in report["modes"]:
        metrics = mode["numeric_metrics"]["all"]
        lines.append(
            f"| {mode['mode']} | {metrics['rows']} | "
            f"{_pct(metrics['exact_accuracy'])} | "
            f"{_pct(metrics['top3_accuracy'])} | "
            f"{_pct(metrics['top5_accuracy'])} | "
            f"{mode['inference']['milliseconds_per_crop']:.2f} |"
        )

    primary = report["modes"][0]
    lines += [
        "",
        f"## Primary mode: {primary['mode']}",
        "",
        "### By split",
        "",
        "| Split | Rows | Exact | Top-3 | Top-5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        item = primary["numeric_metrics"][split]
        lines.append(
            f"| {split} | {item['rows']} | {_pct(item['exact_accuracy'])} | "
            f"{_pct(item['top3_accuracy'])} | {_pct(item['top5_accuracy'])} |"
        )

    lines += [
        "",
        "### By visibility",
        "",
        "| Visibility | Rows | Exact | Top-3 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for visibility, item in primary["visibility_metrics"].items():
        lines.append(
            f"| {visibility} | {item['rows']} | {_pct(item['exact_accuracy'])} | "
            f"{_pct(item['top3_accuracy'])} |"
        )

    conf = primary["confidence"]
    lines += [
        "",
        "## Uncertainty / abstention diagnostics",
        "",
        f"- Readable median prediction score: {_fmt4(conf['readable_pred_score']['median'])}",
        f"- Unreadable median prediction score: {_fmt4(conf['unreadable_pred_score']['median'])}",
        f"- Readable median uncertainty: {_fmt4(conf['readable_uncertainty']['median'])}",
        f"- Unreadable median uncertainty: {_fmt4(conf['unreadable_uncertainty']['median'])}",
        f"- Readability AUROC from prediction score: {_pct(conf['pred_score_readability_auroc'])}",
        f"- Readability AUROC from negative uncertainty: {_pct(conf['uncertainty_readability_auroc'])}",
        f"- Absent-logit argmax on readable crops: {_pct(conf['readable_absent_argmax_rate'])}",
        f"- Absent-logit argmax on unreadable crops: {_pct(conf['unreadable_absent_argmax_rate'])}",
        "",
        "### Selective accuracy by lowest uncertainty",
        "",
        "| Target coverage | Kept | Accuracy | Correct recall | Max uncertainty kept |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in primary["selective_accuracy"]:
        lines.append(
            f"| {_pct(item['target_coverage'])} | {item['kept']} | "
            f"{_pct(item['accuracy'])} | {_pct(item['correct_recall'])} | "
            f"{item['uncertainty_max']:.4f} |"
        )

    lines += [
        "",
        "## Track/identity aggregation",
        "",
        primary["track_metrics"]["note"],
        "",
        "| Scheme | Groups | Exact |",
        "| --- | ---: | ---: |",
    ]
    for name, item in primary["track_metrics"]["schemes"].items():
        lines.append(
            f"| {name} | {item['groups']} | {_pct(item['exact_accuracy'])} |"
        )

    lines += [
        "",
        f"- Ambiguous multi-label groups skipped: "
        f"{primary['track_metrics']['skipped_ambiguous_groups']}",
        "",
        "## Prediction distribution",
        "",
    ]
    for prediction, count in primary["prediction_distribution"][:12]:
        lines.append(f"- `{prediction}`: {count}")

    lines += ["", "## Largest confusions", ""]
    if primary["confusions"]:
        for item in primary["confusions"][:15]:
            lines.append(
                f"- GT `{item['gt']}` -> predicted `{item['prediction']}`: "
                f"{item['count']}"
            )
    else:
        lines.append("- None")

    lines += [
        "",
        "## Interpretation contract",
        "",
        "- No roster constraint or roster rewrite is used.",
        "- Per-crop exact accuracy is measured only on human-readable numeric GT.",
        "- `unreadable` crops are used for abstention/uncertainty diagnostics.",
        "- Track aggregation is a CourtFlux diagnostic, not a claimed reproduction "
        "of the paper's SoccerNet evaluation protocol.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_mode(
    frame: pd.DataFrame,
    gt_path: Path,
    model: UncertaintyJNRModel,
    device: torch.device,
    batch_size: int,
    mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    path_column = "player_crop" if mode == "official_player" else "torso_crop"
    records: list[dict[str, Any]] = []
    tensors: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    total_inference_seconds = 0.0

    def flush_batch() -> None:
        nonlocal tensors, rows, records, total_inference_seconds
        if not tensors:
            return
        batch = torch.stack(tensors).to(device)
        _sync_device(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(batch)
        _sync_device(device)
        elapsed = time.perf_counter() - started
        total_inference_seconds += elapsed

        probs = output["number_probs"].detach().cpu().numpy()
        uncertainty = output["uncertainty"].detach().cpu().numpy()
        all_logits = output["all_logits"].detach().cpu().numpy()

        for idx, row in enumerate(rows):
            row = dict(row)
            top5_idx = np.argsort(probs[idx])[::-1][:5]
            pred = int(top5_idx[0])
            gt = row["jersey_gt_int"]
            record = dict(row)
            record.update(
                {
                    "mode": mode,
                    "pred_number": pred,
                    "pred_score": float(probs[idx, pred]),
                    "uncertainty": float(uncertainty[idx]),
                    "absent_argmax": bool(int(np.argmax(all_logits[idx])) == 100),
                    "top3_numbers": "|".join(map(str, top5_idx[:3].tolist())),
                    "top5_numbers": "|".join(map(str, top5_idx.tolist())),
                    "top3_hit": bool(
                        pd.notna(gt) and int(gt) in top5_idx[:3].tolist()
                    ),
                    "top5_hit": bool(
                        pd.notna(gt) and int(gt) in top5_idx.tolist()
                    ),
                    "_number_probs": probs[idx].astype(float),
                    "_batch_elapsed_seconds": elapsed,
                    "_batch_size": len(rows),
                }
            )
            records.append(record)
        tensors = []
        rows = []

    for _, source_row in frame.iterrows():
        crop_path = gt_path.parent / str(source_row[path_column])
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            raise FileNotFoundError(f"Missing/unreadable crop: {crop_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = _preprocess_rgb(rgb, mode=mode)
        tensors.append(tensor)
        rows.append(source_row.to_dict())
        if len(tensors) >= batch_size:
            flush_batch()
    flush_batch()

    result_frame = pd.DataFrame(records)
    inference_seconds = float(total_inference_seconds)

    numeric_metrics = {"all": _numeric_metrics(result_frame)}
    for split in ("train", "val", "test"):
        numeric_metrics[split] = _numeric_metrics(
            result_frame[result_frame["split"] == split]
        )

    visibility_metrics = {}
    numeric_vis = result_frame[result_frame["label_kind"] == "number"]
    for visibility in sorted(numeric_vis["visibility"].astype(str).unique()):
        visibility_metrics[visibility] = _numeric_metrics(
            result_frame[result_frame["visibility"] == visibility]
        )

    readable = result_frame["label_kind"] == "number"
    unreadable = result_frame["label_kind"] == "unreadable"
    pred_score_auc = _binary_auc(
        readable[readable | unreadable],
        result_frame.loc[readable | unreadable, "pred_score"],
    )
    uncertainty_auc = _binary_auc(
        readable[readable | unreadable],
        -result_frame.loc[readable | unreadable, "uncertainty"],
    )

    numeric = result_frame[readable].copy()
    prediction_counts = Counter(
        int(value) for value in numeric["pred_number"].tolist()
    )
    confusion_counts = Counter(
        (int(row["jersey_gt_int"]), int(row["pred_number"]))
        for _, row in numeric.iterrows()
        if int(row["jersey_gt_int"]) != int(row["pred_number"])
    )

    mode_report = {
        "mode": mode,
        "path_column": path_column,
        "preprocessing": (
            "official JerseyCrop (remove top 1/6 and bottom 1/3), bicubic 224x224, [-1,1]"
            if mode == "official_player"
            else "CourtFlux torso crop, bicubic 224x224, [-1,1]"
        ),
        "inference": {
            "seconds": inference_seconds,
            "milliseconds_per_crop": float(
                1000.0 * inference_seconds / len(result_frame)
            ),
        },
        "numeric_metrics": numeric_metrics,
        "visibility_metrics": visibility_metrics,
        "confidence": {
            "readable_pred_score": _quantiles(
                result_frame.loc[readable, "pred_score"]
            ),
            "unreadable_pred_score": _quantiles(
                result_frame.loc[unreadable, "pred_score"]
            ),
            "readable_uncertainty": _quantiles(
                result_frame.loc[readable, "uncertainty"]
            ),
            "unreadable_uncertainty": _quantiles(
                result_frame.loc[unreadable, "uncertainty"]
            ),
            "pred_score_readability_auroc": pred_score_auc,
            "uncertainty_readability_auroc": uncertainty_auc,
            "readable_absent_argmax_rate": float(
                result_frame.loc[readable, "absent_argmax"].astype(bool).mean()
            )
            if int(readable.sum())
            else None,
            "unreadable_absent_argmax_rate": float(
                result_frame.loc[unreadable, "absent_argmax"].astype(bool).mean()
            )
            if int(unreadable.sum())
            else None,
        },
        "selective_accuracy": _selective_metrics(result_frame),
        "track_metrics": _track_metrics(result_frame),
        "prediction_distribution": [
            [str(number), int(count)]
            for number, count in prediction_counts.most_common()
        ],
        "confusions": [
            {
                "gt": str(gt),
                "prediction": str(pred),
                "count": int(count),
            }
            for (gt, pred), count in confusion_counts.most_common()
        ],
    }
    return mode_report, result_frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark uncertainty-aware ViT jersey OCR against CourtFlux GT"
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument(
        "--model",
        default="weights/jersey_uncertainty_vit_b.pt",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cuda", "cpu"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--crop",
        default="both",
        choices=["official_player", "torso_direct", "both"],
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else gt_path.parent / "ocr_uncertainty_vit_b_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    required = {
        "sample_id",
        "subject_type",
        "jersey_gt",
        "visibility",
        "label_kind",
        "dataset_role",
        "split",
        "player_crop",
        "torso_crop",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ground truth missing required columns: {missing}")

    ocr = frame[
        (frame["dataset_role"] == "ocr")
        & (frame["subject_type"] == "player")
        & frame["label_kind"].isin(["number", "unreadable"])
    ].copy()
    if ocr.empty:
        raise ValueError("No OCR player rows found")

    ocr["jersey_gt_int"] = pd.to_numeric(
        ocr["jersey_gt"].where(ocr["label_kind"] == "number", ""),
        errors="coerce",
    )

    device = _resolve_device(args.device)
    model = UncertaintyJNRModel()
    checkpoint_load = _load_checkpoint(model, model_path)
    model = model.to(device).eval()

    print(
        "[UNCERTAINTY JNR] "
        f"model={MODEL_NAME} timm={model.timm_version} "
        f"device={device.type} coverage={checkpoint_load['coverage']:.2%}",
        flush=True,
    )

    batch_size = max(1, int(args.batch_size))
    modes = (
        ["official_player", "torso_direct"]
        if args.crop == "both"
        else [args.crop]
    )

    mode_reports = []
    output_frames = []
    for mode in modes:
        print(f"[UNCERTAINTY JNR] benchmarking mode={mode}", flush=True)
        mode_report, predictions = _run_mode(
            frame=ocr,
            gt_path=gt_path,
            model=model,
            device=device,
            batch_size=batch_size,
            mode=mode,
        )
        mode_reports.append(mode_report)
        output_frames.append(predictions)

    report = {
        "dataset": {
            "ocr_rows": int(len(ocr)),
            "numeric_rows": int((ocr["label_kind"] == "number").sum()),
            "unreadable_rows": int((ocr["label_kind"] == "unreadable").sum()),
        },
        "model": {
            "checkpoint": str(model_path),
            "model_name": MODEL_NAME,
            "timm_version": model.timm_version,
            "device": device.type,
            "batch_size": batch_size,
            "checkpoint_load": checkpoint_load,
            "contract": {
                "classifier_type": "tied_digit_aware",
                "embedding_type": "multiplicative",
                "per_digit_bias": False,
                "uncertainty_head": "dirichlet",
                "target_size": [224, 224],
            },
        },
        "modes": mode_reports,
    }

    serializable_frames = []
    for predictions in output_frames:
        serializable_frames.append(
            predictions.drop(
                columns=[
                    "_number_probs",
                    "_batch_elapsed_seconds",
                    "_batch_size",
                ],
                errors="ignore",
            )
        )
    pd.concat(serializable_frames, ignore_index=True).to_csv(
        output_dir / "jersey_uncertainty_predictions.csv",
        index=False,
    )
    (output_dir / "jersey_uncertainty_benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        report,
        output_dir / "jersey_uncertainty_benchmark.md",
    )

    primary = mode_reports[0]["numeric_metrics"]["all"]
    print(
        f"benchmark={output_dir / 'jersey_uncertainty_benchmark.md'}",
        flush=True,
    )
    print(
        f"official_exact_accuracy={primary['exact_accuracy']:.6f}",
        flush=True,
    )
    print(
        f"official_top3_accuracy={primary['top3_accuracy']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
