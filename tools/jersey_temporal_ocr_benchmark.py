from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


HF_LISTED_NUMBERS = {"4", "6", "8", "9", "48", "49", "66", "89"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PadToSquare:
    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas


class TemporalJerseyClassifier(nn.Module):
    """Reconstruct the HF temporal checkpoint from its state-dict dimensions."""

    def __init__(
        self,
        projection_in: int,
        projection_out: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        head_in: int,
        tens_classes: int,
        ones_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        self.proj = nn.Linear(projection_in, projection_out)
        self.lstm = nn.LSTM(
            input_size=projection_out,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.head_tens = nn.Linear(head_in, tens_classes)
        self.head_ones = nn.Linear(head_in, ones_classes)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.head_in = head_in

    def _temporal_representation(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [layers * directions, batch, hidden]
        batch = hidden.shape[1]
        directions = 2 if self.bidirectional else 1
        all_hidden_width = self.num_layers * directions * self.hidden_size
        last_layer_width = directions * self.hidden_size

        if self.head_in == all_hidden_width:
            return hidden.transpose(0, 1).contiguous().reshape(batch, -1)

        if self.head_in == last_layer_width:
            start = (self.num_layers - 1) * directions
            last = hidden[start : start + directions]
            return last.transpose(0, 1).contiguous().reshape(batch, -1)

        if self.head_in == self.hidden_size:
            return hidden[-1]

        raise RuntimeError(
            "Unable to map LSTM hidden states to classifier heads: "
            f"head_in={self.head_in}, all_hidden_width={all_hidden_width}, "
            f"last_layer_width={last_layer_width}."
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        batch, steps, channels, height, width = x.shape
        flat = x.reshape(batch * steps, channels, height, width)
        features = self.backbone(flat)
        embeddings = torch.relu(self.proj(features)).reshape(batch, steps, -1)
        _, (hidden, _) = self.lstm(embeddings)
        representation = self._temporal_representation(hidden)
        return self.head_tens(representation), self.head_ones(representation)


def _iter_tensor_dicts(value: Any, path: str = "root", depth: int = 0):
    if depth > 5 or not isinstance(value, dict):
        return
    tensors = {str(key): item for key, item in value.items() if torch.is_tensor(item)}
    if tensors:
        yield path, tensors
    for key, item in value.items():
        if isinstance(item, dict):
            yield from _iter_tensor_dicts(item, f"{path}.{key}", depth + 1)


def _strip_common_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    expected = ("backbone.", "proj.", "lstm.", "head_tens.", "head_ones.")
    if any(any(key.startswith(prefix) for prefix in expected) for key in state):
        return state
    for prefix in ("module.", "model.", "network.", "net."):
        stripped = {
            (key[len(prefix) :] if key.startswith(prefix) else key): value
            for key, value in state.items()
        }
        if any(key.startswith("lstm.") for key in stripped):
            return stripped
    return state


def _select_temporal_state(checkpoint: Any) -> tuple[str, dict[str, torch.Tensor]]:
    best: tuple[tuple[int, int, int], str, dict[str, torch.Tensor]] | None = None
    for source, raw in _iter_tensor_dicts(checkpoint):
        state = _strip_common_prefix(raw)
        score = (
            sum(key.startswith("lstm.") for key in state),
            sum(key.startswith("backbone.features.") for key in state),
            len(state),
        )
        if best is None or score > best[0]:
            best = (score, source, state)
    if best is None or best[0][0] == 0:
        raise RuntimeError("Checkpoint contains no temporal LSTM state dictionary")
    return best[1], best[2]


def _infer_architecture(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    required = [
        "proj.weight",
        "proj.bias",
        "lstm.weight_ih_l0",
        "lstm.weight_hh_l0",
        "head_tens.weight",
        "head_tens.bias",
        "head_ones.weight",
        "head_ones.bias",
    ]
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"Temporal checkpoint missing required keys: {missing}")

    projection_out, projection_in = map(int, state["proj.weight"].shape)
    hidden_size = int(state["lstm.weight_hh_l0"].shape[1])
    lstm_input = int(state["lstm.weight_ih_l0"].shape[1])
    if lstm_input != projection_out:
        raise RuntimeError(
            f"LSTM input={lstm_input} does not match projection output={projection_out}"
        )

    layer_indices = []
    for key in state:
        match = re.fullmatch(r"lstm\.weight_ih_l(\d+)(?:_reverse)?", key)
        if match:
            layer_indices.append(int(match.group(1)))
    num_layers = max(layer_indices, default=0) + 1
    bidirectional = any(key.endswith("_reverse") for key in state if key.startswith("lstm."))
    tens_classes, head_in = map(int, state["head_tens.weight"].shape)
    ones_classes, ones_head_in = map(int, state["head_ones.weight"].shape)
    if ones_head_in != head_in:
        raise RuntimeError("Tens and ones heads use different input widths")

    return {
        "projection_in": projection_in,
        "projection_out": projection_out,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "head_in": head_in,
        "tens_classes": tens_classes,
        "ones_classes": ones_classes,
    }


def _device(name: str) -> torch.device:
    requested = (name or "cpu").lower()
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(model_path: Path, device: torch.device):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    source, state = _select_temporal_state(checkpoint)
    architecture = _infer_architecture(state)
    model = TemporalJerseyClassifier(**architecture)
    model_state = model.state_dict()
    missing = [key for key in model_state if key not in state]
    unexpected = [key for key in state if key not in model_state]
    mismatches = [
        (key, tuple(state[key].shape), tuple(model_state[key].shape))
        for key in model_state
        if key in state and tuple(state[key].shape) != tuple(model_state[key].shape)
    ]
    if missing or unexpected or mismatches:
        raise RuntimeError(
            "Temporal checkpoint does not exactly match reconstructed architecture: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, mismatches={mismatches[:5]}"
        )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    metadata: dict[str, Any] = {"state_source": source, **architecture}
    if isinstance(checkpoint, dict):
        for key in ("epoch", "val_num_acc", "val_acc", "best_acc"):
            value = checkpoint.get(key)
            if isinstance(value, (int, float, str)):
                metadata[key] = value
    metadata["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    metadata["state_keys"] = len(state)
    return model, metadata


def _transform() -> transforms.Compose:
    return transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _uniform_indices(count: int, target: int = 8) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if target <= 0:
        raise ValueError("target must be positive")
    if count == 1:
        return [0] * target
    return np.rint(np.linspace(0, count - 1, target)).astype(int).tolist()


def _raw_prediction(tens_probs: np.ndarray, ones_probs: np.ndarray) -> dict[str, Any]:
    tens = int(np.argmax(tens_probs))
    ones = int(np.argmax(ones_probs))
    raw = f"{tens}{ones}"
    canonical = str(int(raw))
    probability = float(tens_probs[tens] * ones_probs[ones])
    return {
        "tens_digit": tens,
        "ones_digit": ones,
        "prediction": canonical,
        "raw_probability": probability,
    }


def _top_numbers(tens_probs: np.ndarray, ones_probs: np.ndarray, n: int = 5) -> list[str]:
    scores = []
    for number in range(100):
        tens = number // 10
        ones = number % 10
        scores.append((str(number), float(tens_probs[tens] * ones_probs[ones])))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [number for number, _ in scores[:n]]


def _group_key(row: pd.Series) -> str:
    identity = str(row.get("identity_group", "") or "").strip()
    if identity:
        return f"{row.get('video', '')}|identity:{identity}"
    return f"{row.get('video', '')}|track:{row.get('track_id', '')}"


def _sequence_groups(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    work = frame.copy()
    work["benchmark_group"] = work.apply(_group_key, axis=1)
    groups: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for key, rows in work.groupby("benchmark_group", sort=False):
        rows = rows.sort_values("timestamp_sec", key=lambda col: pd.to_numeric(col, errors="coerce"))
        numeric = rows[rows["label_kind"] == "number"]
        labels = sorted(
            {str(int(value)) for value in numeric["jersey_gt"].tolist() if str(value).isdigit()},
            key=int,
        )
        if not labels:
            continue
        if len(labels) > 1:
            skipped.append(
                {
                    "group": key,
                    "reason": "multiple_numeric_labels_without_identity_segmentation",
                    "labels": "|".join(labels),
                    "rows": int(len(rows)),
                }
            )
            continue
        split_counts = Counter(str(value) for value in rows["split"].tolist() if str(value))
        split = split_counts.most_common(1)[0][0] if split_counts else ""
        groups.append(
            {
                "group": key,
                "gt": labels[0],
                "split": split,
                "rows": rows,
                "source_crops": int(len(rows)),
                "numeric_crops": int(len(numeric)),
                "hf_listed_number": labels[0] in HF_LISTED_NUMBERS,
            }
        )
    return groups, skipped


def _safe_div(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"groups": 0, "correct": 0, "exact_accuracy": None, "top3_accuracy": None, "top5_accuracy": None}
    exact = frame["correct"].astype(bool)
    return {
        "groups": int(len(frame)),
        "correct": int(exact.sum()),
        "exact_accuracy": float(exact.mean()),
        "top3_accuracy": float(frame["top3_hit"].astype(bool).mean()),
        "top5_accuracy": float(frame["top5_hit"].astype(bool).mean()),
        "mean_source_crops": float(frame["source_crops"].mean()),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Jersey OCR temporal checkpoint benchmark",
        "",
        "This benchmark evaluates the Hugging Face temporal BiLSTM checkpoint without roster rewriting.",
        "Sequences are built from the available CourtFlux GT player crops, ordered by timestamp and uniformly resampled to 8 frames.",
        "Groups with multiple numeric labels and no identity segmentation are excluded to avoid mixing confirmed tracker ID switches.",
        "",
        "## Checkpoint",
        "",
        f"- Path: `{report['model']['checkpoint']}`",
        f"- Device: `{report['model']['device']}`",
        f"- State source: `{report['model']['architecture']['state_source']}`",
        f"- Projection: {report['model']['architecture']['projection_in']} → {report['model']['architecture']['projection_out']}",
        f"- LSTM: hidden={report['model']['architecture']['hidden_size']}, layers={report['model']['architecture']['num_layers']}, bidirectional={report['model']['architecture']['bidirectional']}",
        f"- Head input: {report['model']['architecture']['head_in']}",
        f"- Parameters: {report['model']['architecture']['parameter_count']:,}",
        f"- Inference: {report['model']['inference_seconds']:.3f}s for {report['dataset']['evaluated_sequences']} sequences",
        "",
        "## Dataset contract",
        "",
        f"- OCR player crops available: {report['dataset']['ocr_rows']}",
        f"- Pure labelled sequence groups: {report['dataset']['evaluated_sequences']}",
        f"- Ambiguous ID-switch groups skipped: {report['dataset']['skipped_groups']}",
        f"- Groups with 8 source crops: {report['dataset']['full_8_crop_groups']}",
        f"- Multi-crop groups: {report['dataset']['multi_crop_groups']}",
        "",
        "The source GT was sampled roughly once per second with at most 8 crops per track. This is therefore a sparse-sequence domain test, not an exact reproduction of the author's original video sampling.",
        "",
        "## Accuracy",
        "",
        "| Crop input | Groups | Exact | Top-3 | Top-5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for crop_type, metrics in report["metrics_by_crop"].items():
        lines.append(
            f"| {crop_type} | {metrics['groups']} | {_pct(metrics['exact_accuracy'])} | {_pct(metrics['top3_accuracy'])} | {_pct(metrics['top5_accuracy'])} |"
        )

    primary = report["primary_crop"]
    lines += [
        "",
        f"Primary comparison crop: **{primary}**",
        "",
        "### By split",
        "",
        "| Split | Groups | Exact | Top-3 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        metrics = report["metrics_by_split"][split]
        lines.append(f"| {split} | {metrics['groups']} | {_pct(metrics['exact_accuracy'])} | {_pct(metrics['top3_accuracy'])} |")

    lines += [
        "",
        "### Sequence support",
        "",
        f"- Full 8-source-crop groups: {_pct(report['support_metrics']['full_8']['exact_accuracy'])} exact ({report['support_metrics']['full_8']['groups']} groups)",
        f"- Multi-crop groups: {_pct(report['support_metrics']['multi_crop']['exact_accuracy'])} exact ({report['support_metrics']['multi_crop']['groups']} groups)",
        f"- HF model-card listed jersey numbers: {_pct(report['support_metrics']['hf_listed']['exact_accuracy'])} exact ({report['support_metrics']['hf_listed']['groups']} groups)",
        "",
    ]

    if report.get("anchor_comparison"):
        comp = report["anchor_comparison"]
        lines += [
            "## Anchor comparison",
            "",
            f"- Existing single-frame anchor exact accuracy: {_pct(comp.get('anchor_exact_accuracy'))}",
            f"- Temporal sequence exact accuracy ({primary}): {_pct(comp.get('temporal_exact_accuracy'))}",
            "",
            "These metrics use different units (anchor crops vs temporal groups), so treat the comparison as directional rather than a paired statistical test.",
            "",
        ]

    lines += ["## Most common predictions", ""]
    for number, count in report["prediction_distribution"][:12]:
        lines.append(f"- `{number}`: {count}")
    lines += ["", "## Largest confusions", ""]
    if report["confusions"]:
        for item in report["confusions"][:15]:
            lines.append(f"- GT `{item['gt']}` → predicted `{item['prediction']}`: {item['count']}")
    else:
        lines.append("- None")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the prxkc temporal jersey BiLSTM checkpoint on CourtFlux GT")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--model", default="weights/best_temporal.pt")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop", default="both", choices=["player", "torso", "both"])
    parser.add_argument("--output-dir")
    parser.add_argument("--anchor-report")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    for path in (gt_path, model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else gt_path.parent / "ocr_temporal_benchmark_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    required = {
        "sample_id", "video", "track_id", "timestamp_sec", "identity_group",
        "subject_type", "jersey_gt", "label_kind", "dataset_role", "split",
        "player_crop", "torso_crop",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ground truth is missing required columns: {missing}")

    ocr = frame[
        (frame["dataset_role"] == "ocr")
        & (frame["subject_type"] == "player")
        & frame["label_kind"].isin(["number", "unreadable"])
    ].copy()
    if ocr.empty:
        raise ValueError("No OCR player rows found")

    groups, skipped = _sequence_groups(ocr)
    if not groups:
        raise ValueError("No label-pure sequence groups available for temporal benchmark")

    device = _device(args.device)
    model, architecture = _build_model(model_path, device)
    print(
        "[TEMPORAL OCR] "
        f"proj={architecture['projection_in']}->{architecture['projection_out']} "
        f"lstm_hidden={architecture['hidden_size']} layers={architecture['num_layers']} "
        f"bidir={architecture['bidirectional']} head_in={architecture['head_in']} "
        f"params={architecture['parameter_count']:,}",
        flush=True,
    )

    crop_types = ["player", "torso"] if args.crop == "both" else [args.crop]
    transform = _transform()
    all_results: list[pd.DataFrame] = []
    total_inference_seconds = 0.0

    for crop_type in crop_types:
        path_column = f"{crop_type}_crop"
        sequences: list[torch.Tensor] = []
        metadata_rows: list[dict[str, Any]] = []

        for group in groups:
            rows = group["rows"]
            source_tensors: list[torch.Tensor] = []
            for _, row in rows.iterrows():
                crop_path = gt_path.parent / str(row[path_column])
                crop = cv2.imread(str(crop_path))
                if crop is None or crop.size == 0:
                    raise FileNotFoundError(f"Missing/unreadable crop: {crop_path}")
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                source_tensors.append(transform(Image.fromarray(rgb)))
            indices = _uniform_indices(len(source_tensors), 8)
            sequences.append(torch.stack([source_tensors[index] for index in indices]))
            metadata_rows.append(
                {
                    "crop_type": crop_type,
                    "group": group["group"],
                    "gt": group["gt"],
                    "split": group["split"],
                    "source_crops": group["source_crops"],
                    "numeric_crops": group["numeric_crops"],
                    "hf_listed_number": group["hf_listed_number"],
                    "sample_indices": "|".join(str(index) for index in indices),
                }
            )

        tens_rows: list[np.ndarray] = []
        ones_rows: list[np.ndarray] = []
        started = time.perf_counter()
        batch_size = max(1, int(args.batch_size))
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                batch = torch.stack(sequences[start : start + batch_size]).to(device)
                tens_logits, ones_logits = model(batch)
                tens_rows.extend(torch.softmax(tens_logits, dim=1).detach().cpu().numpy())
                ones_rows.extend(torch.softmax(ones_logits, dim=1).detach().cpu().numpy())
        elapsed = time.perf_counter() - started
        total_inference_seconds += elapsed

        records = []
        for index, meta in enumerate(metadata_rows):
            tens = tens_rows[index]
            ones = ones_rows[index]
            raw = _raw_prediction(tens, ones)
            top5 = _top_numbers(tens, ones, 5)
            gt = meta["gt"]
            records.append(
                {
                    **meta,
                    **raw,
                    "top3_numbers": "|".join(top5[:3]),
                    "top5_numbers": "|".join(top5),
                    "top3_hit": gt in top5[:3],
                    "top5_hit": gt in top5,
                    "correct": raw["prediction"] == gt,
                }
            )
        all_results.append(pd.DataFrame(records))
        print(
            f"[TEMPORAL OCR] crop={crop_type} sequences={len(records)} elapsed={elapsed:.3f}s "
            f"exact={_metrics(all_results[-1])['exact_accuracy']:.6f}",
            flush=True,
        )

    results = pd.concat(all_results, ignore_index=True)
    primary_crop = "player" if "player" in crop_types else crop_types[0]
    primary = results[results["crop_type"] == primary_crop].copy()

    metrics_by_crop = {
        crop_type: _metrics(results[results["crop_type"] == crop_type])
        for crop_type in crop_types
    }
    metrics_by_split = {
        split: _metrics(primary[primary["split"] == split])
        for split in ("train", "val", "test")
    }
    support_metrics = {
        "full_8": _metrics(primary[primary["source_crops"] >= 8]),
        "multi_crop": _metrics(primary[primary["source_crops"] >= 2]),
        "hf_listed": _metrics(primary[primary["hf_listed_number"].astype(bool)]),
    }

    prediction_distribution = [
        [prediction, count]
        for prediction, count in Counter(primary["prediction"].astype(str)).most_common()
    ]
    confusion_counts = Counter(
        (str(row["gt"]), str(row["prediction"]))
        for _, row in primary.iterrows()
        if str(row["gt"]) != str(row["prediction"])
    )
    confusions = [
        {"gt": gt, "prediction": prediction, "count": count}
        for (gt, prediction), count in confusion_counts.most_common()
    ]

    anchor_report_path = None
    if args.anchor_report:
        anchor_report_path = Path(args.anchor_report).expanduser().resolve()
    else:
        candidate = gt_path.parent / "ocr_benchmark_v1" / "jersey_ocr_benchmark.json"
        if candidate.exists():
            anchor_report_path = candidate
    anchor_comparison = None
    if anchor_report_path and anchor_report_path.exists():
        anchor = json.loads(anchor_report_path.read_text(encoding="utf-8"))
        anchor_comparison = {
            "anchor_report": str(anchor_report_path),
            "anchor_exact_accuracy": anchor.get("numeric_metrics", {}).get("all", {}).get("exact_accuracy"),
            "temporal_exact_accuracy": metrics_by_crop[primary_crop].get("exact_accuracy"),
        }

    report = {
        "dataset": {
            "ocr_rows": int(len(ocr)),
            "evaluated_sequences": int(len(groups)),
            "skipped_groups": int(len(skipped)),
            "full_8_crop_groups": int(sum(group["source_crops"] >= 8 for group in groups)),
            "multi_crop_groups": int(sum(group["source_crops"] >= 2 for group in groups)),
        },
        "model": {
            "checkpoint": str(model_path),
            "device": device.type,
            "batch_size": max(1, int(args.batch_size)),
            "architecture": architecture,
            "inference_seconds": float(total_inference_seconds),
        },
        "preprocessing": {
            "pad_to_square": True,
            "resize": [128, 128],
            "normalization": "ImageNet",
            "sequence_frames": 8,
            "sampling": "uniform over available GT crops; repeats indices when fewer than 8 are available",
        },
        "primary_crop": primary_crop,
        "metrics_by_crop": metrics_by_crop,
        "metrics_by_split": metrics_by_split,
        "support_metrics": support_metrics,
        "prediction_distribution": prediction_distribution,
        "confusions": confusions,
        "skipped": skipped,
        "anchor_comparison": anchor_comparison,
        "limitations": [
            "CourtFlux GT crops are sparse (~1 second sampling), unlike the author's original video sequences.",
            "The HF model card reports training on a limited jersey-number subset; most CourtFlux GT numbers are outside the listed per-class subset.",
            "Confirmed tracker ID switches without identity_group segmentation are excluded rather than mixing physical players in one sequence.",
        ],
    }

    results.to_csv(output_dir / "jersey_temporal_predictions.csv", index=False)
    pd.DataFrame(skipped).to_csv(output_dir / "jersey_temporal_skipped.csv", index=False)
    (output_dir / "jersey_temporal_benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(report, output_dir / "jersey_temporal_benchmark.md")

    print(f"benchmark={output_dir / 'jersey_temporal_benchmark.md'}")
    print(f"predictions={output_dir / 'jersey_temporal_predictions.csv'}")
    print(f"primary_crop={primary_crop}")
    print(f"exact_accuracy={metrics_by_crop[primary_crop]['exact_accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
