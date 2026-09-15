from __future__ import annotations

import os
from pathlib import Path

from services.vision.p1_jersey_analyzer import PerformanceJerseyAnalyzer
from services.vision.p1_jersey_challengers import (
    EBQwenJerseyAnalyzer,
    RoboflowClassificationJerseyAnalyzer,
    SmolVLM2JerseyAnalyzer,
)
from services.vision.p1_parseq_jersey_analyzer import SoccerNetPARSeqJerseyAnalyzer
from services.vision.p1_parseq_literature_analyzer import LiteraturePARSeqJerseyAnalyzer
from services.vision.p1_temporal_jersey_analyzer import PerformanceTemporalJerseyAnalyzer
from services.vision.p1_trocr_jersey_analyzer import TrOCRSmallPrintedJerseyAnalyzer
from services.vision.p1_uncertainty_jersey_analyzer import PerformanceUncertaintyJerseyAnalyzer


def _project_path(project_root: Path, value: str, fallback: Path) -> Path:
    raw = str(value or "").strip()
    path = Path(raw).expanduser() if raw else fallback
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def build_jersey_visual(*, profiling_path: str, device: str, project_root: Path):
    """Build the current P1/Identity-v2 Jersey evidence backend.

    This factory survives the retirement of the old `detection_only` runtime split:
    Jersey remains optional evidence on the single canonical P1 -> P2 path.
    """
    requested = str(os.getenv("BASKETBALL_JERSEY_BACKEND", "best_temporal") or "best_temporal").strip().lower()
    aliases = {
        "temporal": "best_temporal", "bilstm": "best_temporal", "best_temporal.pt": "best_temporal",
        "vit": "uncertainty_vit_b", "vit_b": "uncertainty_vit_b", "uncertainty": "uncertainty_vit_b",
        "legacy": "legacy_digit_heads", "efficientnet": "legacy_digit_heads",
        "roboflow": "roboflow_classification", "rf": "roboflow_classification",
        "ebqwen": "ebqwen_vl", "qwen": "ebqwen_vl",
        "smolvlm2": "smolvlm2_jersey", "smol": "smolvlm2_jersey",
        "trocr": "trocr_small_printed", "trocr_small": "trocr_small_printed", "trocr-small-printed": "trocr_small_printed",
        "parseq": "parseq_soccernet", "soccernet_parseq": "parseq_soccernet", "parseq-soccernet": "parseq_soccernet",
        "parseq_lit": "parseq_soccernet_lit", "parseq-lit": "parseq_soccernet_lit", "parseq_literature": "parseq_soccernet_lit",
    }
    requested = aliases.get(requested, requested)
    allowed = {
        "auto", "best_temporal", "uncertainty_vit_b", "legacy_digit_heads",
        "roboflow_classification", "ebqwen_vl", "smolvlm2_jersey",
        "trocr_small_printed", "parseq_soccernet", "parseq_soccernet_lit",
    }
    if requested not in allowed:
        raise ValueError("Unsupported BASKETBALL_JERSEY_BACKEND: " + requested)

    if requested == "roboflow_classification":
        return RoboflowClassificationJerseyAnalyzer(profiling_path=profiling_path, device=device)
    if requested == "ebqwen_vl":
        return EBQwenJerseyAnalyzer(profiling_path=profiling_path, device=device)
    if requested == "smolvlm2_jersey":
        return SmolVLM2JerseyAnalyzer(profiling_path=profiling_path, device=device)
    if requested == "trocr_small_printed":
        return TrOCRSmallPrintedJerseyAnalyzer(profiling_path=profiling_path, device=device)
    if requested == "parseq_soccernet":
        return SoccerNetPARSeqJerseyAnalyzer(profiling_path=profiling_path, device=device)
    if requested == "parseq_soccernet_lit":
        return LiteraturePARSeqJerseyAnalyzer(profiling_path=profiling_path, device=device)

    temporal_path = _project_path(project_root, os.getenv("BASKETBALL_JERSEY_TEMPORAL_CHECKPOINT", ""), project_root / "weights" / "best_temporal.pt")
    uncertainty_path = _project_path(project_root, os.getenv("BASKETBALL_JERSEY_VIT_CHECKPOINT", ""), project_root / "weights" / "jersey_uncertainty_vit_b.pt")
    legacy_path = _project_path(project_root, os.getenv("BASKETBALL_JERSEY_LEGACY_CHECKPOINT", ""), project_root / "weights" / "jersey_number_recon.pt")

    if requested == "best_temporal" or (requested == "auto" and temporal_path.exists()):
        if not temporal_path.exists():
            raise FileNotFoundError(f"Requested best_temporal Jersey checkpoint is missing: {temporal_path}")
        return PerformanceTemporalJerseyAnalyzer(
            profiling_path=profiling_path,
            device=device,
            model_path=str(temporal_path),
            sequence_length=8,
        )
    if requested == "uncertainty_vit_b" or (requested == "auto" and uncertainty_path.exists()):
        if not uncertainty_path.exists():
            raise FileNotFoundError(f"Requested uncertainty_vit_b Jersey checkpoint is missing: {uncertainty_path}")
        return PerformanceUncertaintyJerseyAnalyzer(
            profiling_path=profiling_path,
            device=device,
            model_path=str(uncertainty_path),
            sequence_length=8,
        )
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy Jersey checkpoint not found: {legacy_path}")
    visual = PerformanceJerseyAnalyzer(
        profiling_path=profiling_path,
        device=device,
        model_path=str(legacy_path),
        sequence_length=8,
    )
    visual.checkpoint_diagnostics = dict(getattr(visual, "checkpoint_diagnostics", {}) or {})
    visual.checkpoint_diagnostics.setdefault("backend", "legacy_digit_heads")
    visual.checkpoint_diagnostics["fallback_reason"] = (
        "temporal_and_uncertainty_checkpoints_missing" if requested == "auto" else "explicit_legacy_backend"
    )
    return visual
