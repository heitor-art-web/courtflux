from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from services.lab_contract import (
    resolve_jersey_backend,
    validate_jersey_backend_contract,
)
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


def _validated_backend(visual: Any, requested: str) -> Any:
    checkpoint = dict(getattr(visual, "checkpoint_diagnostics", {}) or {})
    actual = validate_jersey_backend_contract(requested, checkpoint.get("backend"))
    checkpoint["requested_backend"] = requested
    checkpoint["backend_contract_verified"] = True
    visual.checkpoint_diagnostics = checkpoint
    print(
        f"[JERSEY_BACKEND] requested={requested} actual={actual} contract=verified",
        flush=True,
    )
    return visual


def build_jersey_visual(*, profiling_path: str, device: str, project_root: Path):
    """Build the current P1/Identity-v2 Jersey evidence backend.

    This factory survives the retirement of the old `detection_only` runtime split:
    Jersey remains optional evidence on the single canonical P1 -> P2 path. Backend
    selection is resolved through the shared Lab contract so direct analyzer runs and
    worker/Lab runs cannot silently drift to different defaults.
    """
    requested = resolve_jersey_backend()

    if requested == "roboflow_classification":
        return _validated_backend(
            RoboflowClassificationJerseyAnalyzer(
                profiling_path=profiling_path,
                device=device,
            ),
            requested,
        )
    if requested == "ebqwen_vl":
        return _validated_backend(
            EBQwenJerseyAnalyzer(profiling_path=profiling_path, device=device),
            requested,
        )
    if requested == "smolvlm2_jersey":
        return _validated_backend(
            SmolVLM2JerseyAnalyzer(profiling_path=profiling_path, device=device),
            requested,
        )
    if requested == "trocr_small_printed":
        return _validated_backend(
            TrOCRSmallPrintedJerseyAnalyzer(
                profiling_path=profiling_path,
                device=device,
            ),
            requested,
        )
    if requested == "parseq_soccernet":
        return _validated_backend(
            SoccerNetPARSeqJerseyAnalyzer(
                profiling_path=profiling_path,
                device=device,
            ),
            requested,
        )
    if requested == "parseq_soccernet_lit":
        return _validated_backend(
            LiteraturePARSeqJerseyAnalyzer(
                profiling_path=profiling_path,
                device=device,
            ),
            requested,
        )

    temporal_path = _project_path(
        project_root,
        os.getenv("BASKETBALL_JERSEY_TEMPORAL_CHECKPOINT", ""),
        project_root / "weights" / "best_temporal.pt",
    )
    uncertainty_path = _project_path(
        project_root,
        os.getenv("BASKETBALL_JERSEY_VIT_CHECKPOINT", ""),
        project_root / "weights" / "jersey_uncertainty_vit_b.pt",
    )
    legacy_path = _project_path(
        project_root,
        os.getenv("BASKETBALL_JERSEY_LEGACY_CHECKPOINT", ""),
        project_root / "weights" / "jersey_number_recon.pt",
    )

    if requested == "best_temporal" or (requested == "auto" and temporal_path.exists()):
        if not temporal_path.exists():
            raise FileNotFoundError(
                f"Requested best_temporal Jersey checkpoint is missing: {temporal_path}"
            )
        return _validated_backend(
            PerformanceTemporalJerseyAnalyzer(
                profiling_path=profiling_path,
                device=device,
                model_path=str(temporal_path),
                sequence_length=8,
            ),
            requested,
        )
    if requested == "uncertainty_vit_b" or (
        requested == "auto" and uncertainty_path.exists()
    ):
        if not uncertainty_path.exists():
            raise FileNotFoundError(
                f"Requested uncertainty_vit_b Jersey checkpoint is missing: {uncertainty_path}"
            )
        return _validated_backend(
            PerformanceUncertaintyJerseyAnalyzer(
                profiling_path=profiling_path,
                device=device,
                model_path=str(uncertainty_path),
                sequence_length=8,
            ),
            requested,
        )
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy Jersey checkpoint not found: {legacy_path}")
    visual = PerformanceJerseyAnalyzer(
        profiling_path=profiling_path,
        device=device,
        model_path=str(legacy_path),
        sequence_length=8,
    )
    visual.checkpoint_diagnostics = dict(
        getattr(visual, "checkpoint_diagnostics", {}) or {}
    )
    visual.checkpoint_diagnostics.setdefault("backend", "legacy_digit_heads")
    visual.checkpoint_diagnostics["fallback_reason"] = (
        "temporal_and_uncertainty_checkpoints_missing"
        if requested == "auto"
        else "explicit_legacy_backend"
    )
    return _validated_backend(visual, requested)
