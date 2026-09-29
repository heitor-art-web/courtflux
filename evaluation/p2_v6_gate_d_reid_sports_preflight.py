from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing the raw-track wrapper first applies the upstream crop-source patch.
from evaluation import p2_v6_gate_d_reid_preflight_rawtrack as rawtrack  # noqa: F401
from evaluation import p2_v6_gate_d_reid_preflight as base


SPORTSREID_REF = "87c7d42fbde58cffc6665dd9152bc00f4a03649a"
SPORTSREID_OSNET_SOURCE_URL = (
    "https://raw.githubusercontent.com/shallowlearn/sportsreid/"
    f"{SPORTSREID_REF}/torchreid/models/osnet.py"
)
SPORTSREID_OSNET_GDRIVE_ID = "1To0Ww6_HxU2ITAlb4kQEgYExV-orwit8"
SPORTSREID_CACHE_DIR = REPO_ROOT / "weights" / "reid"
SPORTSREID_SOURCE_PATH = SPORTSREID_CACHE_DIR / f"sportsreid_osnet_{SPORTSREID_REF}.py"
SPORTSREID_WEIGHTS_PATH = SPORTSREID_CACHE_DIR / "sportsreid_osnet_x1_0_soccernet.pth.tar"


def _download_sportsreid_source() -> Path:
    return base._download(SPORTSREID_OSNET_SOURCE_URL, SPORTSREID_SOURCE_PATH)


def _download_sportsreid_weights() -> Path:
    SPORTSREID_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if SPORTSREID_WEIGHTS_PATH.exists() and SPORTSREID_WEIGHTS_PATH.stat().st_size > 0:
        return SPORTSREID_WEIGHTS_PATH
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "SportsReID checkpoint download requires gdown, which is part of the root requirements."
        ) from exc

    print("[ReID preflight] downloading SportsReID OSNet x1.0 SoccerNet weights...", flush=True)
    result = gdown.download(
        id=SPORTSREID_OSNET_GDRIVE_ID,
        output=str(SPORTSREID_WEIGHTS_PATH),
        quiet=False,
    )
    if not result or not SPORTSREID_WEIGHTS_PATH.exists() or SPORTSREID_WEIGHTS_PATH.stat().st_size == 0:
        raise RuntimeError("SportsReID checkpoint download failed")
    return SPORTSREID_WEIGHTS_PATH


def _load_sportsreid_source() -> ModuleType:
    source_path = _download_sportsreid_source()
    spec = importlib.util.spec_from_file_location("courtflux_sportsreid_osnet", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pinned SportsReID OSNet source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported SportsReID checkpoint type")
    for key in ("state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _load_checkpoint(model: torch.nn.Module, weights_path: Path) -> tuple[int, int]:
    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state = _checkpoint_state(checkpoint)
    model_state = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    discarded = 0
    for raw_key, value in state.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[7:]
        if key in model_state and hasattr(value, "shape") and model_state[key].shape == value.shape:
            matched[key] = value
        else:
            discarded += 1
    if len(matched) < 100:
        raise RuntimeError(
            f"SportsReID checkpoint matched only {len(matched)} tensors; refusing incorrect load"
        )
    model.load_state_dict(matched, strict=False)
    return len(matched), discarded


def _sports_build_model(
    device: torch.device,
    backend: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if backend != "sportsreid_osnet_x1_0_soccernet":
        return _ORIGINAL_BUILD_MODEL(device, backend)

    module = _load_sportsreid_source()
    weights_path = _download_sportsreid_weights()
    print("[ReID preflight] loading SportsReID OSNet x1.0 SoccerNet weights...", flush=True)
    model = module.osnet_x1_0(num_classes=1, pretrained=False)
    matched, discarded = _load_checkpoint(model, weights_path)
    model = model.to(device).eval()
    return model, {
        "backend": "sportsreid_osnet_x1_0_soccernet",
        "person_reid_specific": True,
        "sports_broadcast_specific": True,
        "training_domain": "SoccerNet Re-Identification train split",
        "source_ref": SPORTSREID_REF,
        "weights": str(weights_path),
        "weights_sha256": _sha256(weights_path),
        "google_drive_file_id": SPORTSREID_OSNET_GDRIVE_ID,
        "matched_checkpoint_tensors": matched,
        "discarded_checkpoint_tensors": discarded,
        "input_size": [256, 128],
        "device": str(device),
    }


_ORIGINAL_BUILD_MODEL = base._build_model
base._build_model = _sports_build_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V6 Gate-D SportsReID preflight on frozen human SAME/DIFFERENT pairs. "
            "Uses raw-track geometry and a SoccerNet-trained OSNet model; diagnostic only."
        )
    )
    parser.add_argument("--gt", default=str(base.DEFAULT_GT))
    parser.add_argument("--states", default=None)
    parser.add_argument("--video", default=str(base.DEFAULT_VIDEO))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--backend",
        default="sportsreid_osnet_x1_0_soccernet",
        choices=["sportsreid_osnet_x1_0_soccernet"],
    )
    parser.add_argument("--window-frames", type=int, default=base.DEFAULT_WINDOW_FRAMES)
    parser.add_argument(
        "--samples-per-observation",
        type=int,
        default=base.DEFAULT_SAMPLES_PER_OBSERVATION,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.window_frames < 0:
        raise SystemExit("--window-frames must be >= 0")
    if args.samples_per_observation < 1:
        raise SystemExit("--samples-per-observation must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    if args.output is None:
        states_path = (
            Path(args.states).expanduser().resolve()
            if args.states
            else base._latest_states_file()
        )
        args.output = str(states_path.parent / "p2_v6_gate_d_sportsreid_preflight.json")
    base.run(args)


if __name__ == "__main__":
    main()
