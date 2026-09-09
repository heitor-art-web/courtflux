from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def download_dataset(workspace: str, project: str, version: int, location: Path, api_key: str) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            "Roboflow SDK is optional. Install it with: python3 -m pip install -r requirements-roboflow.txt"
        ) from exc

    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(int(version))
        .download("coco", location=str(location))
    )
    return Path(dataset.location).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public Roboflow datasets used by the CourtFlux system-quality-gate experiments."
    )
    parser.add_argument("--root", default="data/external/roboflow")
    parser.add_argument(
        "--only",
        choices=["all", "ball", "court"],
        default="all",
        help="Download the basketball detection dataset, court-keypoint dataset, or both.",
    )
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_PRIVATE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_PRIVATE_API_KEY is required for Roboflow dataset downloads. "
            "Use the private API key from the Roboflow dashboard; do not use the publishable inference key."
        )

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.only in {"all", "ball"}:
        ball_path = download_dataset(
            "roboflow-jvuqo",
            "basketball-player-detection-3-ycjdo",
            18,
            root / "basketball_player_detection_v18",
            api_key,
        )
        print(f"[ROBOFLOW BALL DATASET] {ball_path}")

    if args.only in {"all", "court"}:
        court_path = download_dataset(
            "roboflow-jvuqo",
            "basketball-court-detection-2",
            19,
            root / "basketball_court_detection_v19",
            api_key,
        )
        print(f"[ROBOFLOW COURT DATASET] {court_path}")


if __name__ == "__main__":
    main()
