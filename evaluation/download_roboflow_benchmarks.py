from __future__ import annotations

import argparse
from pathlib import Path


def download_dataset(workspace: str, project: str, version: int, location: Path) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            "Roboflow SDK is optional. Install it with: python3 -m pip install -r requirements-roboflow.txt"
        ) from exc

    rf = Roboflow()
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

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.only in {"all", "ball"}:
        ball_path = download_dataset(
            "roboflow-jvuqo",
            "basketball-player-detection-3-ycjdo",
            18,
            root / "basketball_player_detection_v18",
        )
        print(f"[ROBOFLOW BALL DATASET] {ball_path}")

    if args.only in {"all", "court"}:
        court_path = download_dataset(
            "roboflow-jvuqo",
            "basketball-court-detection-2",
            19,
            root / "basketball_court_detection_v19",
        )
        print(f"[ROBOFLOW COURT DATASET] {court_path}")


if __name__ == "__main__":
    main()
