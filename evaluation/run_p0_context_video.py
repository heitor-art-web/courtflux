from __future__ import annotations

import argparse

from core.config import settings
import services.analytics.analyzer as analyzer
from services.p0_context_pipeline import Pipeline as ContextPipeline
from services.vision.keyframed_camera_tracker import CameraTracker as KeyframedCameraTracker


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing CourtFlux detection-only pipeline with diagnostic OC-SORT "
            "court/player context and periodic court keypoint refresh."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", default="0")
    parser.add_argument("--end", default="")
    parser.add_argument("--model", default=settings.model)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=settings.device, choices=["mps", "cpu", "cuda"])
    parser.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    parser.add_argument("--match-config", default=str(analyzer.DEFAULT_MATCH_CONFIG_PATH))
    parser.add_argument("--no-jersey", action="store_true", default=True)
    parser.add_argument("--jersey-interval", type=int, default=12)
    parser.add_argument("--diagnostics-batch-size", type=int, default=100)
    args = parser.parse_args()

    # Experimental wiring only: do not alter the production/default analyzer imports.
    analyzer.CameraTracker = KeyframedCameraTracker
    analyzer.Pipeline = ContextPipeline
    analyzer.main(
        video_source=args.video,
        start_time=analyzer.parse_time_to_seconds(args.start),
        end_time=analyzer.parse_time_to_seconds(args.end) if args.end else None,
        model_path=args.model,
        detector_conf=float(args.conf),
        imgsz=int(args.imgsz),
        device=args.device,
        tracker=args.tracker,
        reset_db=False,
        match_config_path=args.match_config,
        detection_only=True,
        enable_jersey=not bool(args.no_jersey),
        jersey_interval_frames=int(args.jersey_interval),
        diagnostics_batch_size=int(args.diagnostics_batch_size),
        persist_results=False,
    )


if __name__ == "__main__":
    main()
