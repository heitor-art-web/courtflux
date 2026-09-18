from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from supabase import create_client
except Exception:  # pragma: no cover - runtime dependency guard
    create_client = None

from services.p3.materializer import P3Materializer


def _client():
    url = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key or create_client is None:
        raise RuntimeError(
            "P3 materialization requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
        )
    return create_client(url, key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize CourtFlux P3 Things to Review from a canonical P2 events JSONL "
            "without re-running or mutating the P2 engine."
        )
    )
    parser.add_argument("--events", required=True, help="Canonical P2 events JSONL")
    parser.add_argument("--source-video", required=True, help="Local source-video path")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="evaluation/p3_materialized_clips",
        help="Worker-local scratch directory for generated clips",
    )
    parser.add_argument("--p2-runtime-version", default=None)
    parser.add_argument(
        "--keep-local-clips",
        action="store_true",
        help="Keep local clip bytes after successful durable upload",
    )
    args = parser.parse_args()

    result = P3Materializer(_client()).materialize_p2_jsonl(
        Path(args.events),
        workspace_id=args.workspace_id,
        game_id=args.game_id,
        analysis_run_id=args.analysis_run_id,
        source_video_path=Path(args.source_video),
        output_dir=Path(args.output_dir),
        p2_runtime_version=args.p2_runtime_version,
        cleanup_local_after_upload=not args.keep_local_clips,
        provenance={"entrypoint": "services.p3.materialize_cli"},
    )
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()