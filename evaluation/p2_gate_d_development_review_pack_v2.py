from __future__ import annotations

import argparse
from pathlib import Path

from evaluation import p2_gate_d_development_review_pack as base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for the current team-agnostic Gate-D development review pack."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base.PROJECT_ROOT / "evaluation" / "runs" / "gate_d_development_review_pack",
    )
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help=(
            "Generate Jersey-OFF states only when neither historical states nor states "
            "recorded by an existing review_pack.json are available locally."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
        help="Device used only when --generate-missing actually has to run a probe.",
    )
    args = parser.parse_args()
    base.run(
        args.output.expanduser().resolve(),
        generate_missing=bool(args.generate_missing),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
