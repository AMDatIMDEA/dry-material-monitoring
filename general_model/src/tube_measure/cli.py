from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config
from .pipeline import InferencePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tube-measure",
        description="Measure material fill from Ultralytics YOLO segmentation masks.",
    )
    parser.add_argument("--model", required=True, type=Path, help="YOLO segmentation weights")
    parser.add_argument("--source", required=True, type=Path, help="Image, folder, or video")
    parser.add_argument("--config", required=True, type=Path, help="YAML configuration")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--conf", type=float, default=None, help="Override model confidence")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device override: auto, cpu, cuda, or cuda:<index>",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.conf is not None and not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")
    config = load_config(args.config)
    pipeline = InferencePipeline(
        model_path=args.model,
        config=config,
        confidence=args.conf,
        device=args.device,
    )
    records = pipeline.run(args.source, args.output)
    tube_count = sum(len(record["analysis"]["tubes"]) for record in records)
    print(f"Processed {len(records)} image(s)/frame(s); wrote {tube_count} tube records")
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
