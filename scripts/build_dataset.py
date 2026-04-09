#!/usr/bin/env python3
"""CLI script to build chess position datasets."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_builder import build_dataset
from src.utils import load_config, setup_logging


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Build chess position datasets from multiple sources"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)
    logger = logging.getLogger("chess_llm_bench")

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Print header
    print(f"\n{'='*60}")
    print(f"  DATASET BUILDER")
    print(f"  Source: Lichess Puzzle Database")
    print(f"{'='*60}\n")

    # Build dataset
    dataset = build_dataset(config, args.output_dir)

    # Print summary
    total = sum(len(positions) for positions in dataset.values())

    print(f"\n{'='*60}")
    print(f"  DATASET COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Total positions: {total:,}\n")
    print(f"  {'Tier':<12} {'Count':>10}")
    print(f"  {'-'*22}")
    for tier in ["easy", "medium", "hard", "extreme"]:
        count = len(dataset.get(tier, []))
        bar = "█" * (count // 50000) if count > 0 else ""
        print(f"  {tier:<12} {count:>10,}  {bar}")
    print()


if __name__ == "__main__":  # pragma: no cover
    main()
