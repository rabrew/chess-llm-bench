#!/usr/bin/env python3
"""CLI script to generate benchmark jobs."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_generator import populate_job_queue, estimate_job_count
from src.utils import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark jobs and populate the job queue"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Only estimate job count, don't create jobs",
    )
    parser.add_argument(
        "--tier",
        choices=["easy", "medium", "hard", "extreme"],
        default=None,
        help="Only generate jobs for a specific difficulty tier",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Only generate jobs for a specific model",
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

    if args.estimate:
        # Just estimate
        estimate = estimate_job_count(config)
        print("\nJob count estimate:")
        print(f"  Positions: {estimate['positions']}")
        print(f"  Models: {estimate['models']}")
        print(f"  Prompt formats: {estimate['prompt_formats']}")
        print(f"  Standard jobs: {estimate['standard_jobs']}")
        print(f"  Correction jobs (est.): {estimate['correction_estimate']}")
        print(f"  Total (est.): {estimate['total_estimate']}")
    else:
        # Generate jobs
        tier_msg = f" for tier: {args.tier}" if args.tier else ""
        model_msg = f", model: {args.model}" if args.model else ""
        logger.info(f"Generating benchmark jobs{tier_msg}{model_msg}...")
        inserted = populate_job_queue(config, tier=args.tier, model=args.model)
        print(f"\nInserted {inserted} jobs into the queue")


if __name__ == "__main__":
    main()
