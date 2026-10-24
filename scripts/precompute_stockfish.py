#!/usr/bin/env python3
"""CLI script to pre-compute Stockfish evaluations for all positions (parallel)."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging


def evaluate_position(args):
    """Evaluate a single position with Stockfish (worker function)."""
    fen, stockfish_path, depth = args

    # Import here to avoid issues with multiprocessing
    from src.engine_wrapper import StockfishEngine

    try:
        engine = StockfishEngine(path=stockfish_path, depth=depth, threads=1)
        result = engine.evaluate(fen)
        engine.close()
        return fen, result["eval"], result["best_move"]
    except Exception as exc:  # one bad position must not stop the pool; log it and leave it unset
        print(f"Stockfish failed on {fen}: {exc}", file=sys.stderr)
        return fen, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Stockfish evaluations for all dataset positions (parallel)"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing dataset JSON files",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Override Stockfish search depth",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel Stockfish workers",
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

    # Get Stockfish settings
    stockfish_config = config.get("stockfish", {})
    depth = args.depth or stockfish_config.get("depth", 22)
    stockfish_path = stockfish_config.get("path", "/usr/games/stockfish")
    n_workers = args.workers or stockfish_config.get("workers", mp.cpu_count())

    logger.info(f"Using {n_workers} parallel Stockfish workers at depth {depth}")

    # Collect all positions that need evaluation
    data_dir = Path(args.data_dir)
    tiers = ["easy", "medium", "hard", "extreme"]

    all_positions = []  # (tier, index, fen)
    tier_data = {}  # tier -> positions list

    for tier in tiers:
        file_path = data_dir / f"{tier}.json"
        if not file_path.exists():
            logger.warning(f"Dataset file not found: {file_path}")
            continue

        with open(file_path, "r") as f:
            positions = json.load(f)

        tier_data[tier] = positions

        for i, pos in enumerate(positions):
            if "stockfish_eval" not in pos or "stockfish_best_move" not in pos:
                all_positions.append((tier, i, pos["fen"]))

    if not all_positions:
        print("All positions already have Stockfish evaluations.")
        return

    logger.info(f"Evaluating {len(all_positions)} positions...")

    # Prepare work items
    work_items = [(fen, stockfish_path, depth) for (tier, idx, fen) in all_positions]

    # Process in parallel with detailed progress
    results = {}
    failed = 0

    print(f"\n{'='*60}")
    print("  STOCKFISH EVALUATION")
    print(f"  Positions: {len(all_positions):,}")
    print(f"  Workers:   {n_workers}")
    print(f"  Depth:     {depth}")
    print(f"{'='*60}\n")

    # Save checkpoint every N positions OR every M minutes
    import time
    CHECKPOINT_EVERY = 1000
    CHECKPOINT_MINUTES = 45
    last_checkpoint_count = 0
    last_checkpoint_time = time.time()

    def save_checkpoint():
        """Save current progress to JSON files."""
        for tier, idx, fen in all_positions:
            if fen in results:
                eval_score, best_move = results[fen]
                tier_data[tier][idx]["stockfish_eval"] = eval_score
                tier_data[tier][idx]["stockfish_best_move"] = best_move

        for tier, positions in tier_data.items():
            file_path = data_dir / f"{tier}.json"
            with open(file_path, "w") as f:
                json.dump(positions, f, indent=2)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_position, item): item[0] for item in work_items}

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Evaluating",
            unit="pos",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for future in pbar:
            fen, eval_score, best_move = future.result()
            if eval_score is not None:
                results[fen] = (eval_score, best_move)
            else:
                failed += 1

            # Update progress bar description with stats
            pbar.set_postfix({"valid": len(results), "failed": failed})

            # Save checkpoint periodically (by count OR by time)
            time_since_checkpoint = (time.time() - last_checkpoint_time) / 60
            count_since_checkpoint = len(results) - last_checkpoint_count

            if count_since_checkpoint >= CHECKPOINT_EVERY or time_since_checkpoint >= CHECKPOINT_MINUTES:
                pbar.set_description("Saving checkpoint")
                save_checkpoint()
                last_checkpoint_count = len(results)
                last_checkpoint_time = time.time()
                pbar.set_description("Evaluating")
                logger.info(f"Checkpoint saved: {len(results):,} positions ({time_since_checkpoint:.1f} min since last)")

    # Final save
    save_checkpoint()
    for tier in tier_data:
        logger.info(f"Updated {data_dir / f'{tier}.json'}")

    print(f"\nPre-computed evaluations for {len(results)} positions")


if __name__ == "__main__":
    main()
