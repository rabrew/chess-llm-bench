#!/usr/bin/env python3
"""CLI script to run benchmark workers."""

import argparse
import gc
import json
import multiprocessing
import signal
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.job_queue import JobQueue
from src.llm_client import OllamaClient
from src.utils import load_config, setup_logging, ensure_dir
from src.worker import run_worker
import src.worker as _worker_module


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    print("\nShutdown requested, finishing current jobs...")
    shutdown_requested = True


def worker_process(args):
    """Worker process entry point."""
    worker_id, config_path, max_jobs, dry_run, model = args
    return run_worker(worker_id, config_path, max_jobs, dry_run, model=model)


def write_run_log(config: dict, run_id: str, log_dir: str) -> None:
    """Write run metadata for reproducibility."""
    log_path = Path(log_dir) / f"run_{run_id}.json"
    ensure_dir(log_dir)

    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "models": config.get("models", []),
        "prompt_formats": config.get("benchmark", {}).get("prompt_formats", []),
        "hypotheses": ["H1", "H2", "H3", "H4", "H5"],
        "scoring_method": "Option A",
        "config_snapshot": config,
        "random_seed": config.get("benchmark", {}).get("random_seed", 42),
        "worker_count": config.get("workers", {}).get("count", 4),
        "stockfish_depth": config.get("stockfish", {}).get("depth", 22),
    }

    with open(log_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():  # pragma: no cover
    # Python 3.14 defaults to forkserver on Linux which deadlocks with our setup.
    # Force fork which is stable and avoids the forkserver handshake deadlock.
    multiprocessing.set_start_method('fork', force=True)
    parser = argparse.ArgumentParser(
        description="Run benchmark workers to process jobs"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process jobs without writing results",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Maximum jobs per worker (for testing)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show queue status and exit",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Only process jobs for this specific model",
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
    logger = setup_logging(level=level)

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Get paths
    paths = config.get("paths", {})
    db_path = paths.get("jobs_db", "jobs/db/jobs.db")
    log_dir = paths.get("logs_dir", "results/logs")

    # Initialize job queue
    job_queue = JobQueue(db_path)

    if args.status:
        # Show status and exit
        progress = job_queue.get_progress()
        print("\nJob Queue Status:")
        print(f"  Total jobs: {progress['total']}")
        print(f"  Pending: {progress['pending']}")
        print(f"  In progress: {progress['in_progress']}")
        print(f"  Done: {progress['done']}")
        print(f"  Failed: {progress['failed']}")
        print(f"  Progress: {progress['percent_complete']:.1f}%")
        return

    # Check Ollama availability
    ollama_config = config.get("ollama", {})
    client = OllamaClient(
        base_url=ollama_config.get("base_url", "http://localhost:11434"),
    )
    if not client.is_available():
        logger.error("Ollama is not running. Please start Ollama first.")
        sys.exit(1)

    # Determine worker count
    num_workers = args.workers or config.get("workers", {}).get("count", 4)

    # Reset jobs left in_progress by previously crashed workers
    stale = job_queue.reset_stale_jobs(timeout_minutes=10)
    if stale:
        logger.info(f"Reset {stale} stale in_progress jobs to pending")

    # Check for pending jobs
    progress = job_queue.get_progress()
    if progress["pending"] == 0:
        print("No pending jobs in queue")
        return

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Write run log
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    write_run_log(config, run_id, log_dir)

    print(f"\nStarting {num_workers} workers...")
    print(f"Pending jobs: {progress['pending']}")
    if args.dry_run:
        print("DRY RUN MODE - results will not be saved")

    # For dry run with limited jobs, use single process
    if args.dry_run and (args.max_jobs or 10) <= 10:
        jobs_processed = run_worker(
            "worker_0",
            args.config,
            args.max_jobs or 10,
            args.dry_run,
            model=args.model,
        )
        print(f"\nProcessed {jobs_processed} jobs")
        return

    # Pre-load only the positions needed by pending jobs into a shared dict.
    # Workers are fork()ed below, so they inherit this dict via COW and never
    # need to independently load the multi-GB JSON tier files.
    data_dir = paths.get("data_dir", "data")
    needed_ids: set[int] = set()
    try:
        conn_tmp = sqlite3.connect(db_path)
        cur_tmp = conn_tmp.cursor()
        cur_tmp.execute(
            "SELECT DISTINCT position_id FROM jobs WHERE status IN ('pending', 'in_progress')"
        )
        needed_ids = {row[0] for row in cur_tmp.fetchall()}
        conn_tmp.close()
    except Exception as e:
        logger.warning(f"Could not query needed position IDs: {e}")

    if needed_ids:
        logger.info(f"Pre-loading {len(needed_ids)} positions before forking workers...")
        loader = DataLoader(data_dir)
        for tier in ["easy", "medium", "hard", "extreme"]:
            tier_positions = loader.load_tier(tier)
            for pos in tier_positions:
                if pos["id"] in needed_ids:
                    _worker_module._SHARED_POSITIONS[pos["id"]] = pos
            loader._cache.pop(tier, None)
            gc.collect()
        del loader
        gc.collect()
        logger.info(f"Pre-loaded {len(_worker_module._SHARED_POSITIONS)} positions ({len(_worker_module._SHARED_POSITIONS) * 100 // max(len(needed_ids), 1)}% of needed)")

    # Close the parent's SQLite connection before forking.  Each worker process
    # must open its own connection; sharing the parent's across a fork()
    # boundary is undefined behaviour and can silently corrupt the database.
    job_queue.close()

    # Run workers in parallel
    worker_args = [
        (f"worker_{i}", args.config, args.max_jobs, args.dry_run, args.model)
        for i in range(num_workers)
    ]

    with multiprocessing.Pool(num_workers) as pool:
        try:
            results = pool.map(worker_process, worker_args)
            total_processed = sum(results)
            print(f"\nTotal jobs processed: {total_processed}")
        except Exception as e:
            logger.error(f"Worker pool error: {e}")
            pool.terminate()
            raise


if __name__ == "__main__":  # pragma: no cover
    main()
