"""Count a model's unfinished jobs (pending or in_progress) in the jobs DB.

Used by bin/run_all.sh to tell "queue exhausted" apart from "tier complete".
Opens the DB read-only so a wrong path can never create an empty DB, and exits
1 on any read error instead of reporting 0 (specs/run-all-tier-tracking.md).

Usage:
    python scripts/pending_jobs.py --db jobs/db/jobs.db --model llama3.2:3b
"""

import argparse
import sqlite3
import sys
from pathlib import Path

UNFINISHED_STATUSES = ("pending", "in_progress")


def count_pending(db_path: str, model: str) -> int:
    """Number of the model's jobs that are pending or in progress."""
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"jobs DB not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
    try:
        placeholders = ", ".join("?" for _ in UNFINISHED_STATUSES)
        (count,) = conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE model = ? AND status IN ({placeholders})",
            (model, *UNFINISHED_STATUSES),
        ).fetchone()
    finally:
        conn.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Count a model's unfinished jobs.")
    parser.add_argument("--db", required=True, help="Path to the jobs SQLite DB")
    parser.add_argument("--model", required=True, help="Model tag")
    args = parser.parse_args()
    try:
        print(count_pending(args.db, args.model))
    except (OSError, sqlite3.Error) as exc:
        print(f"Cannot read pending jobs from {args.db}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
