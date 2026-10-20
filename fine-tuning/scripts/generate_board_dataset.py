"""Generate FEN → board description training dataset (Alpaca JSONL format).

Streams from data/{easy,medium,hard,extreme}.json, 5,000 samples per tier.
Output: data/board_dataset.jsonl

Usage:
    python fine-tuning/scripts/generate_board_dataset.py
    python fine-tuning/scripts/generate_board_dataset.py --n-samples 50 --dry-run
    python fine-tuning/scripts/generate_board_dataset.py --n-samples 20000 --output data/board_dataset.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_SRC))

from board_state import fen_to_board_description
from src.utils import setup_logging

INSTRUCTION = (
    "You are given a chess position in FEN notation. "
    "List every piece on the board and its exact square."
)

TIERS = ("easy", "medium", "hard", "extreme")
SAMPLES_PER_TIER = 5000


def generate_records(
    raw_records: list[dict],
    seed: int = 42,
    max_samples: int | None = None,
) -> Iterator[dict]:
    """Yield Alpaca JSONL records from a list of raw position dicts.

    Deduplicates on FEN. Shuffles with seed before yielding.

    Args:
        raw_records: List of position dicts with a 'fen' key.
        seed: Random seed for reproducible shuffling.
        max_samples: Cap on output records (None = no cap).

    Yields:
        Dict with keys: instruction, input, output.
    """
    rng = random.Random(seed)
    shuffled = list(raw_records)
    rng.shuffle(shuffled)

    seen_fens: set[str] = set()
    count = 0

    for record in shuffled:
        if max_samples is not None and count >= max_samples:
            break

        fen = record.get("fen", "").strip()
        if not fen or fen in seen_fens:
            continue
        seen_fens.add(fen)

        try:
            output = fen_to_board_description(fen)
        except Exception:
            continue

        yield {
            "instruction": INSTRUCTION,
            "input": f"FEN: {fen}",
            "output": output,
        }
        count += 1


def load_tier(data_dir: Path, tier: str, n: int, seed: int) -> list[dict]:
    """Load up to n records from a tier JSON file."""
    path = data_dir / f"{tier}.json"
    if not path.exists():
        raise FileNotFoundError(f"Tier file not found: {path}")

    with open(path) as f:
        records = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FEN → board description training dataset."
    )
    parser.add_argument(
        "--n-samples", type=int, default=20000,
        help="Total number of training samples (default: 20000, 5000 per tier)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", default="data/board_dataset.jsonl",
        help="Output JSONL path (default: data/board_dataset.jsonl)",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing tier JSON files (default: data)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first 5 samples to stdout without writing any file",
    )
    args = parser.parse_args()

    logger = setup_logging("generate_board_dataset")
    data_dir = Path(args.data_dir)
    per_tier = args.n_samples // len(TIERS)

    logger.info(f"Loading {per_tier} samples per tier from {data_dir}/")
    all_records: list[dict] = []
    for tier in TIERS:
        try:
            records = load_tier(data_dir, tier, per_tier, args.seed)
            logger.info(f"  {tier}: {len(records)} records loaded")
            all_records.extend(records)
        except FileNotFoundError as e:
            logger.warning(str(e))

    if not all_records:
        logger.error("No records loaded — check data directory.")
        sys.exit(1)

    logger.info(f"Generating records from {len(all_records)} positions...")
    generated = list(generate_records(all_records, seed=args.seed))
    logger.info(f"Generated {len(generated)} unique records after deduplication.")

    if args.dry_run:
        for rec in generated[:5]:
            print(json.dumps(rec, indent=2))
        logger.info("[dry-run] No file written.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for rec in generated:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"Wrote {len(generated)} records to {output_path}")


if __name__ == "__main__":
    main()
