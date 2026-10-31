"""Results needed per model per tier before run_all.sh calls a tier finished (specs/run-all-tier-tracking.md).

    python scripts/tier_target.py --config config/config_plain_board.yaml
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config  # noqa: E402

# The full benchmark always treated 5,900 of 6,000 results as complete: jobs that fail every attempt.
COMPLETE_SHARE = 5900 / 6000


def results_needed_per_tier(benchmark: dict[str, Any]) -> int:
    """floor(positions per tier × prompt formats × COMPLETE_SHARE); raises ValueError on a bad config."""
    positions = benchmark.get("sample_per_tier") or benchmark.get("max_positions_per_tier")
    formats = benchmark.get("prompt_formats")
    if not isinstance(positions, int) or positions <= 0:
        raise ValueError("benchmark needs a positive sample_per_tier or max_positions_per_tier")
    if not formats:
        raise ValueError("benchmark.prompt_formats is missing or empty")
    return math.floor(positions * len(formats) * COMPLETE_SHARE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        print(results_needed_per_tier(load_config(args.config).get("benchmark", {})))
    except (OSError, ValueError) as exc:
        sys.exit(f"tier_target: {exc}")


if __name__ == "__main__":
    main()
