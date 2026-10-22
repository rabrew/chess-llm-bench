"""Check that an imported fine-tuned model actually learned the board task.

Asks the Ollama -board-ft model to read the first N held-out positions with the
training prompt. Exits 1 if mean recall is below MIN_EXPORT_RECALL, so a broken
GGUF export fails the batch. See fine-tuning/specs/gguf-export-check.md.

Usage:
    python fine-tuning/scripts/verify_export.py gemma4:e2b
"""

import argparse
import sys
from pathlib import Path

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT / "scripts"))
sys.path.insert(0, str(FT_ROOT / "src"))

from board_reading_test import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OLLAMA,
    HELD_OUT_N,
    SEED,
    build_prompt,
    query_ollama,
)
from board_state import verify_board_description  # noqa: E402
from finetune_model import load_dataset_records, load_ft_config, split_held_out  # noqa: E402
from import_finetuned_to_ollama import find_model_entry  # noqa: E402
from src.utils import setup_logging  # noqa: E402

MIN_EXPORT_RECALL = 0.80
DEFAULT_SAMPLE = 20


def export_passes(recalls: list[float], threshold: float) -> bool:
    if not recalls:
        raise ValueError("No recall values to check")
    return sum(recalls) / len(recalls) >= threshold


def sample_positions(held_out: list[dict], n: int) -> list[dict]:
    if n <= 0:
        raise ValueError(f"Sample size must be positive, got {n}")
    return held_out[:n]


def score_model(model: str, positions: list[dict], host: str) -> tuple[list[float], int]:
    """Return each position's recall and how many positions scored zero."""
    recalls = []
    for position in positions:
        fen = position["input"].replace("FEN: ", "").strip()
        result = verify_board_description(fen, query_ollama(model, build_prompt(position, "training"), host))
        recalls.append(result["recall"])
    return recalls, sum(recall == 0 for recall in recalls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ollama_base")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=DEFAULT_OLLAMA)
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()
    logger = setup_logging("verify_export")

    config = load_ft_config(args.config)
    ft_tag = find_model_entry(config, args.ollama_base)["ollama_ft_tag"]
    _, held_out = split_held_out(load_dataset_records(config["training"]["dataset_path"]), seed=SEED, n=HELD_OUT_N)
    positions = sample_positions(held_out, args.n)

    recalls, zero_recall = score_model(ft_tag, positions, args.host)
    mean_recall = sum(recalls) / len(recalls)
    passed = export_passes(recalls, MIN_EXPORT_RECALL)
    logger.info(
        f"{ft_tag}: mean recall {mean_recall:.3f} on {len(recalls)} held-out positions, "
        f"{zero_recall} with nothing right — {'PASS' if passed else 'FAIL'} (threshold {MIN_EXPORT_RECALL})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
