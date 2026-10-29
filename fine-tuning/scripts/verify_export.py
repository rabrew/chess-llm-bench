"""Check that an imported fine-tuned model actually learned the board task.

Asks the Ollama fine-tuned model to read the first N held-out positions with the
training prompt. Exits 1 if mean recall is below MIN_EXPORT_RECALL, or, for a
legal-moves model, if legal-move recall or precision is below MIN_EXPORT_LEGAL,
so a broken or undertrained export fails the batch. See
fine-tuning/specs/gguf-export-check.md.

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
from board_state import verify_board_description, verify_legal_moves  # noqa: E402
from finetune_model import load_dataset_records, load_ft_config, split_held_out  # noqa: E402
from import_finetuned_to_ollama import find_model_entry  # noqa: E402
from src.utils import setup_logging  # noqa: E402

MIN_EXPORT_RECALL = 0.80
MIN_EXPORT_LEGAL = 0.9  # criterion 0 in fine-tuning/specs/legal-moves-finetuning.md
DEFAULT_SAMPLE = 20
LEGAL_LINE = "\nLegal moves:"


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("No values to check")
    return sum(values) / len(values)


def export_passes(recalls: list[float], threshold: float) -> bool:
    return _mean(recalls) >= threshold


def legal_passes(recalls: list[float], precisions: list[float], threshold: float) -> bool:
    """Legal-move recall and precision must both reach the threshold on average."""
    return _mean(recalls) >= threshold and _mean(precisions) >= threshold


def detect_task(records: list[dict]) -> str:
    """'board+legal' when the training targets include a legal-move line, else 'board'."""
    if not records:
        raise ValueError("Empty dataset: cannot tell which task the model was trained on")
    return "board+legal" if LEGAL_LINE in records[0]["output"] else "board"


def sample_positions(held_out: list[dict], n: int) -> list[dict]:
    if n <= 0:
        raise ValueError(f"Sample size must be positive, got {n}")
    return held_out[:n]


def score_model(model: str, positions: list[dict], host: str, task: str = "board") -> list[dict]:
    """Per position: board recall, plus legal-move recall and precision for board+legal."""
    scores = []
    for position in positions:
        fen = position["input"].replace("FEN: ", "").strip()
        reply = query_ollama(model, build_prompt(position, "training", task), host, task=task)
        score = {"recall": verify_board_description(fen, reply)["recall"]}
        if task == "board+legal":
            legal = verify_legal_moves(fen, reply)
            score.update(legal_recall=legal["recall"], legal_precision=legal["precision"])
        scores.append(score)
    return scores


def _legal_verdict(scores: list[dict]) -> tuple[bool, str]:
    recalls = [s["legal_recall"] for s in scores]
    precisions = [s["legal_precision"] for s in scores]
    passed = legal_passes(recalls, precisions, MIN_EXPORT_LEGAL)
    return passed, (f"; legal moves recall {_mean(recalls):.3f} precision {_mean(precisions):.3f} — "
                    f"{'PASS' if passed else 'FAIL'} (threshold {MIN_EXPORT_LEGAL})")


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
    records = load_dataset_records(config["training"]["dataset_path"])
    task = detect_task(records)
    _, held_out = split_held_out(records, seed=SEED, n=HELD_OUT_N)
    positions = sample_positions(held_out, args.n)

    scores = score_model(ft_tag, positions, args.host, task)
    recalls = [s["recall"] for s in scores]
    passed = export_passes(recalls, MIN_EXPORT_RECALL)
    legal_ok, legal_note = _legal_verdict(scores) if task == "board+legal" else (True, "")
    logger.info(
        f"{ft_tag}: mean recall {_mean(recalls):.3f} on {len(recalls)} held-out positions, "
        f"{sum(r == 0 for r in recalls)} with nothing right — {'PASS' if passed else 'FAIL'} "
        f"(threshold {MIN_EXPORT_RECALL}){legal_note}"
    )
    return 0 if passed and legal_ok else 1


if __name__ == "__main__":
    sys.exit(main())
