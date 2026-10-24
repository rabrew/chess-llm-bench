"""Before/after board-reading test on the Ollama models (F-9).

Asks each base model and its -board-ft version to list the pieces on the same
200 held-out positions, marks the answers with verify_board_description, and
saves every answer. See fine-tuning/specs/board-reading-test.md.

Usage:
    python fine-tuning/scripts/board_reading_test.py
    python fine-tuning/scripts/board_reading_test.py --models qwen2.5:7b mistral:7b
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_ROOT / "src"))
sys.path.insert(0, str(FT_ROOT / "scripts"))

from board_state import parse_board_description, verify_board_description, verify_legal_moves  # noqa: E402
from finetune_model import (  # noqa: E402
    INSTRUCTION,
    format_alpaca_prompt,
    load_dataset_records,
    load_ft_config,
    split_held_out,
)
from generate_board_dataset import LEGAL_INSTRUCTION  # noqa: E402
from src.utils import setup_logging  # noqa: E402

DEFAULT_CONFIG = "fine-tuning/config/finetune_config.yaml"
OUT_DIR = Path("results/finetuning_experiment")
PROMPT_STYLES = ("format", "training")
DEFAULT_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not DEFAULT_OLLAMA.startswith("http"):
    DEFAULT_OLLAMA = f"http://{DEFAULT_OLLAMA}"

HELD_OUT_N = 200
SEED = 42
NUM_CTX = 1024
NUM_PREDICT = 300
# board+legal answers carry ~250 extra tokens of legal moves (legal-moves-finetuning.md)
NUM_PREDICT_BY_TASK = {"board": NUM_PREDICT, "board+legal": 700}
TASKS = tuple(NUM_PREDICT_BY_TASK)
LEGAL_FORMAT_LINE = "\nLegal moves: <move>, <move>, ... (every legal move for the side to move, in SAN)"
STOP_SEQUENCES = ["\n###"]
REQUEST_TIMEOUT_S = 300
MAX_ATTEMPTS = 3
RETRY_WAIT_S = 5.0
Z_95 = 1.96

# Placeholders only: a concrete example board gets copied verbatim by base models.
FORMAT_HINT = (
    " Answer in exactly this format, with no other text. Replace each <square> with a real square:\n"
    "White: King <square>, Queen <square>, Rooks <square> <square>, Pawns <square> <square> ...\n"
    "Black: King <square>, Knights <square> <square>, Pawns <square> ...\n"
    "Side to move: <White or Black>\n"
    "List only the piece types that are on the board."
)


def _check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")


def build_prompt(position: dict, style: str = "format", task: str = "board") -> str:
    """Same prompt text for every model.

    "format": training instruction + answer-format description + FEN (main run).
    "training": the exact training prompt, instruction + FEN only.
    task "board+legal" uses the legal-moves instruction and asks for the legal-move line.
    """
    if style not in PROMPT_STYLES:
        raise ValueError(f"prompt style must be one of {PROMPT_STYLES}, got {style!r}")
    _check_task(task)
    base_instruction = LEGAL_INSTRUCTION if task == "board+legal" else INSTRUCTION
    hint = FORMAT_HINT + (LEGAL_FORMAT_LINE if task == "board+legal" else "")
    instruction = base_instruction + hint if style == "format" else base_instruction
    return format_alpaca_prompt(instruction, position["input"])


def output_paths(style: str, task: str = "board") -> tuple[Path, Path]:
    prefix = "board_legal_t0" if task == "board+legal" else "board_t0"
    stem = prefix if style == "format" else f"{prefix}_training_prompt"
    return OUT_DIR / f"{stem}.jsonl", OUT_DIR / f"{stem}_summary.csv"


def build_payload(model: str, prompt: str, task: str = "board") -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "raw": True,  # skip each model's chat template so every model sees identical text
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": SEED,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT_BY_TASK[task],
            "stop": STOP_SEQUENCES,
        },
    }


def query_ollama(
    model: str,
    prompt: str,
    host: str,
    post: Callable[..., Any] = requests.post,
    retry_wait_s: float = RETRY_WAIT_S,
    task: str = "board",
) -> str:
    """Return the model's response text, retrying transient failures."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = post(f"{host}/api/generate", json=build_payload(model, prompt, task), timeout=REQUEST_TIMEOUT_S)
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(retry_wait_s)
    raise AssertionError("unreachable")


def score_row(
    model: str, base_model: str, fine_tuned: bool, idx: int, fen: str,
    response: str, seconds: float, digest: str, task: str = "board",
) -> dict[str, Any]:
    verdict = verify_board_description(fen, response)
    row = {
        "model": model,
        "base_model": base_model,
        "fine_tuned": fine_tuned,
        "idx": idx,
        "fen": fen,
        "response": response,
        "parsed": bool(parse_board_description(response)),
        "correct": verdict["correct"],
        "recall": verdict["recall"],
        "precision": verdict["precision"],
        "total_pieces": verdict["total_pieces"],
        "missing_count": verdict["missing_count"],
        "extra_count": verdict["extra_count"],
        "side_to_move_correct": verdict["side_to_move_correct"],
        "seconds": seconds,
        "digest": digest,
    }
    if task == "board+legal":
        legal = verify_legal_moves(fen, response)
        row.update({"legal_found": legal["found"], "legal_recall": legal["recall"],
                    "legal_precision": legal["precision"]})
    return row


def load_done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with open(path) as handle:
        return {(row["model"], row["idx"]) for row in map(json.loads, filter(str.strip, handle))}


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + Z_95**2 / n
    centre = (p + Z_95**2 / (2 * n)) / denom
    half = Z_95 * math.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _legal_means(items: list[dict]) -> dict[str, float]:
    if not any("legal_recall" in r for r in items):
        return {}
    return {
        "mean_legal_recall": _mean([r["legal_recall"] for r in items if "legal_recall" in r]),
        "mean_legal_precision": _mean([r["legal_precision"] for r in items if "legal_precision" in r]),
    }


def summarise(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    summary = []
    for model, items in by_model.items():
        n = len(items)
        perfect = sum(r["correct"] for r in items)
        low, high = wilson_interval(perfect, n)
        stated = [r["side_to_move_correct"] for r in items if r["side_to_move_correct"] is not None]
        summary.append({
            "model": model,
            "base_model": items[0]["base_model"],
            "fine_tuned": items[0]["fine_tuned"],
            "n": n,
            "mean_recall": _mean([r["recall"] for r in items]),
            "mean_precision": _mean([r["precision"] for r in items]),
            "perfect_rate": perfect / n,
            "perfect_ci_low": low,
            "perfect_ci_high": high,
            "mean_extra": _mean([r["extra_count"] for r in items]),
            "mean_missing": _mean([r["missing_count"] for r in items]),
            "side_to_move_acc": _mean([float(s) for s in stated]),
            "unparsed_rate": sum(not r["parsed"] for r in items) / n,
            **_legal_means(items),
        })
    return sorted(summary, key=lambda s: (s["base_model"], s["fine_tuned"]))


def model_pairs(config: dict, available: set[str]) -> tuple[list[tuple[str, str]], list[str]]:
    pairs, missing = [], []
    for entry in config.get("models", []):
        base, ft = entry["ollama_base"], entry["ollama_ft_tag"]
        if ft not in available:
            missing.append(ft)
        elif base in available:
            pairs.append((base, ft))
    return pairs, missing


def ollama_models() -> dict[str, str]:
    """Map of installed Ollama model name to its digest."""
    listing = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True).stdout
    models = {}
    for line in listing.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            models[parts[0]] = parts[1]
    return models


def run_model(model, base, fine_tuned, held_out, out_path, done, host, digest, style, logger, task="board") -> None:
    todo = [(i, p) for i, p in enumerate(held_out) if (model, i) not in done]
    logger.info(f"{model}: {len(todo)} of {len(held_out)} positions to run")
    with open(out_path, "a") as out:
        for i, position in todo:
            fen = position["input"].replace("FEN: ", "").strip()
            started = time.monotonic()
            response = query_ollama(model, build_prompt(position, style, task), host, task=task)
            row = score_row(model, base, fine_tuned, i, fen, response, time.monotonic() - started, digest, task)
            out.write(json.dumps(row) + "\n")
            out.flush()


def write_summary(rows: list[dict], path: Path) -> list[dict]:
    summary = summarise(rows)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="format")
    parser.add_argument("--task", choices=TASKS, default="board", help="board, or board+legal for the legal-moves fine-tune")
    parser.add_argument("--host", default=DEFAULT_OLLAMA)
    parser.add_argument("--models", nargs="*", help="Base model tags to test (default: every pair in Ollama)")
    args = parser.parse_args()
    logger = setup_logging("board_reading_test")

    config = load_ft_config(args.config)
    installed = ollama_models()
    pairs, missing = model_pairs(config, set(installed))
    for tag in missing:
        logger.warning(f"Fine-tuned model not in Ollama, skipping: {tag}")
    if args.models:
        pairs = [p for p in pairs if p[0] in args.models]

    _, held_out = split_held_out(load_dataset_records(config["training"]["dataset_path"]), seed=SEED, n=HELD_OUT_N)
    out_path, summary_path = output_paths(args.prompt_style, args.task)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(out_path)

    for base, ft in pairs:
        for model, fine_tuned in ((base, False), (ft, True)):
            run_model(
                model, base, fine_tuned, held_out, out_path, done, args.host, installed[model],
                args.prompt_style, logger, args.task,
            )

    with open(out_path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    summary = write_summary(rows, summary_path)
    for s in summary:
        logger.info(
            f"{s['model']:28s} perfect {s['perfect_rate']:.1%} "
            f"[{s['perfect_ci_low']:.1%}-{s['perfect_ci_high']:.1%}]  recall {s['mean_recall']:.3f}  "
            f"precision {s['mean_precision']:.3f}  extra {s['mean_extra']:.2f}  unparsed {s['unparsed_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
