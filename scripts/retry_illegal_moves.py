#!/usr/bin/env python3
"""
Retry illegal/missing moves from a completed benchmark run.

For every record in evaluations.jsonl where t2_legal is False, re-prompts
the model with the full legal move list so it must pick a valid move.
Results are written to results/evaluations_retried.jsonl.

Usage:
    python scripts/retry_illegal_moves.py
    python scripts/retry_illegal_moves.py --input results/evaluations.jsonl
    python scripts/retry_illegal_moves.py --model phi4:14b   # one model only
    python scripts/retry_illegal_moves.py --dry-run          # no writes, just print
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import chess
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retry_illegal_moves")

OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 180


# ---------------------------------------------------------------------------
# Prompt + parse
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a chess engine. When given a FEN position, output exactly one legal move "
    "in Standard Algebraic Notation (SAN).\n\n"
    "A legal move must satisfy ALL of the following:\n"
    "- The piece you are moving must actually exist on the square you name.\n"
    "- The piece must be able to reach the destination square by its movement rules "
    "(e.g. a bishop moves diagonally, a knight in an L-shape, a pawn one square forward "
    "or diagonally to capture).\n"
    "- The move must not leave your own king in check.\n"
    "- It must be the turn of the side indicated by the FEN (w = White, b = Black).\n"
    "- Castling (O-O or O-O-O) is only legal if the king and rook have not moved, "
    "there are no pieces between them, and the king does not pass through check.\n"
    "- En passant is only legal if the FEN en-passant square is set.\n\n"
    "Output the move only — no explanation, no punctuation, no move numbers.\n"
    "Examples of correct output: e4, Nf3, O-O, Bxc6, exd5, Qh5+"
)


def build_retry_prompt(fen: str, illegal_move: str | None) -> str:
    if illegal_move:
        problem = f'Your previous move "{illegal_move}" was illegal in this position. Think carefully about the piece locations and movement rules before answering.'
    else:
        problem = "You did not provide a move for this position."
    return (
        f"{problem}\n\n"
        f"Position (FEN): {fen}\n\n"
        "What is the best legal move? Respond with the SAN move only."
    )


def parse_move(response_text: str) -> str | None:
    text = response_text.strip()
    text = re.sub(r"\*\*|`", "", text)
    first_line = text.split("\n")[0].strip()
    token = first_line.split()[0] if first_line.split() else ""
    token = token.rstrip(".,;:").strip("`")
    token = re.sub(r"^\d+\.+", "", token)
    return token if token else None


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def ollama_models() -> set[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except requests.RequestException:
        return set()


def chat(model: str, prompt: str) -> tuple[str, int]:
    """Returns (response_text, inference_ms). Raises on failure."""
    start = time.time()
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    text = r.json().get("message", {}).get("content", "")
    ms = int((time.time() - start) * 1000)
    return text, ms


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def is_legal(fen: str, san: str) -> bool:
    try:
        board = chess.Board(fen)
        move = board.parse_san(san)
        return move in board.legal_moves
    except Exception:
        return False


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/evaluations.jsonl")
    parser.add_argument("--output", default="results/evaluations_retried.jsonl")
    parser.add_argument("--model", help="Only retry records for this model")
    parser.add_argument("--dry-run", action="store_true", help="Don't write results")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if not ollama_available():
        logger.error("Ollama is not running. Start it first.")
        sys.exit(1)

    available_models = ollama_models()

    # Load existing retried records so we can skip already-done jobs
    already_retried = {r["job_id"] for r in load_jsonl(output_path)}

    # Load all records and filter to illegal moves
    all_records = load_jsonl(input_path)
    to_retry = [
        r for r in all_records
        if str(r.get("t2_legal")) != "True"
        and r.get("job_id") not in already_retried
        and (args.model is None or r.get("model") == args.model)
    ]

    logger.info(f"Total records: {len(all_records)}")
    logger.info(f"Illegal/missing moves to retry: {len(to_retry)}")
    logger.info(f"Already retried (skipping): {len(already_retried)}")

    if not to_retry:
        logger.info("Nothing to do.")
        return

    stats = {"retried": 0, "became_legal": 0, "still_illegal": 0, "skipped": 0}

    for i, rec in enumerate(to_retry, 1):
        model = rec["model"]
        fen = rec["fen"]
        job_id = rec["job_id"]
        original_move = rec.get("t2_move")
        stockfish_best = rec.get("t2_best_move")

        logger.info(f"[{i}/{len(to_retry)}] {model} | {job_id}")

        if model not in available_models:
            logger.warning(f"  Model {model} not available in Ollama — skipping")
            stats["skipped"] += 1
            continue

        prompt = build_retry_prompt(fen, original_move)

        try:
            response_text, inference_ms = chat(model, prompt)
        except Exception as e:
            logger.warning(f"  Ollama call failed: {e} — skipping")
            stats["skipped"] += 1
            continue

        retried_move = parse_move(response_text)
        retried_legal = is_legal(fen, retried_move) if retried_move else False

        retried_cpl = None
        if retried_legal:
            if retried_move == stockfish_best:
                retried_cpl = 0
            stats["became_legal"] += 1
        else:
            stats["still_illegal"] += 1

        result = {
            "job_id": job_id,
            "model": model,
            "fen": fen,
            "original_move": original_move,
            "retried_move": retried_move,
            "retried_legal": retried_legal,
            "retried_cpl": retried_cpl,
            "stockfish_best_move": stockfish_best,
            "inference_ms": inference_ms,
        }

        if args.dry_run:
            print(json.dumps(result))
        else:
            append_jsonl(output_path, result)

        stats["retried"] += 1

    print("\n=== Retry Summary ===")
    print(f"  Retried        : {stats['retried']}")
    print(f"  Became legal   : {stats['became_legal']} ({100*stats['became_legal']/max(stats['retried'],1):.1f}%)")
    print(f"  Still illegal  : {stats['still_illegal']}")
    print(f"  Skipped        : {stats['skipped']}")
    if not args.dry_run and stats["retried"] > 0:
        print(f"  Output         : {output_path}")


if __name__ == "__main__":
    main()
