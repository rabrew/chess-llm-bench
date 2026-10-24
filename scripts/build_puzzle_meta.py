"""Join Lichess puzzle metadata back to the benchmark positions (T3-4, Issue 21).

The benchmark shows the CSV ``FEN``, which is the position *before* the
puzzle's first move (``Moves[0]``, the setup move). This records the full
theme list, the setup move and the puzzle position for each benchmark FEN.
Spec: specs/puzzle-metadata-join.md.

Usage (from the repo root):
    python scripts/build_puzzle_meta.py --results results/evaluations.jsonl
"""

import argparse
import csv
import io
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import chess
import zstandard

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import PUZZLE_META_FILENAME as META_FILENAME  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("chess_llm_bench")

DEFAULT_CSV = "data/lichess_puzzles.csv.zst"
DEFAULT_OUT = f"data/{META_FILENAME}"
MIN_COVERAGE = 0.99


def read_puzzle_rows(csv_path: Path) -> Iterable[dict]:
    """Stream rows of the zstd-compressed Lichess puzzle CSV."""
    with open(csv_path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        yield from csv.DictReader(text)


def collect_fens(results_paths: list[Path]) -> set[str]:
    fens: set[str] = set()
    for path in results_paths:
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                try:
                    fen = json.loads(line).get("fen")
                except json.JSONDecodeError as exc:
                    logger.warning(f"{path}:{line_no}: skipping malformed JSON ({exc})")
                    continue
                if isinstance(fen, str):
                    fens.add(fen)
    return fens


def _setup_move(fen: str, uci: str) -> tuple[str | None, str | None]:
    """SAN of the setup move and the FEN after it; (None, None) if it is not legal."""
    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None, None
    if move not in board.legal_moves:
        return None, None
    san = board.san(move)
    board.push(move)
    return san, board.fen()


def _meta_entry(row: dict) -> dict:
    setup_uci = row["Moves"].split()[0]
    san, puzzle_fen = _setup_move(row["FEN"], setup_uci)
    return {
        "puzzle_id": row["PuzzleId"],
        "themes_all": row["Themes"].split(),
        "setup_move_uci": setup_uci,
        "setup_move_san": san,
        "puzzle_fen": puzzle_fen,
        "rating": int(row["Rating"]),
    }


def build_meta(rows: Iterable[dict], fens: set[str]) -> dict[str, dict]:
    """Metadata for each benchmark FEN found in the puzzle rows.

    Raises:
        ValueError: if fewer than MIN_COVERAGE of the FENs are found.
    """
    meta: dict[str, dict] = {}
    duplicates = 0
    for row in rows:
        fen = row["FEN"]
        if fen not in fens:
            continue
        if fen in meta:
            duplicates += 1
            continue
        meta[fen] = _meta_entry(row)
    if duplicates:
        logger.warning(f"{duplicates} duplicate puzzle rows for benchmark FENs; kept the first")
    illegal = sum(1 for entry in meta.values() if entry["setup_move_san"] is None)
    if illegal:
        logger.warning(f"{illegal} puzzles have a setup move that is illegal on their FEN")
    coverage = len(meta) / len(fens) if fens else 0.0
    if coverage < MIN_COVERAGE:
        raise ValueError(f"Only {len(meta)}/{len(fens)} benchmark FENs found in the puzzle CSV")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Join Lichess puzzle metadata to benchmark positions.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"Lichess puzzle CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--results", action="append", required=True, help="Results JSONL (repeatable)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output JSON (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    setup_logging()

    paths = [Path(args.csv)] + [Path(p) for p in args.results]
    missing = [p for p in paths if not p.exists()]
    if missing:
        logger.error(f"Not found: {', '.join(map(str, missing))}")
        sys.exit(1)

    fens = collect_fens(paths[1:])
    try:
        meta = build_meta(read_puzzle_rows(paths[0]), fens)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    out = Path(args.out)
    out.write_text(json.dumps({"_source": str(paths[0]), "positions": meta}))
    logger.info(f"Wrote {out}: {len(meta)} positions")


if __name__ == "__main__":
    main()
