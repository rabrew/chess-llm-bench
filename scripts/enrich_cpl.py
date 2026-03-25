"""Post-processing script to fill t2_cpl and t2_best_move using Lc0 (GPU).

Run after the full benchmark pipeline completes:
    python scripts/enrich_cpl.py
    python scripts/enrich_cpl.py --dry-run
    python scripts/enrich_cpl.py --nodes 400
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine_wrapper import Lc0Engine
from src.utils import load_config, setup_logging

logger = logging.getLogger("chess_llm_bench")

TIERS = ("easy", "medium", "hard", "extreme")


# ---------------------------------------------------------------------------
# Pure logic (testable without engine or filesystem)
# ---------------------------------------------------------------------------

def compute_cpl(fen: str, eval_before: int, eval_after: int) -> int:
    """Compute centipawn loss from Lc0 pre- and post-move evals (White's POV).

    Args:
        fen: FEN of the position *before* the move (used only to determine side).
        eval_before: Lc0 eval of the position before the move (White's POV).
        eval_after: Lc0 eval of the position after the model's move (White's POV).

    Returns:
        Non-negative CPL. 0 means the model played the best move.
    """
    board = chess.Board(fen)
    if board.turn == chess.WHITE:
        cpl = eval_before - eval_after
    else:
        cpl = eval_after - eval_before
    return max(0, cpl)


def load_positions(data_dir: str) -> dict[int, dict]:
    """Load position dicts from all tier JSON files, keyed by position id.

    Args:
        data_dir: Directory containing easy.json, medium.json, etc.

    Returns:
        Dict mapping position_id (int) -> position dict.
    """
    positions: dict[int, dict] = {}
    for tier in TIERS:
        path = Path(data_dir) / f"{tier}.json"
        if not path.exists():
            logger.warning(f"Dataset file not found, skipping: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for pos in data:
            positions[int(pos["id"])] = pos
    return positions


def collect_work(
    records: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split records into those needing CPL enrichment and those that don't.

    A record needs enrichment if t2_legal is True, t2_move is not None,
    and t2_cpl is None.

    Args:
        records: All records from evaluations.jsonl.

    Returns:
        (to_enrich, passthrough) — passthrough records are written unchanged.
    """
    to_enrich = []
    passthrough = []
    for r in records:
        if r.get("t2_legal") is True and r.get("t2_move") is not None and r.get("t2_cpl") is None:
            to_enrich.append(r)
        else:
            passthrough.append(r)
    return to_enrich, passthrough


def enrich_records(records: list[dict], engine: Any) -> list[dict]:
    """Fill t2_cpl and t2_best_move for a list of records using Lc0.

    Deduplicates FEN evaluations:
    - Each unique pre-move FEN is evaluated once.
    - Each unique (fen, model_move) pair is evaluated once (skipped if model
      played the best move, since CPL=0).

    Args:
        records: Records to enrich (all must have t2_legal=True, t2_move set,
                 t2_cpl=None).
        engine: Lc0Engine instance.

    Returns:
        Same records with t2_cpl and t2_best_move filled in where possible.
    """
    if not records:
        return records

    # --- Pass 1: evaluate all unique pre-move FENs ---
    unique_fens = {r["fen"] for r in records}
    pre_move_results: dict[str, dict] = {}  # fen -> {"eval": int, "best_move": str}

    logger.info(f"Evaluating {len(unique_fens)} unique pre-move FENs...")
    for fen in unique_fens:
        try:
            pre_move_results[fen] = engine.evaluate(fen)
        except Exception as e:
            logger.warning(f"Pre-move eval failed for FEN {fen!r}: {e}")
            pre_move_results[fen] = None

    # --- Pass 2: evaluate unique (fen, model_move) pairs where move != best move ---
    pairs_to_eval: set[tuple[str, str]] = set()
    for r in records:
        fen = r["fen"]
        move = r["t2_move"]
        pre = pre_move_results.get(fen)
        if pre is None:
            continue
        best_move = pre.get("best_move")
        if move != best_move:
            pairs_to_eval.add((fen, move))

    post_move_results: dict[tuple[str, str], int | None] = {}  # (fen, move) -> eval_after

    logger.info(f"Evaluating {len(pairs_to_eval)} unique post-move positions...")
    for fen, move_san in pairs_to_eval:
        try:
            board = chess.Board(fen)
            chess_move = board.parse_san(move_san)
            board.push(chess_move)
            result = engine.evaluate(board.fen())
            post_move_results[(fen, move_san)] = result["eval"]
        except Exception as e:
            logger.warning(f"Post-move eval failed for {move_san!r} on {fen!r}: {e}")
            post_move_results[(fen, move_san)] = None

    # --- Fill in each record ---
    enriched = []
    for r in records:
        r = dict(r)  # shallow copy; don't mutate caller's data
        fen = r["fen"]
        move = r["t2_move"]
        pre = pre_move_results.get(fen)

        if pre is None:
            enriched.append(r)
            continue

        r["t2_best_move"] = pre.get("best_move")

        if move == pre.get("best_move"):
            r["t2_cpl"] = 0
        else:
            eval_after = post_move_results.get((fen, move))
            if eval_after is not None:
                r["t2_cpl"] = compute_cpl(fen, pre["eval"], eval_after)
            # else leave t2_cpl as None (engine failed)

        enriched.append(r)

    return enriched


# ---------------------------------------------------------------------------
# I/O and CLI
# ---------------------------------------------------------------------------

def run(results_file: str, data_dir: str, engine: Any, dry_run: bool) -> None:
    """Main enrichment logic: read, enrich, write atomically."""
    results_path = Path(results_file)

    logger.info(f"Reading {results_path}...")
    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    to_enrich, passthrough = collect_work(records)

    logger.info(
        f"Records: {len(records)} total, "
        f"{len(to_enrich)} to enrich, "
        f"{len(passthrough)} passthrough"
    )

    if not to_enrich:
        logger.info("Nothing to enrich.")
        return

    if dry_run:
        unique_fens = len({r["fen"] for r in to_enrich})
        logger.info(f"[dry-run] Would evaluate {unique_fens} unique pre-move FENs")
        logger.info(f"[dry-run] No files written.")
        return

    enriched = enrich_records(to_enrich, engine)

    # Merge: preserve original order
    enriched_by_id = {r["job_id"]: r for r in enriched}
    all_records = []
    failed = 0
    success = 0
    for r in records:
        job_id = r["job_id"]
        if job_id in enriched_by_id:
            out = enriched_by_id[job_id]
            if out.get("t2_cpl") is not None or out.get("t2_best_move") is not None:
                success += 1
            else:
                failed += 1
            all_records.append(out)
        else:
            all_records.append(r)

    # Atomic write
    tmp_path = results_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    tmp_path.replace(results_path)

    logger.info(
        f"\nCPL enrichment complete.\n"
        f"  Records processed : {len(records)}\n"
        f"  Enriched          : {success}\n"
        f"  Failed            : {failed}\n"
        f"  Skipped (illegal/done) : {len(passthrough)}\n"
        f"  Output            : {results_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich evaluations.jsonl with CPL via Lc0.")
    parser.add_argument("--results", default="results/evaluations.jsonl",
                        help="Path to evaluations.jsonl (default: results/evaluations.jsonl)")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing tier JSON files (default: data)")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Config file path (default: config/config.yaml)")
    parser.add_argument("--nodes", type=int, default=None,
                        help="Override Lc0 nodes per position (default: from config or 800)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be enriched without writing anything")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    lc0_config = config.get("lc0", {})
    binary = lc0_config.get("binary", "/home/rabrew/lc0-src/build/release/lc0")
    weights = lc0_config.get("weights", "/home/rabrew/lc0-nets/network.pb")
    nodes = args.nodes or lc0_config.get("nodes", 800)

    if args.dry_run:
        run(args.results, args.data_dir, engine=None, dry_run=True)
        return

    logger.info(f"Starting Lc0: {binary} (nodes={nodes})")
    with Lc0Engine(path=binary, weights=weights, nodes=nodes) as engine:
        run(args.results, args.data_dir, engine=engine, dry_run=False)


if __name__ == "__main__":
    main()
