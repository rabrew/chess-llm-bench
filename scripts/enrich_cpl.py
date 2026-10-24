"""Post-processing script to fill t2_cpl and t2_best_move using Lc0 (GPU).

Run after the full benchmark pipeline completes:
    python scripts/enrich_cpl.py
    python scripts/enrich_cpl.py --dry-run
    python scripts/enrich_cpl.py --nodes 400
    python scripts/enrich_cpl.py --backfill-eval-before   # T2-6, specs/cpl-eval-before.md
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine_wrapper import Lc0Engine
from src.utils import load_config, setup_logging

logger = logging.getLogger("chess_llm_bench")

TIERS = ("easy", "medium", "hard", "extreme")
BACKUP_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


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
    pre_move_results = evaluate_unique_fens({r["fen"] for r in records}, engine)

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
        r.update(_eval_before_fields(pre, engine))

        if move == pre.get("best_move"):
            r["t2_cpl"] = 0
        else:
            eval_after = post_move_results.get((fen, move))
            if eval_after is not None:
                r["t2_cpl"] = compute_cpl(fen, pre["eval"], eval_after)
            # else leave t2_cpl as None (engine failed)

        enriched.append(r)

    return enriched


def evaluate_unique_fens(fens: set[str], engine: Any) -> dict[str, dict | None]:
    """Evaluate each FEN once; a failed evaluation maps to None and is logged."""
    results: dict[str, dict | None] = {}
    logger.info(f"Evaluating {len(fens)} unique pre-move FENs...")
    for fen in fens:
        try:
            results[fen] = engine.evaluate(fen)
        except Exception as e:  # engine crash, bad FEN: logged and counted, never fatal
            logger.warning(f"Pre-move eval failed for FEN {fen!r}: {e}")
            results[fen] = None
    return results


def _eval_before_fields(pre: dict, engine: Any) -> dict[str, Any]:
    """The 800-node 'before' eval that t2_cpl is measured from (specs/cpl-eval-before.md)."""
    return {"t2_eval_before": pre["eval"], "t2_eval_nodes": getattr(engine, "nodes", None)}


def _needs_eval_before(record: dict) -> bool:
    return record.get("t2_cpl") is not None and record.get("t2_eval_before") is None


def backfill_eval_before(records: list[dict], engine: Any) -> tuple[list[dict], int]:
    """Add t2_eval_before to records that have a CPL but no stored 'before' eval.

    Only starting positions are evaluated; t2_cpl and post-move evals are untouched.

    Returns:
        (records as new dicts where filled, number of records whose FEN failed)
    """
    fens = {r["fen"] for r in records if _needs_eval_before(r)}
    pre_move_results = evaluate_unique_fens(fens, engine) if fens else {}
    filled, failed = [], 0
    for r in records:
        if not _needs_eval_before(r):
            filled.append(r)
            continue
        pre = pre_move_results.get(r["fen"])
        if pre is None:
            failed += 1
            filled.append(r)
            continue
        filled.append({**r, **_eval_before_fields(pre, engine)})
    return filled, failed


# ---------------------------------------------------------------------------
# I/O and CLI
# ---------------------------------------------------------------------------

def _read_records(results_path: Path) -> list[dict]:
    logger.info(f"Reading {results_path}...")
    records = []
    with open(results_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON on line {lineno}")
    return records


def _write_records_atomically(results_path: Path, records: list[dict]) -> None:
    tmp_path = results_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp_path.replace(results_path)


def _backup(results_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime(BACKUP_TIMESTAMP_FORMAT)
    backup_path = results_path.with_name(f"{results_path.name}.bak-{stamp}")
    shutil.copy2(results_path, backup_path)
    logger.info(f"Backed up {results_path} to {backup_path}")
    return backup_path


def run_backfill(results_path: Path, engine: Any, backup: bool = True) -> None:
    """Fill t2_eval_before on existing results (T2-6)."""
    records = _read_records(results_path)
    filled, failed = backfill_eval_before(records, engine)
    changed = sum(1 for before, after in zip(records, filled) if before is not after)
    logger.info(f"Backfill: {changed} records filled, {failed} failed")
    if not changed:
        return
    if backup:
        _backup(results_path)
    _write_records_atomically(results_path, filled)


def run(results_file: str, data_dir: str, engine: Any, dry_run: bool,
        backfill: bool = False, backup: bool = True) -> None:
    """Main enrichment logic: read, enrich, write atomically."""
    results_path = Path(results_file)
    if backfill:
        run_backfill(results_path, engine, backup=backup)
        return

    records = _read_records(results_path)
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
        logger.info("[dry-run] No files written.")
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

    _write_records_atomically(results_path, all_records)

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
    parser.add_argument("--backfill-eval-before", action="store_true",
                        help="Add the Lc0 'before' eval to records that already have a CPL (T2-6)")
    parser.add_argument("--no-backup", action="store_true",
                        help="With --backfill-eval-before: do not copy the results file first")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    lc0_config = config.get("lc0", {})
    binary = lc0_config.get("binary")
    weights = lc0_config.get("weights")
    if not binary or not weights:
        raise SystemExit(
            "lc0.binary and lc0.weights must be set in config/config.yaml "
            "before running enrich_cpl.py"
        )
    nodes = args.nodes or lc0_config.get("nodes", 800)

    if args.dry_run:
        run(args.results, args.data_dir, engine=None, dry_run=True)
        return

    logger.info(f"Starting Lc0: {binary} (nodes={nodes})")
    with Lc0Engine(path=binary, weights=weights, nodes=nodes) as engine:
        run(args.results, args.data_dir, engine=engine, dry_run=False,
            backfill=args.backfill_eval_before, backup=not args.no_backup)


if __name__ == "__main__":
    main()
