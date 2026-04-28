#!/usr/bin/env python3
"""Ultra-fast Lc0 evaluations using valuehead mode with minimal nodes."""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Lock

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging

import chess

file_lock = Lock()


class FastLc0Engine:
    """Fast Lc0 engine using valuehead mode with 1 node."""

    def __init__(self, engine_id=0):
        self.engine_id = engine_id
        self.process = subprocess.Popen(
            [
                "/home/rabrew/lc0-src/build/release/lc0",
                "valuehead",
                f"--weights=/home/rabrew/lc0-nets/network.pb",
                "--backend=cuda-auto",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._init_uci()

    def _init_uci(self):
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def _readline(self):
        return self.process.stdout.readline().strip()

    def _wait_for(self, target):
        while True:
            line = self._readline()
            if target in line:
                return line

    def evaluate(self, fen):
        """Evaluate position with 1 node (instant)."""
        self._send(f"position fen {fen}")
        self._send("go nodes 1")

        eval_cp = 0
        best_move = None

        while True:
            line = self._readline()
            if line.startswith("info"):
                parts = line.split()
                if "score" in parts and "cp" in parts:
                    try:
                        idx = parts.index("cp")
                        eval_cp = int(parts[idx + 1])
                    except (ValueError, IndexError):
                        pass
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move_uci = parts[1]
                    try:
                        board = chess.Board(fen)
                        move = chess.Move.from_uci(best_move_uci)
                        best_move = board.san(move)
                    except (ValueError, chess.InvalidMoveError):
                        best_move = best_move_uci
                break

        # Adjust for side to move
        board = chess.Board(fen)
        if board.turn == chess.BLACK:
            eval_cp = -eval_cp

        return {"eval": eval_cp, "best_move": best_move}

    def close(self):
        try:
            self._send("quit")
            self.process.wait(timeout=2)
        except Exception:
            self.process.kill()


def main():
    parser = argparse.ArgumentParser(description="Fast Lc0 GPU evaluation")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--workers", type=int, default=8, help="Parallel Lc0 instances")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    import logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("chess_llm_bench")

    data_dir = Path(args.data_dir)
    tiers = ["easy", "medium", "hard", "extreme"]

    tier_data = {}
    positions_to_eval = []

    for tier in tiers:
        file_path = data_dir / f"{tier}.json"
        if not file_path.exists():
            continue
        with open(file_path, "r") as f:
            positions = json.load(f)
        tier_data[tier] = positions
        for i, pos in enumerate(positions):
            if "stockfish_eval" not in pos:
                positions_to_eval.append((tier, i, pos["fen"]))

    if not positions_to_eval:
        print("All positions already evaluated.")
        return

    print(f"\n{'='*60}")
    print(f"  LC0 FAST GPU EVALUATION")
    print(f"  Positions: {len(positions_to_eval):,}")
    print(f"  Workers: {args.workers}")
    print(f"  Mode: valuehead (1 node per position)")
    print(f"{'='*60}\n")

    # Create engines
    print(f"Starting {args.workers} Lc0 engines...")
    engines = []
    for i in range(args.workers):
        engines.append(FastLc0Engine(i))
        print(f"  Engine {i+1}/{args.workers} ready")
    print()

    # Process
    results = {}
    evaluated = 0
    failed = 0

    CHECKPOINT_EVERY = 5000
    last_checkpoint = 0

    def save_checkpoint():
        with file_lock:
            for tier, idx, fen in positions_to_eval:
                if fen in results:
                    r = results[fen]
                    tier_data[tier][idx]["stockfish_eval"] = r["eval"]
                    tier_data[tier][idx]["stockfish_best_move"] = r["best_move"]
            for t, positions in tier_data.items():
                with open(data_dir / f"{t}.json", "w") as f:
                    json.dump(positions, f)

    def eval_work(args):
        engine, fen = args
        try:
            return fen, engine.evaluate(fen), None
        except Exception as e:
            return fen, None, str(e)

    try:
        work_items = [(engines[i % len(engines)], fen) for i, (_, _, fen) in enumerate(positions_to_eval)]

        pbar = tqdm(total=len(work_items), desc="Evaluating", unit="pos")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_fen = {executor.submit(eval_work, w): w[1] for w in work_items}

            for future in as_completed(future_to_fen):
                fen, result, error = future.result()
                if result:
                    results[fen] = result
                    evaluated += 1
                else:
                    failed += 1

                pbar.update(1)
                pbar.set_postfix({"done": evaluated, "failed": failed, "pos/s": f"{pbar.format_dict['rate']:.0f}" if pbar.format_dict.get('rate') else "?"})

                if evaluated - last_checkpoint >= CHECKPOINT_EVERY:
                    pbar.set_description("Checkpoint")
                    save_checkpoint()
                    last_checkpoint = evaluated
                    logger.info(f"Checkpoint: {evaluated:,}")
                    pbar.set_description("Evaluating")

        pbar.close()

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        save_checkpoint()
        for e in engines:
            e.close()

    print(f"\nDone: {evaluated:,} evaluated, {failed} failed")


if __name__ == "__main__":
    main()
