#!/usr/bin/env python3
"""Ultra-fast Lc0 batch evaluation - optimized version with pre-encoding."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import chess

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, setup_logging

import onnxruntime as ort
from tqdm import tqdm


def encode_position(board: chess.Board) -> np.ndarray:
    """Encode a chess position to Lc0's 112-plane input format."""
    planes = np.zeros((112, 8, 8), dtype=np.float32)
    flip = board.turn == chess.BLACK
    piece_to_plane = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    for history_idx in range(8):
        base = history_idx * 13
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            if flip:
                rank = 7 - rank
            is_ours = (piece.color == board.turn)
            plane_offset = 0 if is_ours else 6
            plane_idx = base + plane_offset + piece_to_plane[piece.piece_type]
            planes[plane_idx, rank, file] = 1.0

    # Castling rights
    if board.turn == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[104, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[105, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[106, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[107, :, :] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[104, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[105, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[106, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[107, :, :] = 1.0

    planes[108, :, :] = 1.0
    planes[109, :, :] = board.halfmove_clock / 100.0
    planes[111, :, :] = 1.0

    return planes


def wdl_to_centipawns(wdl: np.ndarray) -> int:
    """Convert WDL probabilities to centipawn evaluation."""
    w, d, l = wdl[0], wdl[1], wdl[2]
    score = w + d * 0.5
    if score >= 0.999:
        return 10000
    elif score <= 0.001:
        return -10000
    else:
        cp = int(111.714 * np.tan(1.5620688 * (score - 0.5)))
        return max(-10000, min(10000, cp))


def main():
    parser = argparse.ArgumentParser(description="Fast batch Lc0 evaluation v2")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=200000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    import logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("chess_llm_bench")

    # Load ONNX model — path comes from config/config.yaml (lc0.onnx_model)
    config = load_config(getattr(args, "config", None) or "config/config.yaml")
    onnx_path = config.get("lc0", {}).get("onnx_model")
    if not onnx_path:
        raise SystemExit(
            "lc0.onnx_model must be set in config/config.yaml "
            "before running precompute_lc0_batch_v2.py"
        )
    print(f"Loading Lc0 ONNX model from {onnx_path}...")
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"Using providers: {sess.get_providers()}")

    # Load positions
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
    print(f"  LC0 BATCH GPU EVALUATION v2 (OPTIMIZED)")
    print(f"  Positions: {len(positions_to_eval):,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Checkpoint every: {args.checkpoint_every:,}")
    print(f"{'='*60}\n")

    # PHASE 1: Pre-encode all positions
    print("Phase 1: Pre-encoding all positions...")
    encoded_data = []
    meta_data = []  # (tier, idx, is_black)

    encode_pbar = tqdm(positions_to_eval, desc="Encoding", unit="pos")
    for tier, idx, fen in encode_pbar:
        try:
            board = chess.Board(fen)
            encoded = encode_position(board)
            encoded_data.append(encoded)
            meta_data.append((tier, idx, board.turn == chess.BLACK))
        except Exception:
            continue
    encode_pbar.close()

    # Stack into single array for fast slicing
    print(f"Stacking {len(encoded_data):,} encoded positions...")
    all_encoded = np.stack(encoded_data, axis=0)
    del encoded_data  # Free memory

    print(f"Encoded shape: {all_encoded.shape}")
    print(f"Memory: {all_encoded.nbytes / 1e9:.2f} GB\n")

    # PHASE 2: GPU inference in batches
    print("Phase 2: GPU inference...")

    def save_checkpoint():
        for tier, positions in tier_data.items():
            with open(data_dir / f"{tier}.json", "w") as f:
                json.dump(positions, f)
        logger.info(f"Checkpoint saved")

    evaluated = 0
    last_checkpoint = 0
    n_positions = len(meta_data)

    try:
        pbar = tqdm(total=n_positions, desc="Evaluating", unit="pos")

        for batch_start in range(0, n_positions, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_positions)
            batch_input = all_encoded[batch_start:batch_end]

            # GPU inference
            outputs = sess.run(None, {"/input/planes": batch_input})
            wdl_batch = outputs[1]

            # Store results
            for i in range(batch_end - batch_start):
                tier, idx, is_black = meta_data[batch_start + i]
                wdl = wdl_batch[i]
                cp = wdl_to_centipawns(wdl)

                if is_black:
                    cp = -cp

                tier_data[tier][idx]["stockfish_eval"] = cp
                tier_data[tier][idx]["stockfish_best_move"] = None
                evaluated += 1

            pbar.update(batch_end - batch_start)
            pbar.set_postfix({"done": f"{evaluated:,}"})

            # Checkpoint (less frequently)
            if evaluated - last_checkpoint >= args.checkpoint_every:
                pbar.set_description("Checkpoint")
                save_checkpoint()
                last_checkpoint = evaluated
                pbar.set_description("Evaluating")

        pbar.close()

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        save_checkpoint()

    print(f"\nDone: {evaluated:,} positions evaluated")


if __name__ == "__main__":
    main()
