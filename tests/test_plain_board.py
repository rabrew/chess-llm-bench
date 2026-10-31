"""Tests for the plain-board prompts (specs/plain-board-prompts.md)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fine-tuning" / "src"))

from board_state import fen_to_board_description  # noqa: E402
from scripts.plain_board_compare import (  # noqa: E402
    MAJOR_BARRIER,
    bootstrap_ci,
    MIXED,
    NOT_BARRIER,
    legal_for_other_side,
    pair_rows,
    pooled_measures,
    verdict,
)
from scripts.plain_board_probe import longest_prompt_position, truncated  # noqa: E402
from src.board_text import fen_to_board_text, piece_line  # noqa: E402
from src.job_generator import subsample_per_tier  # noqa: E402
from src.llm_client import (  # noqa: E402
    BOARD_MOVE_SYSTEM_PROMPT,
    BOARD_TEXT_FORMATS,
    MOVE_SYSTEM_PROMPT,
    OllamaClient,
    build_board_move_prompt,
    build_prompt,
    retry_move_prompt,
)
from src.metrics import COMBINED_PROMPT_FORMATS, MOVE_PROMPT_FORMATS  # noqa: E402
from src.worker import Worker  # noqa: E402

START_FEN = chess.STARTING_FEN
EP_FEN = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
BLACK_NO_CASTLE_FEN = "r3k2r/8/8/8/8/8/8/R3K2R b - - 0 1"
PROMOTED_FEN = "4k3/8/8/8/8/8/8/QQ2K3 w - - 0 1"


def _placement(fen):
    return fen.split()[0]


# ---------- board text ----------

def test_start_position_diagram_is_exact():
    text = fen_to_board_text(START_FEN)
    assert "8 | r n b q k b n r" in text
    assert "1 | R N B Q K B N R" in text
    assert "    a b c d e f g h" in text
    assert "Side to move: White" in text
    assert "Castling still possible: White king side, White queen side, Black king side, Black queen side" in text
    assert "En passant: none" in text


def test_black_to_move_without_castling():
    text = fen_to_board_text(BLACK_NO_CASTLE_FEN)
    assert "Side to move: Black" in text
    assert "Castling still possible: none" in text


def test_en_passant_square_named():
    assert "En passant: f6" in fen_to_board_text(EP_FEN)


def test_promoted_pieces_listed():
    assert "White pieces: King e1, Queen a1 b1" in fen_to_board_text(PROMOTED_FEN)


@pytest.mark.parametrize("fen", [START_FEN, EP_FEN, BLACK_NO_CASTLE_FEN, PROMOTED_FEN])
def test_board_text_contains_no_fen(fen):
    text = fen_to_board_text(fen)
    assert _placement(fen) not in text
    assert "FEN" not in text


@pytest.mark.parametrize("fen", [START_FEN, EP_FEN, PROMOTED_FEN])
def test_piece_lines_match_fine_tuning_wording(fen):
    """One wording for piece lists: the training data's lines, with 'pieces' added."""
    board = chess.Board(fen)
    expected = fen_to_board_description(fen).split("\n")[:2]
    ours = [piece_line(board, chess.WHITE), piece_line(board, chess.BLACK)]
    assert ours == [line.replace(":", " pieces:", 1) for line in expected]


# ---------- prompts ----------

def test_board_text_prompt_has_no_fen_and_same_questions():
    plain = build_prompt(fen=EP_FEN, prompt_format="board_text")
    control = build_prompt(fen=EP_FEN, prompt_format="fen_only")
    assert _placement(EP_FEN) not in plain and "FEN" not in plain
    assert "Legal moves:" not in plain
    assert plain.split("Answer all three questions below.")[1] == control.split("Answer all three questions below.")[1]


def test_board_text_legal_prompt_lists_exactly_the_legal_moves():
    prompt = build_prompt(fen=EP_FEN, prompt_format="board_text+legal")
    line = next(row for row in prompt.split("\n") if row.startswith("Legal moves:"))
    listed = line.removeprefix("Legal moves:").strip().split(", ")
    board = chess.Board(EP_FEN)
    assert listed == sorted(board.san(move) for move in board.legal_moves)
    assert _placement(EP_FEN) not in prompt


def test_board_formats_ignore_move_history():
    prompt = build_prompt(fen=START_FEN, pgn_moves="1. e4", prompt_format="board_text")
    assert "Moves played so far" not in prompt


def test_board_move_prompt_and_system_prompt_have_no_fen():
    prompt = build_board_move_prompt(EP_FEN)
    assert _placement(EP_FEN) not in prompt and "FEN" not in prompt
    assert "It is White to move." in prompt and "Legal moves: " in prompt
    assert "FEN" not in BOARD_MOVE_SYSTEM_PROMPT


def test_retry_move_prompt_picks_by_format():
    for fmt in BOARD_TEXT_FORMATS:
        prompt, system = retry_move_prompt(EP_FEN, fmt)
        assert _placement(EP_FEN) not in prompt and system == BOARD_MOVE_SYSTEM_PROMPT
    prompt, system = retry_move_prompt(EP_FEN, "fen_only")
    assert EP_FEN in prompt and system == MOVE_SYSTEM_PROMPT


def test_metrics_count_board_formats_as_combined_move_prompts():
    for fmt in BOARD_TEXT_FORMATS:
        assert fmt in MOVE_PROMPT_FORMATS and fmt in COMBINED_PROMPT_FORMATS


def test_chat_reports_prompt_tokens():
    reply = MagicMock()
    reply.json.return_value = {"message": {"content": "e4"}, "prompt_eval_count": 321}
    with patch("src.llm_client.requests.post", return_value=reply):
        result = OllamaClient(max_retries=1).chat("llama3.2:3b", "prompt")
    assert result["prompt_tokens"] == 321


# ---------- worker: the re-ask never shows a FEN ----------

def _worker(tmp_path, replies):
    config = {
        "paths": {"jobs_db": str(tmp_path / "jobs.db"), "results_file": str(tmp_path / "r.jsonl"),
                  "data_dir": str(tmp_path / "data")},
        "ollama": {"base_url": "http://localhost:11434", "timeout": 180, "max_retries": 1},
        "evaluation": {"centipawn_eval_range": {"min": -2000, "max": 2000}, "cpl_threshold": 50},
        "correction_loop": {"enabled": False},
        "benchmark": {"random_seed": 42},
    }
    with patch("src.worker.OllamaClient"), patch("src.worker.DataLoader"), \
         patch("src.worker.ResultWriter"), patch("src.worker.JobQueue") as queue_cls:
        queue_cls.return_value.get_done_job_ids.return_value = []
        worker = Worker("w", config)
    worker.llm_client = MagicMock()
    worker.llm_client.chat.side_effect = replies
    worker.data_loader = MagicMock()
    worker.data_loader.get_by_id.return_value = {"id": 1, "fen": EP_FEN, "theme": "fork",
                                                 "stockfish_eval": 30, "stockfish_best_move": "exf6"}
    worker.result_writer = MagicMock()
    worker.job_queue = MagicMock()
    worker.correction_manager = MagicMock()
    return worker


def _reply(text):
    return {"response": text, "inference_ms": 10, "success": True, "model": "llama3.2:3b"}


@pytest.mark.parametrize("fmt", BOARD_TEXT_FORMATS)
def test_illegal_move_on_plain_format_reasks_without_fen(tmp_path, fmt):
    worker = _worker(tmp_path, [_reply("Eval: 10\nMove: Qz9\nExplanation: Equal — quiet."), _reply("exf6")])
    job = {"job_id": "j", "job_type": "standard", "position_id": 1, "model": "llama3.2:3b",
           "prompt_format": fmt, "trial": 1}
    result = worker.process_job(job)
    first_call, retry_call = worker.llm_client.chat.call_args_list
    assert _placement(EP_FEN) not in first_call.args[1]
    assert _placement(EP_FEN) not in retry_call.args[1]
    assert retry_call.kwargs["system_prompt"] == BOARD_MOVE_SYSTEM_PROMPT
    assert result["raw_move_retry_response"] == "exf6"


# ---------- sampling ----------

def _tier(name, ids):
    return [{"id": i, "difficulty": name} for i in ids]


def test_subsample_takes_count_per_tier_from_inside_and_is_repeatable():
    positions = _tier("easy", range(100)) + _tier("hard", range(100, 200))
    first = subsample_per_tier(positions, count=10, seed=42)
    assert first == subsample_per_tier(list(reversed(positions)), count=10, seed=42)
    assert sum(p["difficulty"] == "easy" for p in first) == 10
    assert sum(p["difficulty"] == "hard" for p in first) == 10
    assert {p["id"] for p in first} <= set(range(200))


def test_subsample_unset_or_larger_than_tier_keeps_everything():
    positions = _tier("easy", range(5))
    assert subsample_per_tier(positions, count=0, seed=42) == positions
    assert sorted(p["id"] for p in subsample_per_tier(positions, count=50, seed=42)) == list(range(5))


# ---------- comparison ----------

def _rows(fmt, model, ids, **columns):
    frame = pd.DataFrame({"model": model, "position_id": list(ids), "prompt_format": fmt})
    for name, values in columns.items():
        frame[name] = values
    return frame


def test_pair_rows_keeps_only_shared_positions_with_values():
    df = pd.concat([
        _rows("fen_only", "m", [1, 2, 3], t2_cpl_clamped=[0.0, 1000.0, np.nan]),
        _rows("board_text", "m", [2, 3, 4], t2_cpl_clamped=[0.0, 5.0, 1000.0]),
    ])
    left, right = pair_rows(df, "fen_only", "board_text", "t2_cpl_clamped")
    assert list(left["position_id"]) == [2] and list(right["position_id"]) == [2]


def test_pooled_measures_from_paired_rows():
    rows = _rows("board_text", "m", range(4), t2_cpl_clamped=[0.0, 1000.0, 1000.0, 10.0],
                 t2_first_attempt_legal=[1.0, 0.0, 0.0, 0.0],
                 t1_model_eval=[1, 2, 3, 4], t1_stockfish_eval=[10, 20, 30, 40],
                 t3_side_claimed=["White", "Black", "White", "Black"])
    measures = pooled_measures(rows)
    assert measures["blunder_rate"] == 0.5
    assert measures["first_answer_legal"] == 0.25
    assert measures["t1_rho"] == pytest.approx(1.0)


def _verdict_inputs(drop=0.01, rho=0.1, kappa=0.1, legal=0.1, separate=False):
    return {"blunder_drop": drop, "blunder_cis_separate": separate,
            "t1_rho": rho, "side_kappa": kappa, "first_answer_legal": legal}


def test_verdict_not_barrier_when_all_four_hold():
    assert verdict(_verdict_inputs()) == NOT_BARRIER


def test_verdict_major_on_large_separate_blunder_drop_or_high_rho():
    assert verdict(_verdict_inputs(drop=0.06, separate=True)) == MAJOR_BARRIER
    assert verdict(_verdict_inputs(rho=0.31)) == MAJOR_BARRIER


def test_verdict_mixed_otherwise():
    assert verdict(_verdict_inputs(drop=0.06, separate=False)) == MIXED
    assert verdict(_verdict_inputs(kappa=0.25)) == MIXED
    assert verdict(_verdict_inputs(legal=0.3)) == MIXED


def test_legal_for_other_side():
    assert legal_for_other_side(START_FEN, "e5")
    assert not legal_for_other_side(START_FEN, "e4")
    assert not legal_for_other_side(START_FEN, "Qz9")


def test_bootstrap_is_repeatable_with_the_seed():
    rows = _rows("board_text", "m", range(40), t2_cpl_clamped=[1000.0 if i % 3 else 0.0 for i in range(40)])
    first = bootstrap_ci(rows, lambda r: float((r["t2_cpl_clamped"] >= 1000).mean()), seed=42, resamples=200)
    again = bootstrap_ci(rows, lambda r: float((r["t2_cpl_clamped"] >= 1000).mean()), seed=42, resamples=200)
    assert first == again and first[0] < 2 / 3 < first[1]


# ---------- probe ----------

def test_truncated_when_prompt_fills_the_context_or_count_is_unknown():
    assert not truncated(900, 1024)
    assert truncated(1024, 1024)
    assert truncated(None, 1024)


def test_longest_prompt_position_has_the_most_text():
    positions = [{"fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1"}, {"fen": START_FEN}]
    assert longest_prompt_position(positions)["fen"] == START_FEN
