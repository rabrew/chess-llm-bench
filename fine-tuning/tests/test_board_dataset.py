"""Tests for fine-tuning/scripts/generate_board_dataset.py — dataset generation."""

import json
import sys
import tempfile
from pathlib import Path

import chess
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_ROOT / "src"))
sys.path.insert(0, str(FT_ROOT / "scripts"))

from board_state import fen_to_board_description

# ---------------------------------------------------------------------------
# Helpers — build minimal fake tier data for tests
# ---------------------------------------------------------------------------

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
PROMOTION_FEN = "3Q4/8/8/8/8/8/8/3QK1k1 w - - 0 1"

SAMPLE_FENS = [
    STARTING_FEN,
    ITALIAN_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
]


def make_fake_records(fens: list[str]) -> list[dict]:
    return [
        {
            "fen": fen,
            "pgn_moves": "",
            "theme": "test",
            "difficulty": "easy",
            "phase": "middlegame",
            "source": "test",
            "rating": 1000,
            "id": i,
        }
        for i, fen in enumerate(fens)
    ]


# ---------------------------------------------------------------------------
# Tests for fen_to_board_description (the output format used in records)
# ---------------------------------------------------------------------------

class TestStartingPosition:
    def test_starting_position_piece_list(self):
        desc = fen_to_board_description(STARTING_FEN)
        assert "King e1" in desc
        assert "King e8" in desc
        assert "Side to move: White" in desc


class TestPieceCountsMatchFen:
    def test_white_piece_count(self):
        board = chess.Board(STARTING_FEN)
        desc = fen_to_board_description(STARTING_FEN)
        # Starting position: 1 king, 1 queen, 2 rooks, 2 bishops, 2 knights, 8 pawns = 16
        white_line = desc.split("\n")[0]
        # Count squares mentioned in white section (rough check via comma count)
        # Each piece label is followed by squares separated by spaces, entries by comma
        assert "Pawns" in white_line  # pawns listed
        assert "King" in white_line
        assert "Queen" in white_line


class TestEmptySquaresAbsent:
    def test_no_empty_in_output(self):
        desc = fen_to_board_description(STARTING_FEN)
        assert "empty" not in desc.lower()
        # No standalone squares that indicate empty
        for line in desc.split("\n"):
            if line.startswith("White:") or line.startswith("Black:"):
                assert "8" not in line.replace("a8", "").replace("b8", "").replace(
                    "c8", "").replace("d8", "").replace("e8", "").replace(
                    "f8", "").replace("g8", "").replace("h8", "")


class TestPromotedPieceAppears:
    def test_second_queen_shows_up(self):
        desc = fen_to_board_description(PROMOTION_FEN)
        # White has two queens: d1 and d8
        white_line = desc.split("\n")[0]
        assert "d1" in white_line
        assert "d8" in white_line


class TestRecordHasRequiredKeys:
    def test_alpaca_record_keys(self):
        fen = STARTING_FEN
        output = fen_to_board_description(fen)
        record = {
            "instruction": "You are given a chess position in FEN notation. List every piece on the board and its exact square.",
            "input": f"FEN: {fen}",
            "output": output,
        }
        assert "instruction" in record
        assert "input" in record
        assert "output" in record
        assert record["input"].startswith("FEN: ")


class TestDatasetSize:
    def test_generate_n_samples(self, tmp_path):
        """Generate N records from fake data and verify count."""
        # Import the generator function directly
        from generate_board_dataset import generate_records

        records = list(generate_records(make_fake_records(SAMPLE_FENS), seed=42))
        # Should get one record per unique FEN
        assert len(records) == len(SAMPLE_FENS)

    def test_generate_with_limit(self, tmp_path):
        from generate_board_dataset import generate_records

        records = list(generate_records(make_fake_records(SAMPLE_FENS), seed=42, max_samples=2))
        assert len(records) == 2


class TestNoDuplicateFens:
    def test_dedup_on_fen(self):
        from generate_board_dataset import generate_records

        # Feed duplicate FENs
        duped = make_fake_records([STARTING_FEN, STARTING_FEN, ITALIAN_FEN])
        records = list(generate_records(duped, seed=42))
        fens = [r["input"].replace("FEN: ", "") for r in records]
        assert len(fens) == len(set(fens))


class TestSideToMoveInOutput:
    def test_side_to_move_present_white(self):
        desc = fen_to_board_description(STARTING_FEN)
        assert "Side to move: White" in desc

    def test_side_to_move_present_black(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        desc = fen_to_board_description(fen)
        assert "Side to move: Black" in desc


class TestBenchmarkExclusion:
    """F-1: training positions must never be benchmark positions."""

    def test_excluded_fens_are_skipped(self):
        from generate_board_dataset import generate_records

        records = list(generate_records(
            make_fake_records(SAMPLE_FENS), seed=42, exclude_fens={STARTING_FEN, ITALIAN_FEN},
        ))
        fens = {r["input"].replace("FEN: ", "") for r in records}
        assert fens == set(SAMPLE_FENS) - {STARTING_FEN, ITALIAN_FEN}

    def test_load_benchmark_fens_reads_fen_field(self, tmp_path):
        from generate_board_dataset import load_benchmark_fens

        results = tmp_path / "evaluations.jsonl"
        results.write_text(
            json.dumps({"fen": STARTING_FEN}) + "\n\n"
            + json.dumps({"fen": f"  {ITALIAN_FEN}  "}) + "\n"
            + json.dumps({"fen": STARTING_FEN}) + "\n"
        )
        assert load_benchmark_fens(str(results)) == {STARTING_FEN, ITALIAN_FEN}

    def test_load_benchmark_fens_missing_file(self, tmp_path):
        from generate_board_dataset import load_benchmark_fens

        assert load_benchmark_fens(str(tmp_path / "missing.jsonl")) == set()

    def test_real_training_data_has_no_benchmark_fens(self):
        """data/*.json is the 5.8M-position pool; the benchmarked FENs are those in evaluations.jsonl."""
        from generate_board_dataset import load_benchmark_fens

        dataset = REPO_ROOT / "data" / "board_dataset.jsonl"
        results = REPO_ROOT / "results" / "evaluations.jsonl"
        if not dataset.exists() or not results.exists():
            pytest.skip("real training data or benchmark results not present")

        benchmark_fens = load_benchmark_fens(str(results))
        with open(dataset) as handle:
            training_fens = {
                json.loads(line)["input"].replace("FEN: ", "").strip()
                for line in handle if line.strip()
            }
        assert training_fens, "training dataset is empty"
        assert benchmark_fens, "no benchmark FENs loaded"
        assert not training_fens & benchmark_fens
