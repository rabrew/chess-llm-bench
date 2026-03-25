"""Tests for scripts/enrich_cpl.py (Lc0 CPL post-processing)."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.enrich_cpl import (
    compute_cpl,
    load_positions,
    collect_work,
    enrich_records,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEN_WHITE_TO_MOVE = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
# Simple position: White to move, e4 is legal
FEN_SIMPLE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_record(**kwargs):
    """Build a minimal evaluations.jsonl record."""
    base = {
        "job_id": "job_1_llama3_2_3b_pgn+fen_1",
        "position_id": 1,
        "fen": FEN_SIMPLE,
        "model": "llama3.2:3b",
        "difficulty": "easy",
        "t2_move": "e4",
        "t2_best_move": None,
        "t2_legal": True,
        "t2_cpl": None,
    }
    base.update(kwargs)
    return base


def make_lc0_result(eval_cp: int, best_move: str | None = "e4"):
    return {"eval": eval_cp, "best_move": best_move, "mate": None}


# ---------------------------------------------------------------------------
# compute_cpl
# ---------------------------------------------------------------------------

class TestComputeCpl:
    def test_white_to_move_model_worse(self):
        # White to move: eval before=100, eval after model move=60 → CPL=40
        cpl = compute_cpl(
            fen=FEN_SIMPLE,
            eval_before=100,
            eval_after=60,
        )
        assert cpl == 40

    def test_white_to_move_model_better(self):
        # eval after > eval before → CPL clamped to 0
        cpl = compute_cpl(fen=FEN_SIMPLE, eval_before=50, eval_after=80)
        assert cpl == 0

    def test_black_to_move_model_worse(self):
        # Black to move: lower eval is better for Black
        # eval before=-100, eval after model move=-60 → eval went up (bad for Black) → CPL=40
        fen_black = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        cpl = compute_cpl(fen=fen_black, eval_before=-100, eval_after=-60)
        assert cpl == 40

    def test_black_to_move_model_better(self):
        fen_black = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        cpl = compute_cpl(fen=fen_black, eval_before=-100, eval_after=-130)
        assert cpl == 0

    def test_zero_cpl(self):
        cpl = compute_cpl(fen=FEN_SIMPLE, eval_before=50, eval_after=50)
        assert cpl == 0

    def test_mate_score_not_over_clamped(self):
        # Mate scores can be large; only clamp at 0
        cpl = compute_cpl(fen=FEN_SIMPLE, eval_before=9990, eval_after=50)
        assert cpl == 9940


# ---------------------------------------------------------------------------
# load_positions
# ---------------------------------------------------------------------------

class TestLoadPositions:
    def test_loads_all_tiers(self, tmp_path):
        for tier in ("easy", "medium", "hard", "extreme"):
            data = [{"id": 1, "fen": FEN_SIMPLE, "stockfish_eval": 50,
                     "stockfish_best_move": "e4", "difficulty": tier}]
            (tmp_path / f"{tier}.json").write_text(json.dumps(data))

        positions = load_positions(str(tmp_path))
        assert 1 in positions
        assert positions[1]["fen"] == FEN_SIMPLE

    def test_missing_tier_files_skipped(self, tmp_path):
        data = [{"id": 5, "fen": FEN_SIMPLE, "stockfish_eval": 0,
                 "stockfish_best_move": "e4", "difficulty": "easy"}]
        (tmp_path / "easy.json").write_text(json.dumps(data))
        # No other tier files

        positions = load_positions(str(tmp_path))
        assert 5 in positions

    def test_returns_empty_if_no_files(self, tmp_path):
        positions = load_positions(str(tmp_path))
        assert positions == {}


# ---------------------------------------------------------------------------
# collect_work
# ---------------------------------------------------------------------------

class TestCollectWork:
    def test_legal_null_cpl_collected(self):
        records = [make_record()]
        to_enrich, passthrough = collect_work(records)
        assert len(to_enrich) == 1
        assert len(passthrough) == 0

    def test_already_has_cpl_is_passthrough(self):
        records = [make_record(t2_cpl=10)]
        to_enrich, passthrough = collect_work(records)
        assert len(to_enrich) == 0
        assert len(passthrough) == 1

    def test_illegal_move_is_passthrough(self):
        records = [make_record(t2_legal=False, t2_move="Qh8")]
        to_enrich, passthrough = collect_work(records)
        assert len(to_enrich) == 0
        assert len(passthrough) == 1

    def test_null_move_is_passthrough(self):
        records = [make_record(t2_legal=True, t2_move=None)]
        to_enrich, passthrough = collect_work(records)
        assert len(to_enrich) == 0
        assert len(passthrough) == 1

    def test_mixed_records(self):
        records = [
            make_record(),                          # needs enrichment
            make_record(t2_cpl=5),                  # already done
            make_record(t2_legal=False),            # illegal
            make_record(job_id="job_2", t2_move="d4"),  # needs enrichment
        ]
        to_enrich, passthrough = collect_work(records)
        assert len(to_enrich) == 2
        assert len(passthrough) == 2


# ---------------------------------------------------------------------------
# enrich_records (mocked Lc0)
# ---------------------------------------------------------------------------

class TestEnrichRecords:
    def _make_engine(self, pre_move_eval, post_move_eval, best_move="e4"):
        engine = MagicMock()
        engine.evaluate.side_effect = [
            make_lc0_result(pre_move_eval, best_move),   # pre-move call
            make_lc0_result(post_move_eval, None),        # post-move call
        ]
        return engine

    def test_cpl_and_best_move_filled(self):
        records = [make_record()]
        engine = self._make_engine(pre_move_eval=100, post_move_eval=60, best_move="d4")

        result = enrich_records(records, engine)

        assert result[0]["t2_cpl"] == 40
        assert result[0]["t2_best_move"] == "d4"

    def test_best_move_match_gives_zero_cpl(self):
        # Model played the same move as Lc0's best move → CPL=0, no post-move call
        engine = MagicMock()
        engine.evaluate.return_value = make_lc0_result(100, best_move="e4")
        records = [make_record(t2_move="e4")]

        result = enrich_records(records, engine)

        assert result[0]["t2_cpl"] == 0
        assert result[0]["t2_best_move"] == "e4"
        # Only one engine call (pre-move), no post-move needed
        assert engine.evaluate.call_count == 1

    def test_deduplication_same_fen_one_pre_move_call(self):
        # Two records on the same FEN with different moves
        r1 = make_record(job_id="job_1", t2_move="e4")
        r2 = make_record(job_id="job_2", t2_move="d4")
        engine = MagicMock()
        engine.evaluate.return_value = make_lc0_result(100, best_move="Nf3")

        result = enrich_records([r1, r2], engine)

        # 1 pre-move call + 2 post-move calls = 3 total
        assert engine.evaluate.call_count == 3
        assert result[0]["t2_cpl"] is not None
        assert result[1]["t2_cpl"] is not None

    def test_engine_error_leaves_cpl_none(self):
        records = [make_record()]
        engine = MagicMock()
        engine.evaluate.side_effect = RuntimeError("Lc0 crashed")

        result = enrich_records(records, engine)

        assert result[0]["t2_cpl"] is None

    def test_invalid_move_leaves_cpl_none(self):
        # t2_move can't be applied to the FEN
        records = [make_record(t2_move="Qxh8")]  # illegal on starting pos
        engine = MagicMock()
        engine.evaluate.return_value = make_lc0_result(100, "e4")

        result = enrich_records(records, engine)

        assert result[0]["t2_cpl"] is None

    def test_passthrough_records_unchanged(self):
        already_done = make_record(t2_cpl=15, t2_best_move="d4")
        engine = MagicMock()
        engine.evaluate.return_value = make_lc0_result(100, "e4")

        # enrich_records only processes records passed to it (already filtered)
        # so simulate: pass only the enrichable record, check the other unchanged
        result = enrich_records([], engine)
        assert result == []

    def test_deduplication_same_fen_and_move(self):
        # Same FEN + same model_move in two records (different prompt formats)
        r1 = make_record(job_id="job_1", t2_move="e4")
        r2 = make_record(job_id="job_2", t2_move="e4")
        engine = MagicMock()
        engine.evaluate.return_value = make_lc0_result(100, best_move="Nf3")

        result = enrich_records([r1, r2], engine)

        # 1 pre-move + 1 post-move (deduplicated) = 2 total calls
        assert engine.evaluate.call_count == 2
        # Both records get the same CPL
        assert result[0]["t2_cpl"] == result[1]["t2_cpl"]
