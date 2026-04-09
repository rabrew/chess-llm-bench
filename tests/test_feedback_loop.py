"""Tests for src/feedback_loop.py"""

import pytest
from unittest.mock import MagicMock, patch

from src.feedback_loop import (
    build_correction_prompt,
    select_follow_up_position,
    calculate_learning_delta,
    calculate_net_feedback_effect,
    CorrectionLoopManager,
)
from src.data_loader import DataLoader
from src.job_queue import JobQueue


FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
FOLLOW_UP = {"id": 2, "fen": FEN, "pgn_moves": "1. e4 e5"}


class TestBuildCorrectionPrompt:
    def test_includes_original_fen(self):
        prompt = build_correction_prompt(FEN, "Qd4", "Nf6", FOLLOW_UP)
        assert FEN in prompt

    def test_includes_model_move(self):
        prompt = build_correction_prompt(FEN, "Qd4", "Nf6", FOLLOW_UP)
        assert "Qd4" in prompt

    def test_includes_best_move(self):
        prompt = build_correction_prompt(FEN, "Qd4", "Nf6", FOLLOW_UP)
        assert "Nf6" in prompt

    def test_includes_follow_up_fen(self):
        prompt = build_correction_prompt(FEN, "Qd4", "Nf6", FOLLOW_UP)
        assert FOLLOW_UP["fen"] in prompt

    def test_custom_prompt_format(self):
        prompt = build_correction_prompt(FEN, "Qd4", "Nf6", FOLLOW_UP, prompt_format="fen_only")
        assert FEN in prompt


class TestSelectFollowUpPosition:
    def test_returns_similar_position(self):
        mock_loader = MagicMock(spec=DataLoader)
        similar = {"id": 99, "fen": "different_fen", "theme": "pin", "difficulty": "easy"}
        mock_loader.get_similar.return_value = similar

        original = {"id": 1, "theme": "pin", "difficulty": "easy", "fen": FEN}
        result = select_follow_up_position(original, mock_loader, set(), seed=42)
        assert result == similar

    def test_excludes_original_id(self):
        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.get_similar.return_value = None

        original = {"id": 5, "theme": "pin", "difficulty": "easy", "fen": FEN}
        select_follow_up_position(original, mock_loader, set(), seed=42)

        # The exclude_ids passed to get_similar should contain original id
        call_kwargs = mock_loader.get_similar.call_args
        exclude_ids = call_kwargs[1].get("exclude_ids") or call_kwargs[0][1]
        assert 5 in exclude_ids

    def test_returns_none_when_no_position_available(self):
        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.get_similar.return_value = None

        original = {"id": 1, "theme": "rare_theme", "difficulty": "extreme"}
        result = select_follow_up_position(original, mock_loader, set(), seed=42)
        assert result is None

    def test_existing_exclude_ids_preserved(self):
        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.get_similar.return_value = None

        original = {"id": 1, "theme": "pin", "difficulty": "easy"}
        select_follow_up_position(original, mock_loader, {10, 20}, seed=42)
        call_kwargs = mock_loader.get_similar.call_args
        exclude_ids = call_kwargs[1].get("exclude_ids") or call_kwargs[0][1]
        assert 10 in exclude_ids
        assert 20 in exclude_ids


class TestCalculateLearningDelta:
    def test_improvement(self):
        assert calculate_learning_delta(100, 40) == 60

    def test_no_improvement(self):
        assert calculate_learning_delta(40, 100) == -60

    def test_none_first(self):
        assert calculate_learning_delta(None, 40) is None

    def test_none_second(self):
        assert calculate_learning_delta(40, None) is None

    def test_both_none(self):
        assert calculate_learning_delta(None, None) is None

    def test_zero_delta(self):
        assert calculate_learning_delta(50, 50) == 0


class TestCalculateNetFeedbackEffect:
    def test_positive_effect(self):
        assert calculate_net_feedback_effect(60, 20) == 40

    def test_negative_effect(self):
        assert calculate_net_feedback_effect(10, 30) == -20

    def test_none_correction(self):
        assert calculate_net_feedback_effect(None, 20) is None

    def test_none_control(self):
        assert calculate_net_feedback_effect(40, None) is None

    def test_both_none(self):
        assert calculate_net_feedback_effect(None, None) is None


class TestCorrectionLoopManager:
    @pytest.fixture
    def mock_loader(self):
        loader = MagicMock(spec=DataLoader)
        loader.get_similar.return_value = {"id": 99, "fen": FEN, "theme": "pin", "difficulty": "easy"}
        return loader

    @pytest.fixture
    def mock_queue(self, tmp_path):
        db = str(tmp_path / "jobs.db")
        return JobQueue(db)

    @pytest.fixture
    def manager(self, mock_loader, mock_queue):
        config = {"correction_loop": {"enabled": True}, "benchmark": {"random_seed": 42}}
        return CorrectionLoopManager(mock_loader, mock_queue, config)

    def test_trigger_correction_disabled(self, mock_loader, mock_queue):
        config = {"correction_loop": {"enabled": False}}
        manager = CorrectionLoopManager(mock_loader, mock_queue, config)
        job = {"job_id": "j1", "position_id": 1, "fen": FEN, "model": "x",
               "prompt_format": "pgn+fen", "theme": "pin", "difficulty": "easy"}
        result = manager.trigger_correction(job, {})
        assert result is None

    def test_trigger_correction_no_follow_up(self, mock_queue):
        loader = MagicMock(spec=DataLoader)
        loader.get_similar.return_value = None
        config = {"correction_loop": {"enabled": True}, "benchmark": {"random_seed": 42}}
        manager = CorrectionLoopManager(loader, mock_queue, config)

        job = {"job_id": "j1", "position_id": 1, "fen": FEN, "model": "x",
               "prompt_format": "pgn+fen", "theme": "pin", "difficulty": "easy"}
        result = manager.trigger_correction(job, {})
        assert result is None

    def test_trigger_correction_success(self, manager):
        job = {"job_id": "j1", "position_id": 1, "fen": FEN, "model": "llama3.2:3b",
               "prompt_format": "pgn+fen", "theme": "pin", "difficulty": "easy"}
        result = manager.trigger_correction(job, {})
        assert result is not None
        corr_id, ctrl_id = result
        assert "correction" in corr_id
        assert "control" in ctrl_id

    def test_tracks_used_positions(self, manager):
        job = {"job_id": "j1", "position_id": 1, "fen": FEN, "model": "llama3.2:3b",
               "prompt_format": "pgn+fen", "theme": "pin", "difficulty": "easy"}
        manager.trigger_correction(job, {})
        assert 99 in manager.used_positions

    def test_get_correction_prompt(self, manager):
        corr_job = {
            "fen": FEN,
            "pgn_moves": "1. e4",
            "prompt_format": "pgn+fen",
        }
        parent_result = {
            "fen": FEN,
            "t2_move": "Qd4",
            "t2_best_move": "Nf6",
        }
        prompt = manager.get_correction_prompt(corr_job, parent_result)
        assert "Qd4" in prompt
        assert "Nf6" in prompt
