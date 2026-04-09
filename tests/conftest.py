"""Pytest fixtures for Chess LLM Benchmark tests."""

import pytest


@pytest.fixture
def sample_position():
    """Sample chess position for testing."""
    return {
        "id": 1,
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "pgn_moves": "1. e4 e5 2. Nf3 Nc6 3. Bc4",
        "theme": "pin",
        "difficulty": "medium",
        "phase": "opening",
        "source": "lichess_puzzles",
        "stockfish_eval": 50,
        "stockfish_best_move": "Nf6",
    }


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for testing."""
    return """Eval: 45
Move: Nf6
Explanation: Equal — Both sides have developed pieces normally in this Italian Game opening."""


@pytest.fixture
def sample_parsed_response():
    """Parsed LLM response for testing."""
    return {
        "eval": 45,
        "move": "Nf6",
        "explanation": "Equal — Both sides have developed pieces normally in this Italian Game opening.",
        "side_claimed": "Equal",
        "parse_errors": [],
    }


@pytest.fixture
def sample_job():
    """Sample benchmark job for testing."""
    return {
        "job_id": "job_00001_qwen2_5_7b_pgn+fen_1",
        "job_type": "standard",
        "position_id": 1,
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "pgn_moves": "1. e4 e5 2. Nf3 Nc6 3. Bc4",
        "model": "qwen2.5:7b",
        "prompt_format": "pgn+fen",
        "difficulty": "medium",
        "phase": "opening",
        "source": "lichess_puzzles",
        "theme": "pin",
        "trial": 1,
        "hash": "abc123def456",
    }
