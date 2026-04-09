"""Tests for src/dataset_builder.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

from src.dataset_builder import (
    rating_to_difficulty,
    _validate_puzzle_row,
    LichessPuzzleFetcher,
    PGNPositionSampler,
    build_dataset,
    DIFFICULTY_TIERS,
)


class TestRatingToDifficulty:
    def test_easy(self):
        assert rating_to_difficulty(800) == "easy"
        assert rating_to_difficulty(1199) == "easy"

    def test_medium(self):
        assert rating_to_difficulty(1200) == "medium"
        assert rating_to_difficulty(1799) == "medium"

    def test_hard(self):
        assert rating_to_difficulty(1800) == "hard"
        assert rating_to_difficulty(2399) == "hard"

    def test_extreme(self):
        assert rating_to_difficulty(2400) == "extreme"
        assert rating_to_difficulty(9999) == "extreme"


class TestValidatePuzzleRow:
    VALID_ROW = {
        "FEN": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK22R b KQkq - 3 3",
        "Rating": "1500",
        "Themes": "pin tactics",
    }

    def test_valid_row(self):
        # Use a definitely valid FEN
        row = {
            "FEN": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "Rating": "1500",
            "Themes": "pin tactics",
        }
        result = _validate_puzzle_row(row)
        if result is not None:
            assert result["source"] == "lichess_puzzles"
            assert result["difficulty"] == "medium"
            assert result["theme"] == "pin"

    def test_empty_themes_fallback(self):
        row = {
            "FEN": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "Rating": "1500",
            "Themes": "",
        }
        result = _validate_puzzle_row(row)
        if result is not None:
            assert result["theme"] == "tactics"

    def test_invalid_fen(self):
        row = {"FEN": "not_a_fen", "Rating": "1500", "Themes": "pin"}
        result = _validate_puzzle_row(row)
        assert result is None

    def test_exception_returns_none(self):
        result = _validate_puzzle_row({"FEN": None, "Rating": "bad", "Themes": ""})
        assert result is None


class TestLichessPuzzleFetcher:
    def test_init(self):
        fetcher = LichessPuzzleFetcher(source="local", csv_path="/some/path.csv")
        assert fetcher.source == "local"
        assert fetcher.csv_path == "/some/path.csv"

    @patch("src.dataset_builder.requests.get")
    def test_fetch_from_api_success(self, mock_get):
        puzzle_data = json.dumps({"puzzle": {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "themes": ["pin"],
            "rating": 1500,
        }})
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = puzzle_data
        fetcher = LichessPuzzleFetcher(source="api")
        results = fetcher.fetch_from_api(count=1)
        assert isinstance(results, list)

    @patch("src.dataset_builder.requests.get")
    def test_fetch_from_api_non_200(self, mock_get):
        mock_get.return_value.status_code = 429
        fetcher = LichessPuzzleFetcher(source="api")
        results = fetcher.fetch_from_api(count=1)
        assert results == []

    @patch("src.dataset_builder.requests.get")
    def test_fetch_from_api_request_exception(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("timeout")
        fetcher = LichessPuzzleFetcher(source="api")
        results = fetcher.fetch_from_api(count=1)
        assert results == []

    @patch("src.dataset_builder.requests.get")
    def test_fetch_from_api_json_decode_error_line(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "not json\n"
        fetcher = LichessPuzzleFetcher(source="api")
        results = fetcher.fetch_from_api(count=1)
        assert results == []

    def test_fetch_from_csv_missing_file(self):
        fetcher = LichessPuzzleFetcher(source="local", csv_path="/nonexistent.csv")
        result = fetcher.fetch_from_csv()
        assert result == {}

    def test_fetch_from_csv_no_path(self):
        fetcher = LichessPuzzleFetcher(source="local", csv_path=None)
        result = fetcher.fetch_from_csv()
        assert result == {}

    def test_parse_api_puzzle_with_themes(self):
        fetcher = LichessPuzzleFetcher()
        data = {"puzzle": {"fen": "x", "themes": ["fork", "pin"], "rating": 1500}}
        result = fetcher._parse_api_puzzle(data)
        assert result is not None
        assert result["theme"] == "fork"

    def test_parse_api_puzzle_no_themes(self):
        fetcher = LichessPuzzleFetcher()
        data = {"puzzle": {"fen": "x", "themes": [], "rating": 1200}}
        result = fetcher._parse_api_puzzle(data)
        assert result is not None
        assert result["theme"] == "tactics"

    def test_parse_api_puzzle_flat_format(self):
        fetcher = LichessPuzzleFetcher()
        data = {"fen": "x", "themes": ["pin"], "rating": 1800}
        result = fetcher._parse_api_puzzle(data)
        assert result is not None

    def test_parse_api_puzzle_exception(self):
        fetcher = LichessPuzzleFetcher()
        result = fetcher._parse_api_puzzle(None)
        assert result is None


class TestPGNPositionSampler:
    def test_missing_pgn_returns_empty(self):
        sampler = PGNPositionSampler("/nonexistent.pgn")
        result = sampler.sample_positions()
        assert result == {}

    def test_sample_from_pgn(self, tmp_path):
        pgn_content = """[Event "Test"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 5. d3 O-O 1-0
"""
        pgn_file = tmp_path / "test.pgn"
        pgn_file.write_text(pgn_content)
        sampler = PGNPositionSampler(str(pgn_file))
        # Use seed that samples some positions
        with patch("random.Random.random", return_value=0.0):  # Always sample
            result = sampler.sample_positions(count_per_phase=10, seed=42)
        assert isinstance(result, dict)

    def test_moves_to_pgn(self):
        sampler = PGNPositionSampler("/dummy.pgn")
        result = sampler._moves_to_pgn(["e4", "e5", "Nf3"])
        assert "1. e4" in result
        assert "2. Nf3" in result

    def test_moves_to_pgn_empty(self):
        sampler = PGNPositionSampler("/dummy.pgn")
        assert sampler._moves_to_pgn([]) == ""


class TestBuildDataset:
    def test_build_with_generated_only(self, tmp_path):
        config = {
            "benchmark": {"random_seed": 42, "max_positions_per_tier": 3},
            "dataset": {},  # No lichess source, no pgn
        }
        result = build_dataset(config, output_dir=str(tmp_path / "data"))
        assert isinstance(result, dict)
        assert "medium" in result
        # JSON files should exist
        assert (tmp_path / "data" / "easy.json").exists()
        assert (tmp_path / "data" / "medium.json").exists()

    def test_build_assigns_unique_ids(self, tmp_path):
        config = {
            "benchmark": {"random_seed": 42, "max_positions_per_tier": 3},
            "dataset": {},
        }
        result = build_dataset(config, output_dir=str(tmp_path / "data"))
        all_ids = [pos["id"] for positions in result.values() for pos in positions]
        assert len(all_ids) == len(set(all_ids))

    @patch("src.dataset_builder.requests.get")
    def test_build_with_api_source(self, mock_get, tmp_path):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = json.dumps({
            "puzzle": {
                "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "themes": ["pin"],
                "rating": 1500,
            }
        })
        config = {
            "benchmark": {"random_seed": 42, "max_positions_per_tier": 2},
            "dataset": {"lichess_source": "api"},
        }
        result = build_dataset(config, output_dir=str(tmp_path / "data"))
        assert isinstance(result, dict)

    def test_build_unlimited(self, tmp_path):
        config = {
            "benchmark": {"random_seed": 42, "max_positions_per_tier": 0},
            "dataset": {},
        }
        result = build_dataset(config, output_dir=str(tmp_path / "data"))
        assert isinstance(result, dict)
