"""Tests for data loader."""

import json
import os
import tempfile

import pytest

from src.data_loader import DataLoader


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample dataset
        positions = [
            {
                "id": 1,
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "theme": "opening",
                "difficulty": "easy",
                "phase": "opening",
                "source": "generated",
            },
            {
                "id": 2,
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "theme": "tactics",
                "difficulty": "easy",
                "phase": "opening",
                "source": "lichess_puzzles",
            },
        ]

        with open(os.path.join(tmpdir, "easy.json"), "w") as f:
            json.dump(positions, f)

        with open(os.path.join(tmpdir, "medium.json"), "w") as f:
            json.dump([], f)

        yield tmpdir


class TestDataLoader:
    def test_load_tier(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.load_tier("easy")
        assert len(positions) == 2

    def test_load_missing_tier(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.load_tier("extreme")  # Doesn't exist
        assert len(positions) == 0

    def test_filter_by_source(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.filter(source="lichess_puzzles")
        assert len(positions) == 1
        assert positions[0]["id"] == 2

    def test_filter_by_theme(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.filter(theme="opening")
        assert len(positions) == 1
        assert positions[0]["id"] == 1

    def test_get_by_id(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        pos = loader.get_by_id(2)
        assert pos is not None
        assert pos["theme"] == "tactics"

    def test_get_by_id_not_found(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        pos = loader.get_by_id(999)
        assert pos is None

    def test_sample(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.sample(count=1, seed=42)
        assert len(positions) == 1

    def test_get_stats(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        stats = loader.get_stats()
        assert stats["total"] == 2
        assert stats["by_difficulty"]["easy"] == 2

    def test_cache_hit(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        p1 = loader.load_tier("easy")
        p2 = loader.load_tier("easy")  # Should hit cache (line 34)
        assert p1 is p2

    def test_filter_by_phase(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.filter(phase="opening")
        assert len(positions) == 2

    def test_filter_by_multiple_phases(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.filter(phase=["opening", "endgame"])
        assert len(positions) == 2

    def test_filter_phase_no_match(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        positions = loader.filter(phase="endgame")
        assert len(positions) == 0

    def test_filter_with_explicit_positions(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        all_pos = loader.load_tier("easy")
        filtered = loader.filter(positions=all_pos, source="lichess_puzzles")
        assert len(filtered) == 1

    def test_sample_with_filter_kwargs(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        # filter_kwargs path (line 131)
        result = loader.sample(count=10, seed=42, source="lichess_puzzles")
        assert len(result) == 1

    def test_sample_returns_all_when_small(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        result = loader.sample(count=100, seed=42)  # More than available → returns all
        assert len(result) == 2

    def test_get_similar_same_theme_difficulty(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        pos = {"id": 1, "theme": "tactics", "difficulty": "easy", "fen": "x"}
        result = loader.get_similar(pos, seed=42)
        assert result is not None
        assert result["id"] != 1

    def test_get_similar_fallback_to_difficulty_only(self, temp_data_dir):
        """Theme doesn't match anything, falls back to difficulty-only."""
        loader = DataLoader(temp_data_dir)
        pos = {"id": 1, "theme": "rare_theme_xyz", "difficulty": "easy", "fen": "x"}
        result = loader.get_similar(pos, seed=42)
        # Falls back to same difficulty — should find pos id=2
        assert result is not None

    def test_get_similar_returns_none_when_no_candidates(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        # Use a difficulty that has no other positions
        pos = {"id": 1, "theme": "opening", "difficulty": "extreme", "fen": "x"}
        result = loader.get_similar(pos, seed=42)
        assert result is None

    def test_get_similar_excludes_ids(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        pos = {"id": 1, "theme": "tactics", "difficulty": "easy", "fen": "x"}
        # Exclude the only other candidate
        result = loader.get_similar(pos, exclude_ids={2}, seed=42)
        assert result is None

    def test_clear_cache(self, temp_data_dir):
        loader = DataLoader(temp_data_dir)
        loader.load_tier("easy")
        assert "easy" in loader._cache
        loader.clear_cache()
        assert loader._cache == {}
