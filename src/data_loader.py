"""Load and filter chess position datasets."""

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger("chess_llm_bench")


class DataLoader:
    """Load and filter chess position datasets."""

    def __init__(self, data_dir: str = "data"):
        """Initialize data loader.

        Args:
            data_dir: Directory containing dataset JSON files
        """
        self.data_dir = Path(data_dir)
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._id_index: dict[int, dict[str, Any]] | None = None

    def load_tier(self, difficulty: str) -> list[dict[str, Any]]:
        """Load positions for a specific difficulty tier.

        Args:
            difficulty: Difficulty tier (easy, medium, hard, extreme)

        Returns:
            List of position dictionaries
        """
        if difficulty in self._cache:
            return self._cache[difficulty]

        file_path = self.data_dir / f"{difficulty}.json"
        if not file_path.exists():
            logger.warning(f"Dataset file not found: {file_path}")
            return []

        with open(file_path, "r") as f:
            positions = json.load(f)

        self._cache[difficulty] = positions
        logger.debug(f"Loaded {len(positions)} positions from {file_path}")
        return positions

    def load_all(self) -> list[dict[str, Any]]:
        """Load all positions from all difficulty tiers.

        Returns:
            Combined list of all positions
        """
        all_positions = []
        for tier in ["easy", "medium", "hard", "extreme"]:
            all_positions.extend(self.load_tier(tier))
        return all_positions

    def filter(
        self,
        positions: list[dict[str, Any]] | None = None,
        difficulty: str | list[str] | None = None,
        phase: str | list[str] | None = None,
        source: str | list[str] | None = None,
        theme: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter positions by multiple criteria.

        Args:
            positions: Positions to filter (loads all if None)
            difficulty: Filter by difficulty tier(s)
            phase: Filter by game phase(s)
            source: Filter by source(s)
            theme: Filter by theme(s)

        Returns:
            Filtered list of positions
        """
        if positions is None:
            positions = self.load_all()

        # Normalize filter values to lists
        def to_list(val):
            if val is None:
                return None
            if isinstance(val, str):
                return [val]
            return list(val)

        difficulties = to_list(difficulty)
        phases = to_list(phase)
        sources = to_list(source)
        themes = to_list(theme)

        filtered = []
        for pos in positions:
            if difficulties and pos.get("difficulty") not in difficulties:
                continue
            if phases and pos.get("phase") not in phases:
                continue
            if sources and pos.get("source") not in sources:
                continue
            if themes and pos.get("theme") not in themes:
                continue
            filtered.append(pos)

        return filtered

    def sample(
        self,
        positions: list[dict[str, Any]] | None = None,
        count: int = 100,
        seed: int = 42,
        **filter_kwargs,
    ) -> list[dict[str, Any]]:
        """Sample positions with optional filtering.

        Args:
            positions: Positions to sample from (loads all if None)
            count: Number of positions to sample
            seed: Random seed for reproducibility
            **filter_kwargs: Additional filter arguments

        Returns:
            Sampled list of positions
        """
        if positions is None:
            positions = self.load_all()

        if filter_kwargs:
            positions = self.filter(positions, **filter_kwargs)

        if len(positions) <= count:
            return positions

        rng = random.Random(seed)
        return rng.sample(positions, count)

    def get_by_id(self, position_id: int) -> dict[str, Any] | None:
        """Get a position by its ID.

        Args:
            position_id: Unique position identifier

        Returns:
            Position dictionary or None if not found
        """
        if self._id_index is None:
            self._id_index = {pos["id"]: pos for pos in self.load_all() if "id" in pos}
        return self._id_index.get(position_id)

    def get_similar(
        self,
        position: dict[str, Any],
        exclude_ids: set[int] | None = None,
        seed: int = 42,
    ) -> dict[str, Any] | None:
        """Get a position similar to the given one (same theme and difficulty).

        Args:
            position: Reference position
            exclude_ids: Position IDs to exclude
            seed: Random seed

        Returns:
            Similar position or None if not found
        """
        exclude_ids = exclude_ids or set()
        theme = position.get("theme")
        difficulty = position.get("difficulty")

        candidates = self.filter(
            theme=theme,
            difficulty=difficulty,
        )

        # Filter out excluded positions
        candidates = [
            p for p in candidates
            if p.get("id") not in exclude_ids and p.get("id") != position.get("id")
        ]

        if not candidates:
            # Fall back to same difficulty only
            candidates = self.filter(difficulty=difficulty)
            candidates = [
                p for p in candidates
                if p.get("id") not in exclude_ids and p.get("id") != position.get("id")
            ]

        if not candidates:
            return None

        rng = random.Random(seed)
        return rng.choice(candidates)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the loaded dataset.

        Returns:
            Dictionary with dataset statistics
        """
        all_positions = self.load_all()

        stats = {
            "total": len(all_positions),
            "by_difficulty": {},
            "by_phase": {},
            "by_source": {},
            "by_theme": {},
        }

        for pos in all_positions:
            # Count by difficulty
            diff = pos.get("difficulty", "unknown")
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

            # Count by phase
            phase = pos.get("phase", "unknown")
            stats["by_phase"][phase] = stats["by_phase"].get(phase, 0) + 1

            # Count by source
            source = pos.get("source", "unknown")
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

            # Count by theme
            theme = pos.get("theme", "unknown")
            stats["by_theme"][theme] = stats["by_theme"].get(theme, 0) + 1

        return stats

    def clear_cache(self) -> None:
        """Clear the position cache."""
        self._cache.clear()
