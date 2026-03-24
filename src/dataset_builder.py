"""Build chess position datasets from multiple sources."""

import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator
import multiprocessing as mp

import chess
import chess.pgn
import pandas as pd
import requests

from .position_generator import generate_positions, validate_position, determine_phase

logger = logging.getLogger("chess_llm_bench")


# Difficulty tier boundaries (Lichess puzzle rating)
DIFFICULTY_TIERS = {
    "easy": (0, 1200),
    "medium": (1200, 1800),
    "hard": (1800, 2400),
    "extreme": (2400, 4000),
}


def rating_to_difficulty(rating: int) -> str:
    """Convert Lichess puzzle rating to difficulty tier."""
    for tier, (low, high) in DIFFICULTY_TIERS.items():
        if low <= rating < high:
            return tier
    return "extreme"


def _validate_puzzle_row(row: dict) -> dict[str, Any] | None:
    """Validate and parse a puzzle row (for parallel processing)."""
    try:
        fen = row.get("FEN", "")
        rating = int(row.get("Rating", 1500))
        themes = str(row.get("Themes", "")).split()
        primary_theme = themes[0] if themes else "tactics"

        board = chess.Board(fen)
        if not validate_position(board):
            return None

        phase = determine_phase(board, 20)

        return {
            "fen": fen,
            "pgn_moves": "",
            "theme": primary_theme,
            "difficulty": rating_to_difficulty(rating),
            "phase": phase,
            "source": "lichess_puzzles",
            "rating": rating,
        }
    except Exception:
        return None


class LichessPuzzleFetcher:
    """Fetch puzzles from Lichess API or local CSV."""

    API_URL = "https://lichess.org/api/puzzle/daily"
    PUZZLE_DB_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

    def __init__(self, source: str = "local", csv_path: str | None = None):
        """Initialize puzzle fetcher.

        Args:
            source: "api" or "local"
            csv_path: Path to local puzzle CSV file
        """
        self.source = source
        self.csv_path = csv_path

    def fetch_from_api(self, count: int = 100) -> list[dict[str, Any]]:
        """Fetch puzzles from Lichess API.

        Note: The API has rate limits and doesn't support bulk fetching easily.
        This method is primarily for testing with a few puzzles.

        Args:
            count: Number of puzzles to attempt to fetch

        Returns:
            List of puzzle position dictionaries
        """
        puzzles = []
        logger.info(f"Fetching up to {count} puzzles from Lichess API...")

        try:
            # Fetch random puzzles via the puzzle storm endpoint or similar
            # Note: Full API access may require different endpoints
            response = requests.get(
                "https://lichess.org/api/puzzle/activity",
                headers={"Accept": "application/x-ndjson"},
                timeout=30,
            )

            if response.status_code != 200:
                logger.warning(f"Lichess API returned {response.status_code}")
                return puzzles

            for line in response.text.strip().split("\n")[:count]:
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    puzzle = self._parse_api_puzzle(data)
                    if puzzle:
                        puzzles.append(puzzle)
                except json.JSONDecodeError:
                    continue

        except requests.RequestException as e:
            logger.error(f"Failed to fetch from Lichess API: {e}")

        logger.info(f"Fetched {len(puzzles)} puzzles from API")
        return puzzles

    def _parse_api_puzzle(self, data: dict) -> dict[str, Any] | None:
        """Parse a puzzle from API response."""
        try:
            puzzle_data = data.get("puzzle", data)
            return {
                "fen": puzzle_data.get("fen", ""),
                "pgn_moves": "",
                "theme": puzzle_data.get("themes", ["tactics"])[0] if puzzle_data.get("themes") else "tactics",
                "difficulty": rating_to_difficulty(puzzle_data.get("rating", 1500)),
                "phase": "middlegame",  # Most puzzles are middlegame
                "source": "lichess_puzzles",
            }
        except (KeyError, IndexError):
            return None

    def fetch_from_csv(
        self,
        count_per_tier: int = 500,
        seed: int = 42,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch puzzles from local CSV file using parallel processing.

        Args:
            count_per_tier: Number of puzzles to fetch per difficulty tier
            seed: Random seed for sampling

        Returns:
            Dictionary mapping difficulty tier to list of puzzles
        """
        if not self.csv_path or not Path(self.csv_path).exists():
            logger.error(f"Puzzle CSV not found: {self.csv_path}")
            return {}

        rng = random.Random(seed)
        puzzles_by_tier: dict[str, list[dict[str, Any]]] = {
            tier: [] for tier in DIFFICULTY_TIERS
        }

        print(f"  Loading: {self.csv_path}")

        # Use pandas for fast CSV loading - only columns we need
        print(f"  Reading CSV with pandas...", end=" ", flush=True)
        df = pd.read_csv(
            self.csv_path,
            usecols=["FEN", "Rating", "Themes"],
            dtype={"FEN": str, "Rating": int, "Themes": str},
        )
        print(f"done! ({len(df):,} puzzles)")

        # Add difficulty tier column
        df["tier"] = df["Rating"].apply(rating_to_difficulty)

        # Unlimited mode if count_per_tier <= 0
        unlimited = count_per_tier <= 0

        # Sample or use all positions per tier
        print(f"\n  Puzzles by difficulty tier:")
        sampled_rows = []
        for tier in DIFFICULTY_TIERS:
            tier_df = df[df["tier"] == tier]
            if unlimited:
                # Use all positions in this tier
                sampled_rows.append(tier_df)
                print(f"    {tier:<10} {len(tier_df):>12,} positions")
            else:
                # Sample with oversample factor
                oversample_factor = 3
                n_sample = min(len(tier_df), count_per_tier * oversample_factor)
                if n_sample > 0:
                    sampled = tier_df.sample(n=n_sample, random_state=seed)
                    sampled_rows.append(sampled)
                    print(f"    {tier:<10} {n_sample:>12,} sampled")

        if not sampled_rows:
            return puzzles_by_tier

        sampled_df = pd.concat(sampled_rows)
        n_workers = min(mp.cpu_count(), 8)
        print(f"\n  Validating {len(sampled_df):,} positions using {n_workers} CPU cores...")

        # Convert to list of dicts for parallel processing
        rows_to_validate = sampled_df.to_dict("records")

        # Parallel validation with progress bar
        from tqdm import tqdm
        valid_count = 0
        invalid_count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_validate_puzzle_row, row): row for row in rows_to_validate}
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="  Validating",
                unit="pos",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            for future in pbar:
                puzzle = future.result()
                if puzzle:
                    tier = puzzle["difficulty"]
                    if unlimited or len(puzzles_by_tier[tier]) < count_per_tier:
                        puzzles_by_tier[tier].append(puzzle)
                        valid_count += 1
                else:
                    invalid_count += 1

        total = sum(len(p) for p in puzzles_by_tier.values())
        print(f"\n  Validation complete: {valid_count:,} valid, {invalid_count:,} invalid")

        return puzzles_by_tier

    def _parse_csv_row(self, row: dict) -> dict[str, Any] | None:
        """Parse a puzzle from CSV row.

        Expected CSV columns: PuzzleId, FEN, Moves, Rating, RatingDeviation,
        Popularity, NbPlays, Themes, GameUrl, OpeningTags
        """
        try:
            fen = row.get("FEN", "")
            rating = int(row.get("Rating", 1500))
            themes = row.get("Themes", "").split()
            primary_theme = themes[0] if themes else "tactics"

            # Determine phase from FEN
            board = chess.Board(fen)
            if not validate_position(board):
                return None

            phase = determine_phase(board, 20)  # Assume middlegame depth

            return {
                "fen": fen,
                "pgn_moves": "",  # CSV doesn't have game history
                "theme": primary_theme,
                "difficulty": rating_to_difficulty(rating),
                "phase": phase,
                "source": "lichess_puzzles",
                "rating": rating,
            }
        except (KeyError, ValueError):
            return None


class PGNPositionSampler:
    """Sample positions from PGN game files."""

    def __init__(self, pgn_path: str):
        """Initialize PGN sampler.

        Args:
            pgn_path: Path to PGN file
        """
        self.pgn_path = pgn_path

    def sample_positions(
        self,
        count_per_phase: int = 500,
        seed: int = 42,
    ) -> dict[str, list[dict[str, Any]]]:
        """Sample positions from games in the PGN file.

        Args:
            count_per_phase: Target positions per phase
            seed: Random seed

        Returns:
            Dictionary mapping phase to list of positions
        """
        if not Path(self.pgn_path).exists():
            logger.error(f"PGN file not found: {self.pgn_path}")
            return {}

        rng = random.Random(seed)
        positions_by_phase: dict[str, list[dict[str, Any]]] = {
            "opening": [],
            "middlegame": [],
            "endgame": [],
        }

        logger.info(f"Sampling positions from {self.pgn_path}...")

        with open(self.pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                positions = self._extract_positions_from_game(game, rng)
                for pos in positions:
                    phase = pos["phase"]
                    if len(positions_by_phase[phase]) < count_per_phase * 2:
                        positions_by_phase[phase].append(pos)

                # Early exit if we have enough
                if all(
                    len(p) >= count_per_phase * 2
                    for p in positions_by_phase.values()
                ):
                    break

        # Final sampling to exact counts
        for phase in positions_by_phase:
            if len(positions_by_phase[phase]) > count_per_phase:
                positions_by_phase[phase] = rng.sample(
                    positions_by_phase[phase], count_per_phase
                )

        total = sum(len(p) for p in positions_by_phase.values())
        logger.info(f"Sampled {total} positions from PGN")

        return positions_by_phase

    def _extract_positions_from_game(
        self,
        game: chess.pgn.Game,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        """Extract interesting positions from a single game."""
        positions = []
        board = game.board()
        move_history = []
        move_count = 0

        for move in game.mainline_moves():
            san = board.san(move)
            move_history.append(san)
            board.push(move)
            move_count += 1

            # Sample some positions (not every move)
            if rng.random() > 0.1:  # 10% chance to sample each position
                continue

            if not validate_position(board):
                continue

            phase = determine_phase(board, move_count)

            positions.append({
                "fen": board.fen(),
                "pgn_moves": self._moves_to_pgn(move_history),
                "theme": "game_position",
                "difficulty": "medium",  # Will be adjusted by eval
                "phase": phase,
                "source": "real_game",
            })

        return positions

    def _moves_to_pgn(self, moves: list[str]) -> str:
        """Convert move list to PGN format."""
        pgn_parts = []
        for i, move in enumerate(moves):
            if i % 2 == 0:
                pgn_parts.append(f"{i // 2 + 1}. {move}")
            else:
                pgn_parts.append(move)
        return " ".join(pgn_parts)


def build_dataset(
    config: dict[str, Any],
    output_dir: str = "data",
) -> dict[str, list[dict[str, Any]]]:
    """Build the complete dataset from all sources.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save dataset files

    Returns:
        Dictionary mapping difficulty tier to positions
    """
    seed = config.get("benchmark", {}).get("random_seed", 42)
    max_per_tier = config.get("benchmark", {}).get("max_positions_per_tier", 500)
    dataset_config = config.get("dataset", {})

    all_positions: dict[str, list[dict[str, Any]]] = {
        "easy": [],
        "medium": [],
        "hard": [],
        "extreme": [],
    }

    # 1. Lichess puzzles
    lichess_source = dataset_config.get("lichess_source", "local")
    csv_path = dataset_config.get("lichess_csv_path")

    fetcher = LichessPuzzleFetcher(source=lichess_source, csv_path=csv_path)

    if lichess_source == "local" and csv_path:
        puzzles_by_tier = fetcher.fetch_from_csv(
            count_per_tier=max_per_tier // 3,
            seed=seed,
        )
        for tier, puzzles in puzzles_by_tier.items():
            all_positions[tier].extend(puzzles)
    elif lichess_source == "api":
        puzzles = fetcher.fetch_from_api(count=max_per_tier)
        for puzzle in puzzles:
            tier = puzzle["difficulty"]
            all_positions[tier].append(puzzle)

    # 2. Real game positions
    pgn_path = dataset_config.get("pgn_path")
    if pgn_path and Path(pgn_path).exists():
        sampler = PGNPositionSampler(pgn_path)
        positions_by_phase = sampler.sample_positions(
            count_per_phase=max_per_tier // 3,
            seed=seed,
        )
        # Distribute across difficulty tiers (will be adjusted by Stockfish eval later)
        for phase, positions in positions_by_phase.items():
            for pos in positions:
                # Temporary difficulty assignment
                tier = "medium"
                all_positions[tier].append(pos)

    # 3. Generated positions
    generated = generate_positions(
        count=max_per_tier,
        seed=seed,
    )
    for pos in generated:
        # Assign to medium tier by default (will be adjusted by Stockfish eval)
        tier = "medium"
        all_positions[tier].append(pos)

    # Assign unique IDs and ensure we don't exceed max_per_tier
    rng = random.Random(seed)
    position_id = 0
    final_dataset: dict[str, list[dict[str, Any]]] = {}

    # max_per_tier <= 0 means unlimited
    unlimited = max_per_tier <= 0

    for tier in all_positions:
        positions = all_positions[tier]
        if not unlimited and len(positions) > max_per_tier:
            positions = rng.sample(positions, max_per_tier)

        for pos in positions:
            pos["id"] = position_id
            position_id += 1

        final_dataset[tier] = positions

    # Save to files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for tier, positions in final_dataset.items():
        file_path = output_path / f"{tier}.json"
        with open(file_path, "w") as f:
            json.dump(positions, f, indent=2)
        logger.info(f"Saved {len(positions)} positions to {file_path}")

    total = sum(len(p) for p in final_dataset.values())
    logger.info(f"Built dataset with {total} total positions")

    return final_dataset
