"""Build chess position datasets from multiple sources."""

import hashlib
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
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
PGN_PHASE_COUNT = 3


def _fen_to_id(fen: str) -> int:
    """Compute a stable integer ID from a FEN string.

    Uses the first 8 hex chars of SHA-256 (32 bits). For ≤4000 positions the
    birthday-collision probability is <0.0002%, which is acceptable.  IDs are
    stable across dataset rebuilds with different max_positions_per_tier values,
    preventing job-deduplication failures in the job queue.
    """
    return int(hashlib.sha256(fen.encode()).hexdigest()[:8], 16)


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
        except (KeyError, IndexError, AttributeError):
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

        puzzles_by_tier: dict[str, list[dict[str, Any]]] = {
            tier: [] for tier in DIFFICULTY_TIERS
        }

        print(f"  Loading: {self.csv_path}")

        # Use pandas for fast CSV loading - only columns we need
        print("  Reading CSV with pandas...", end=" ", flush=True)
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
        print("\n  Puzzles by difficulty tier:")
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
            # Quotas must follow the seeded input order, not process completion
            # order. All tasks still run in parallel while results are collected.
            futures = [executor.submit(_validate_puzzle_row, row) for row in rows_to_validate]
            pbar = tqdm(
                futures,
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
                    if phase not in positions_by_phase:
                        continue
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


def _puzzle_positions(dataset_config: dict, max_per_tier: int, seed: int) -> dict[str, list[dict]]:
    """Load only the explicitly configured puzzle source at its full quota."""
    source = dataset_config.get("lichess_source")
    if source in (None, "none"):
        return {}
    if source not in ("local", "api"):
        raise ValueError(f"Unknown lichess_source: {source!r}")
    csv_path = dataset_config.get("lichess_csv_path")
    if source == "local" and not csv_path:
        raise ValueError("Local Lichess source requires dataset.lichess_csv_path")
    if source == "api" and max_per_tier <= 0:
        raise ValueError("API source requires a positive max_positions_per_tier; unlimited network fetching is unsupported")
    fetcher = LichessPuzzleFetcher(source=source, csv_path=csv_path)
    if source == "local":
        return fetcher.fetch_from_csv(count_per_tier=max_per_tier, seed=seed)
    positions = {tier: [] for tier in DIFFICULTY_TIERS}
    for puzzle in fetcher.fetch_from_api(count=max_per_tier * len(DIFFICULTY_TIERS)):
        positions[puzzle["difficulty"]].append(puzzle)
    return positions


def _append_optional_sources(all_positions: dict, dataset_config: dict, max_per_tier: int, seed: int) -> None:
    """Mixed sources are opt-in when a Lichess source is configured."""
    puzzle_source = dataset_config.get("lichess_source") in ("local", "api")
    pgn_path = dataset_config.get("pgn_path")
    if dataset_config.get("include_pgn", not puzzle_source) and pgn_path and Path(pgn_path).exists():
        sampler = PGNPositionSampler(pgn_path)
        positions_by_phase = sampler.sample_positions(count_per_phase=max_per_tier // PGN_PHASE_COUNT, seed=seed)
        for positions in positions_by_phase.values():
            all_positions["medium"].extend(positions)
    if dataset_config.get("include_generated", not puzzle_source):
        all_positions["medium"].extend(generate_positions(count=max_per_tier, seed=seed))


def build_dataset(
    config: dict[str, Any],
    output_dir: str = "data",
) -> dict[str, list[dict[str, Any]]]:
    """Build a puzzle-only dataset, or explicitly opt into additional sources.

    Configured local/API puzzle sources do not mix in games or synthetic boards
    unless dataset.include_pgn / dataset.include_generated is true. With no
    puzzle source, legacy generated-only (and configured PGN) builds remain
    available. API fetching requires a positive finite quota; CSV quota <= 0
    retains all valid rows. Optional non-puzzle positions stay in medium; no
    engine-based difficulty reassignment is performed.

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

    for tier, puzzles in _puzzle_positions(dataset_config, max_per_tier, seed).items():
        all_positions[tier].extend(puzzles)
    _append_optional_sources(all_positions, dataset_config, max_per_tier, seed)

    # Deduplicate positions by FEN across all tiers before ID assignment.
    # Positions from different sources (Lichess, PGN, generated) can share the
    # same FEN; keeping duplicates would inflate statistics and produce multiple
    # jobs for the same board position.
    seen_fens: set[str] = set()
    deduped: dict[str, list[dict[str, Any]]] = {tier: [] for tier in all_positions}
    for tier, positions in all_positions.items():
        for pos in positions:
            fen = pos["fen"]
            if fen not in seen_fens:
                seen_fens.add(fen)
                deduped[tier].append(pos)
    all_positions = deduped

    # Sample and assign stable FEN-hash IDs.
    # Sequential counters shift when max_positions_per_tier changes, breaking
    # job-queue deduplication on rebuild.  Hash-based IDs are invariant.
    rng = random.Random(seed)
    final_dataset: dict[str, list[dict[str, Any]]] = {}
    unlimited = max_per_tier <= 0

    assigned_ids: set[int] = set()
    for tier in all_positions:
        positions = all_positions[tier]
        if not unlimited and len(positions) > max_per_tier:
            positions = rng.sample(positions, max_per_tier)

        for pos in positions:
            pos_id = _fen_to_id(pos["fen"])
            # Collision guard: rehash with a counter suffix rather than
            # incrementing, so the resolved ID is still deterministic across
            # rebuilds with different dataset sizes.
            counter = 0
            while pos_id in assigned_ids:
                counter += 1
                pos_id = _fen_to_id(f"{pos['fen']}#{counter}")
            assigned_ids.add(pos_id)
            pos["id"] = pos_id

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
