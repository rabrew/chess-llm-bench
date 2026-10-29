"""T2-5: does the model ranking by blunder rate survive a stronger Lc0 search?

Re-scores a seeded sample of the base models' legal moves at higher node counts and compares the
per-model blunder rates with the stored 800-node ones (specs/engine-strength-check.md).

    python scripts/engine_strength_check.py
    python scripts/engine_strength_check.py --nodes 800 8000 --per-tier 250
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine_wrapper import Lc0Engine  # noqa: E402
from src.metrics import BLUNDER_CPL_CP, compute_clamped_cpl, load_results_df  # noqa: E402
from src.utils import load_config, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

SEED = 42
PER_TIER = 250
DEFAULT_NODES = (800, 8000)
STORED = "stored"
RHO_THRESHOLD = 0.9
CACHE_DIR = Path("results/engine_strength")
METRICS_DIR = Path("results/metrics")


# ---------- sampling and work ----------

def sample_positions(tiers: dict[int, str], per_tier: int, seed: int) -> list[int]:
    """per_tier position ids from each tier, drawn from sorted ids with one seeded generator."""
    rng = random.Random(seed)
    sampled = []
    for tier in sorted(set(tiers.values())):
        ids = sorted(pid for pid, t in tiers.items() if t == tier)
        sampled += rng.sample(ids, min(per_tier, len(ids)))
    return sorted(sampled)


def fen_after(fen: str, move_san: str) -> str | None:
    """The FEN after a SAN move, or None when the move doesn't parse in that position."""
    board = chess.Board(fen)
    try:
        board.push_san(move_san)
    except ValueError:
        return None
    return board.fen()


def work_fens(rows: pd.DataFrame) -> set[str]:
    """Every pre-move FEN plus every position after a played move."""
    pairs = set(zip(rows["fen"], rows["t2_move"]))
    after = {fen_after(fen, move) for fen, move in pairs}
    return set(rows["fen"]) | (after - {None})


# ---------- engine cache ----------

def load_cache(path: Path) -> dict[str, dict]:
    """FEN -> {"eval", "best_move"} from a cache file; a malformed line raises, naming the file."""
    if not path.exists():
        return {}
    evals = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            entry = json.loads(line)
            evals[entry["fen"]] = {"eval": entry["eval"], "best_move": entry["best_move"]}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Malformed line {number} in {path}: {exc}") from exc
    return evals


def evaluate_missing(fens: set[str], cache_path: Path, engine: Any) -> tuple[dict[str, dict], int]:
    """Evaluate the FENs not yet cached, appending each to the cache. Returns (evals, failures)."""
    evals = load_cache(cache_path)
    todo = sorted(fens - evals.keys())
    logger.info(f"{cache_path.name}: {len(fens) - len(todo)} cached, {len(todo)} to evaluate")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with cache_path.open("a") as cache:
        for done, fen in enumerate(todo, start=1):
            try:
                result = engine.evaluate(fen)
            except Exception as exc:  # engine crash on one position: counted and reported, never fatal
                logger.warning(f"Lc0 failed on {fen!r}: {exc}")
                failed += 1
                continue
            evals[fen] = {"eval": result["eval"], "best_move": result["best_move"]}
            cache.write(json.dumps({"fen": fen, **evals[fen]}) + "\n")
            if done % 1000 == 0:
                cache.flush()
                logger.info(f"{cache_path.name}: {done}/{len(todo)}")
    return evals, failed


# ---------- scoring ----------

def _row_cpl(fen: str, move: str, evals: dict[str, dict]) -> float:
    pre = evals.get(fen)
    if pre is None:
        return np.nan
    if move == pre["best_move"]:
        return 0.0
    post = evals.get(fen_after(fen, move))
    if post is None:
        return np.nan
    return compute_clamped_cpl(pre["eval"], post["eval"], chess.Board(fen).turn == chess.WHITE)


def score_rows(rows: pd.DataFrame, evals: dict[str, dict]) -> pd.Series:
    """Clamped CPL per row, scored as enrich_cpl.py does; NaN where an evaluation is missing."""
    return pd.Series([_row_cpl(f, m, evals) for f, m in zip(rows["fen"], rows["t2_move"])],
                     index=rows.index, dtype=float)


def blunder_table(rows: pd.DataFrame, cpl_by_strength: dict[str, pd.Series]) -> pd.DataFrame:
    """Per model and strength: n, blunder rate, mean clamped CPL and rank (1 = fewest blunders)."""
    columns = {}
    for strength, cpl in cpl_by_strength.items():
        grouped = cpl.dropna().groupby(rows["model"])
        columns[f"n_{strength}"] = grouped.size()
        columns[f"blunder_rate_{strength}"] = grouped.apply(lambda c: float((c >= BLUNDER_CPL_CP).mean()))
        columns[f"mean_cpl_{strength}"] = grouped.mean()
        columns[f"rank_{strength}"] = columns[f"blunder_rate_{strength}"].rank(method="average")
    return pd.DataFrame(columns).rename_axis("model").reset_index()


def rank_agreement(table: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    """Spearman ρ (Pearson on average ranks, no scipy) and the largest rank shift between strengths."""
    both = table.dropna(subset=[f"blunder_rate_{a}", f"blunder_rate_{b}"])
    rate_a, rate_b = both[f"blunder_rate_{a}"], both[f"blunder_rate_{b}"]
    shift = (both[f"blunder_rate_{a}"].rank() - both[f"blunder_rate_{b}"].rank()).abs()
    return {"rho": float(rate_a.rank().corr(rate_b.rank())), "max_rank_shift": float(shift.max()),
            "models": len(both)}


# ---------- run ----------

def _sampled_rows(results_file: str, per_tier: int) -> pd.DataFrame:
    df = load_results_df(results_file)
    legal = df[df["t2_cpl_clamped"].notna()]
    tiers = dict(zip(legal["position_id"].astype(int), legal["difficulty"]))
    keep = set(sample_positions(tiers, per_tier, SEED))
    return legal[legal["position_id"].astype(int).isin(keep)].reset_index(drop=True)


def run(results_file: str, lc0: dict, nodes: tuple[int, ...], per_tier: int) -> dict:
    rows = _sampled_rows(results_file, per_tier)
    fens = work_fens(rows)
    logger.info(f"{rows['position_id'].nunique()} positions, {len(rows)} moves, {len(fens)} FENs per strength")
    cpl_by_strength = {STORED: rows["t2_cpl_clamped"].astype(float)}
    failures = {}
    for count in nodes:
        with Lc0Engine(path=lc0["binary"], weights=lc0["weights"], nodes=count) as engine:
            evals, failures[str(count)] = evaluate_missing(fens, CACHE_DIR / f"lc0_{count}.jsonl", engine)
        cpl_by_strength[str(count)] = score_rows(rows, evals)
    table = blunder_table(rows, cpl_by_strength)
    table.to_csv(METRICS_DIR / "engine_strength.csv", index=False)
    comparisons = {f"{STORED}_vs_{n}": rank_agreement(table, STORED, str(n)) for n in nodes}
    strongest = f"{STORED}_vs_{max(nodes)}"
    summary = {
        "seed": SEED, "per_tier": per_tier, "positions": int(rows["position_id"].nunique()),
        "moves": len(rows), "nodes": list(nodes), "lc0_binary": lc0["binary"], "lc0_weights": lc0["weights"],
        "engine_failures": failures,
        "overall_blunder_rate": {s: float((c.dropna() >= BLUNDER_CPL_CP).mean()) for s, c in cpl_by_strength.items()},
        "comparisons": comparisons, "criterion": f"{strongest} rho >= {RHO_THRESHOLD}",
        "ranking_holds": comparisons[strongest]["rho"] >= RHO_THRESHOLD,
    }
    (METRICS_DIR / "engine_strength_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="T2-5 engine-strength sensitivity check.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--nodes", type=int, nargs="+", default=list(DEFAULT_NODES))
    parser.add_argument("--per-tier", type=int, default=PER_TIER)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    lc0 = config.get("lc0", {})
    if not lc0.get("binary") or not lc0.get("weights"):
        raise SystemExit(f"lc0.binary and lc0.weights must be set in {args.config}")
    results_file = config.get("paths", {}).get("results_file", "results/evaluations.jsonl")
    summary = run(results_file, lc0, tuple(args.nodes), args.per_tier)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
