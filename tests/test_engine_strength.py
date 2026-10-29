"""Tests for scripts/engine_strength_check.py (T2-5, specs/engine-strength-check.md)."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.engine_strength_check import (  # noqa: E402
    blunder_table,
    evaluate_missing,
    fen_after,
    load_cache,
    rank_agreement,
    sample_positions,
    score_rows,
    work_fens,
)

WHITE_FEN = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
BLACK_FEN = "4k3/4p3/8/8/8/8/8/4K3 b - - 0 1"


def _rows(entries):
    return pd.DataFrame(entries, columns=["model", "fen", "t2_move"])


class _Engine:
    def __init__(self, evals, fail=()):
        self.evals, self.fail, self.calls = evals, set(fail), []

    def evaluate(self, fen):
        self.calls.append(fen)
        if fen in self.fail:
            raise RuntimeError("engine died")
        return self.evals[fen]


# ---------- sampling ----------

def test_sample_positions_takes_per_tier_and_is_seeded():
    tiers = {i: ("easy" if i < 10 else "hard") for i in range(20)}
    first = sample_positions(tiers, per_tier=3, seed=42)
    assert first == sample_positions(tiers, per_tier=3, seed=42)
    assert sum(1 for i in first if i < 10) == 3 and sum(1 for i in first if i >= 10) == 3


def test_sample_positions_small_tier_takes_all():
    assert sample_positions({1: "easy", 2: "easy"}, per_tier=5, seed=42) == [1, 2]


# ---------- work ----------

def test_work_fens_has_pre_and_post_positions():
    rows = _rows([("m", WHITE_FEN, "e4"), ("n", WHITE_FEN, "e4"), ("m", WHITE_FEN, "Kd2")])
    fens = work_fens(rows)
    assert fens == {WHITE_FEN, fen_after(WHITE_FEN, "e4"), fen_after(WHITE_FEN, "Kd2")}


def test_fen_after_unparseable_move_is_none():
    assert fen_after(WHITE_FEN, "Qh5") is None


# ---------- scoring ----------

def test_best_move_scores_zero():
    evals = {WHITE_FEN: {"eval": 300, "best_move": "e4"}}
    assert score_rows(_rows([("m", WHITE_FEN, "e4")]), evals).tolist() == [0.0]


def test_cpl_is_from_the_movers_side_and_clamped():
    after_w, after_b = fen_after(WHITE_FEN, "Kd2"), fen_after(BLACK_FEN, "Kd8")
    evals = {
        WHITE_FEN: {"eval": 5000, "best_move": "e4"}, after_w: {"eval": -200, "best_move": None},
        BLACK_FEN: {"eval": -100, "best_move": "e5"}, after_b: {"eval": 50, "best_move": None},
    }
    cpl = score_rows(_rows([("m", WHITE_FEN, "Kd2"), ("m", BLACK_FEN, "Kd8")]), evals)
    assert cpl.tolist() == [1200.0, 150.0]  # 5000 clamps to 1000; Black loses when White's eval rises


def test_missing_eval_gives_nan():
    cpl = score_rows(_rows([("m", WHITE_FEN, "Kd2")]), {WHITE_FEN: {"eval": 0, "best_move": "e4"}})
    assert math.isnan(cpl.iloc[0])


# ---------- cache ----------

def test_evaluate_missing_skips_cached_and_appends(tmp_path):
    cache = tmp_path / "lc0_800.jsonl"
    cache.write_text(json.dumps({"fen": WHITE_FEN, "eval": 1, "best_move": "e4"}) + "\n")
    engine = _Engine({BLACK_FEN: {"eval": -5, "best_move": "e5", "mate": None}})
    evals, failed = evaluate_missing({WHITE_FEN, BLACK_FEN}, cache, engine)
    assert engine.calls == [BLACK_FEN] and failed == 0
    assert evals[BLACK_FEN] == {"eval": -5, "best_move": "e5"}
    assert load_cache(cache) == evals


def test_engine_failure_is_counted_not_cached(tmp_path):
    cache = tmp_path / "lc0_800.jsonl"
    evals, failed = evaluate_missing({WHITE_FEN}, cache, _Engine({}, fail=[WHITE_FEN]))
    assert failed == 1 and evals == {} and load_cache(cache) == {}


def test_malformed_cache_raises_with_path(tmp_path):
    cache = tmp_path / "lc0_800.jsonl"
    cache.write_text("{not json\n")
    with pytest.raises(ValueError, match="lc0_800.jsonl"):
        load_cache(cache)


# ---------- comparison ----------

def test_blunder_table_ranks_fewest_blunders_first():
    rows = _rows([("a", WHITE_FEN, "x"), ("a", WHITE_FEN, "x"), ("b", WHITE_FEN, "x"), ("b", WHITE_FEN, "x")])
    table = blunder_table(rows, {"stored": pd.Series([0.0, 1000.0, 1000.0, 1000.0], index=rows.index)})
    table = table.set_index("model")
    assert table.loc["a", "blunder_rate_stored"] == 0.5
    assert table.loc["a", "rank_stored"] == 1 and table.loc["b", "rank_stored"] == 2


def test_blunder_table_drops_unscored_rows():
    rows = _rows([("a", WHITE_FEN, "x"), ("a", WHITE_FEN, "x")])
    table = blunder_table(rows, {"8000": pd.Series([float("nan"), 1000.0], index=rows.index)})
    assert table.loc[0, "n_8000"] == 1 and table.loc[0, "blunder_rate_8000"] == 1.0


def test_rank_agreement():
    table = pd.DataFrame({"model": list("abcd"), "blunder_rate_x": [0.1, 0.2, 0.3, 0.4],
                          "blunder_rate_y": [0.1, 0.2, 0.3, 0.4], "blunder_rate_z": [0.4, 0.3, 0.2, 0.1]})
    same = rank_agreement(table, "x", "y")
    assert same["rho"] == pytest.approx(1.0) and same["max_rank_shift"] == 0.0 and same["models"] == 4
    assert rank_agreement(table, "x", "z")["rho"] == pytest.approx(-1.0)
