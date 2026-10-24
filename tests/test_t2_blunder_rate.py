"""Blunder rate with bootstrap CIs and CPL buckets (specs/t2-blunder-rate.md, T2-1)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    CPL_BUCKETS,
    compute_blunder_rates,
    compute_cpl_buckets,
    load_results_df,
    save_metrics,
)

BUCKET_KEYS = [key for key, _, _ in CPL_BUCKETS]


def _df(cpls_by_model: dict[str, list[float | None]]) -> pd.DataFrame:
    rows = [{"model": model, "t2_cpl_clamped": cpl}
            for model, cpls in cpls_by_model.items() for cpl in cpls]
    return pd.DataFrame(rows)


def _bucket_of(cpl: float) -> str:
    row = compute_cpl_buckets(_df({"m": [cpl]})).iloc[0]
    return next(key for key in BUCKET_KEYS if row[f"{key}_count"] == 1)


# ---------- buckets ----------

@pytest.mark.parametrize("cpl, bucket", [
    (0, "perfect"), (24.9, "perfect"), (25, "strong"), (99.9, "strong"), (100, "decent"),
    (500, "very_weak"), (999.9, "very_weak"), (1000, "blunder"), (1900, "blunder"),
])
def test_bucket_edges(cpl, bucket):
    assert _bucket_of(cpl) == bucket


def test_bucket_counts_sum_to_n_and_shares_to_one():
    row = compute_cpl_buckets(_df({"m": [0, 30, 150, 700, 1000, 1200, None]})).iloc[0]
    assert row["n"] == 6
    assert sum(row[f"{key}_count"] for key in BUCKET_KEYS) == 6
    assert sum(row[f"{key}_share"] for key in BUCKET_KEYS) == pytest.approx(1)


def test_blunder_rate_matches_bucket_counts():
    df = _df({"m": [0, 30, 150, 700, 1000, 1200, 1500, 10]})
    rate = compute_blunder_rates(df).iloc[0]["blunder_rate"]
    share = compute_cpl_buckets(df).iloc[0]["blunder_share"]
    assert rate == pytest.approx(share) == pytest.approx(3 / 8)


# ---------- confidence intervals ----------

def test_ci_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    cpls = list(np.where(rng.random(2000) < 0.4, 1000, 0))
    first = compute_blunder_rates(_df({"m": cpls})).iloc[0]
    second = compute_blunder_rates(_df({"m": cpls})).iloc[0]
    assert 0 <= first["ci_low"] <= first["blunder_rate"] <= first["ci_high"] <= 1
    assert first["ci_high"] - first["ci_low"] < 0.06
    assert (first["ci_low"], first["ci_high"]) == (second["ci_low"], second["ci_high"])


def test_ci_does_not_depend_on_other_models():
    cpls = [0, 1000] * 100
    alone = compute_blunder_rates(_df({"a": cpls})).iloc[0]
    table = compute_blunder_rates(_df({"a": cpls, "b": [1000] * 50}))
    together = table[table["model"] == "a"].iloc[0]
    assert (alone["ci_low"], alone["ci_high"]) == (together["ci_low"], together["ci_high"])


def test_extreme_rate_gives_zero_width_interval():
    row = compute_blunder_rates(_df({"m": [0] * 30})).iloc[0]
    assert row["blunder_rate"] == row["ci_low"] == row["ci_high"] == 0


# ---------- ranking ----------

def test_sorted_best_first_with_mean_and_median():
    table = compute_blunder_rates(_df({"bad": [1000] * 90 + [0] * 10, "good": [0] * 90 + [1000] * 10}))
    assert list(table["model"]) == ["good", "bad"]
    good = table.iloc[0]
    assert good["mean_cpl_clamped"] == pytest.approx(100)
    assert good["median_cpl_clamped"] == 0


def test_far_apart_models_not_tied_identical_models_tied():
    far = compute_blunder_rates(_df({"a": [0] * 500, "b": [1000] * 500}))
    assert not far["rank_tied"].any()
    same = compute_blunder_rates(_df({"a": [0, 1000] * 200, "b": [0, 1000] * 200}))
    assert same["rank_tied"].all()


def test_single_model_not_tied():
    assert not compute_blunder_rates(_df({"m": [0, 1000]})).iloc[0]["rank_tied"]


def test_model_without_cpl_rows_omitted():
    df = _df({"has": [0, 1000], "none": [None, None]})
    assert list(compute_blunder_rates(df)["model"]) == ["has"]
    assert list(compute_cpl_buckets(df)["model"]) == ["has"]


# ---------- save_metrics ----------

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _record(job, cpl):
    return {
        "job_id": job, "job_type": "standard", "position_id": 1, "fen": FEN,
        "model": "qwen2.5:7b", "prompt_format": "move_only",
        "difficulty": "easy", "phase": "opening", "source": "lichess_puzzles",
        "inference_ms": 100, "t1_model_eval": None, "t1_stockfish_eval": 0,
        "t1_absolute_error": None, "t1_direction_correct": None,
        "t2_move": "e4", "t2_legal": cpl is not None, "t2_cpl": cpl,
        "t2_eval_before": 0, "t2_eval_nodes": 800,
        "t3_score": None, "t3_p1_side_correct": None, "t3_p2_theme_correct": None,
    }


def _save(tmp_path, records):
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "metrics"
    save_metrics(load_results_df(str(path), data_dir=str(tmp_path / "no-data")), output_dir=str(out))
    return out


def test_save_metrics_writes_tables_and_summary(tmp_path):
    out = _save(tmp_path, [_record("a", 0), _record("b", 1500), _record("c", 40)])
    assert (out / "cpl_buckets.csv").exists()
    assert (out / "t2_blunder_rate.csv").exists()
    pooled = json.loads((out / "summary.json").read_text())["overall_metrics"]["t2_blunder_rate"]
    assert pooled["n"] == 3
    assert pooled["rate"] == pytest.approx(1 / 3)
    assert pooled["ci_low"] <= pooled["rate"] <= pooled["ci_high"]


def test_save_metrics_without_cpl_rows(tmp_path):
    out = _save(tmp_path, [_record("a", None)])
    assert json.loads((out / "summary.json").read_text())["overall_metrics"]["t2_blunder_rate"] is None
    assert not (out / "cpl_buckets.csv").exists()
    assert not (out / "t2_blunder_rate.csv").exists()
