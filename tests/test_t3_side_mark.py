"""Chance-corrected T3 side mark (specs/t3-side-mark-chance.md, T3-5)."""

import json

import pandas as pd
import pytest

from src.metrics import compute_side_mark_agreement, load_results_df, save_metrics

TRUTH = [300, 200, 120, 60, 10, -10, -80, -400]  # W W W W E E B B
LABEL = {"W": "White", "B": "Black", "E": "Equal"}
TRUTH_SIDES = ["White"] * 4 + ["Equal"] * 2 + ["Black"] * 2


def _df(claims, truth=TRUTH, model="m"):
    return pd.DataFrame({"model": model, "t3_side_claimed": claims, "t1_stockfish_eval": truth})


def _only(table):
    assert len(table) == 1
    return table.iloc[0]


def test_perfect_claimer():
    row = _only(compute_side_mark_agreement(_df(TRUTH_SIDES)))
    assert row["accuracy"] == 1
    assert row["cohen_kappa"] == pytest.approx(1)
    assert row["balanced_accuracy"] == pytest.approx(1)


def test_always_white_scores_majority_share_and_zero_kappa():
    row = _only(compute_side_mark_agreement(_df(["White"] * len(TRUTH))))
    assert row["accuracy"] == pytest.approx(0.5)
    assert row["majority_label"] == "W"
    assert row["majority_share"] == pytest.approx(0.5)
    assert row["cohen_kappa"] == pytest.approx(0)


def test_truth_threshold_is_50():
    row = _only(compute_side_mark_agreement(_df(["Equal", "White"], truth=[50, 51])))
    assert row["accuracy"] == 1


def test_invalid_claims_excluded():
    row = _only(compute_side_mark_agreement(_df(["White", None, "Maybe", "Black"], truth=[300, 300, 300, -300])))
    assert row["n"] == 2


def test_equal_shares():
    claims = ["White"] * 7 + ["Equal"]
    row = _only(compute_side_mark_agreement(_df(claims)))
    assert row["equal_claim_share"] == pytest.approx(1 / 8)
    assert row["equal_truth_share"] == pytest.approx(2 / 8)


def _record(i, claim, truth):
    return {
        "job_id": f"j{i}", "job_type": "standard", "position_id": i,
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "model": "qwen2.5:7b", "prompt_format": "explanation_only",
        "difficulty": "easy", "phase": "opening", "source": "lichess_puzzles", "inference_ms": 1,
        "t1_model_eval": None, "t1_stockfish_eval": truth, "t1_absolute_error": None, "t1_direction_correct": None,
        "t2_move": None, "t2_legal": False, "t2_cpl": None,
        "t3_side_claimed": claim, "t3_score": None, "t3_p1_side_correct": None, "t3_p2_theme_correct": None,
    }


def _save(tmp_path, records):
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "metrics"
    save_metrics(load_results_df(str(path), data_dir=str(tmp_path / "no-data")), output_dir=str(out))
    return out


def test_save_metrics_writes_csv_and_summary(tmp_path):
    out = _save(tmp_path, [_record(i, c, t) for i, (c, t) in enumerate(zip(TRUTH_SIDES, TRUTH))])
    assert (out / "t3_side_mark_agreement.csv").exists()
    pooled = json.loads((out / "summary.json").read_text())["overall_metrics"]["t3_side_mark"]
    assert pooled["n"] == len(TRUTH)
    assert pooled["accuracy"] == 1


def test_summary_none_without_claims(tmp_path):
    out = _save(tmp_path, [_record(0, None, 100)])
    assert json.loads((out / "summary.json").read_text())["overall_metrics"]["t3_side_mark"] is None
