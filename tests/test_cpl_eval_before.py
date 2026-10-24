"""Clamped CPL uses the stored 800-node "before" eval (specs/cpl-eval-before.md, T2-6)."""

import json
import math

import pytest

from src.metrics import EVAL_CLAMP_CP, load_results_df

WHITE_TO_MOVE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_CP = 10000


def _record(cpl, before=None, no_search=40, job="j1"):
    record = {
        "job_id": job, "job_type": "standard", "position_id": 1, "fen": WHITE_TO_MOVE,
        "model": "qwen2.5:7b", "prompt_format": "move_only",
        "t1_model_eval": None, "t1_stockfish_eval": no_search, "t1_absolute_error": None,
        "t2_move": "e4", "t2_legal": True, "t2_cpl": cpl, "t3_score": None,
    }
    if before is not None:
        record["t2_eval_before"] = before
        record["t2_eval_nodes"] = 800
    return record


def _load(tmp_path, records):
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return load_results_df(str(path), data_dir=str(tmp_path / "no-data"))


def test_mate_before_and_missed_tactic_stays_within_cap(tmp_path):
    """Missing a mate that leaves the side +100 loses 900 clamped cp, not over 1000."""
    row = _load(tmp_path, [_record(cpl=MATE_CP - 100, before=MATE_CP)]).iloc[0]
    assert row["t2_cpl_clamped"] == pytest.approx(EVAL_CLAMP_CP - 100)
    assert row["t2_eval_before_source"] == "lc0_search"


def test_without_stored_before_uses_old_formula(tmp_path):
    row = _load(tmp_path, [_record(cpl=MATE_CP - 100)]).iloc[0]
    # after = 40 − 9900 → clamped −1000, so CPL = 40 − (−1000)
    assert row["t2_cpl_clamped"] == pytest.approx(40 + EVAL_CLAMP_CP)
    assert row["t2_eval_before_source"] == "no_search"


def test_zero_cpl_with_stored_before(tmp_path):
    row = _load(tmp_path, [_record(cpl=0, before=350)]).iloc[0]
    assert row["t2_cpl_clamped"] == 0


def test_row_without_cpl_has_no_source(tmp_path):
    row = _load(tmp_path, [_record(cpl=None, before=350)]).iloc[0]
    assert math.isnan(row["t2_cpl_clamped"])
    assert row["t2_eval_before_source"] is None or math.isnan(row["t2_eval_before_source"])


def test_mixed_rows_pick_source_per_row(tmp_path):
    df = _load(tmp_path, [_record(cpl=50, before=200, job="a"), _record(cpl=50, job="b")])
    assert list(df["t2_eval_before_source"]) == ["lc0_search", "no_search"]
    assert list(df["t2_cpl_clamped"]) == [pytest.approx(50), pytest.approx(50)]
