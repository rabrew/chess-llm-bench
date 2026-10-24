"""Base vs fine-tuned comparison (specs/t1-zero-baseline-and-ft-comparison.md)."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

FT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FT_ROOT.parent
sys.path.insert(0, str(FT_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import compare_finetuning_results as cmp  # noqa: E402
from src.metrics import compute_eval_rank_correlation  # noqa: E402

BASE, FT = "llama3.2:3b", "llama3.2:3b-board-ft"
FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
TRUTH_EVALS = [-450, -120, -60, -20, 0, 40, 90, 130, 300, 600]


def _record(model, pos, fmt, model_eval=None, truth=0, move=None, legal=False,
            cpl=None, t3=None, job_type="standard"):
    return {
        "job_id": f"{model}_{pos}_{fmt}_{job_type}",
        "job_type": job_type,
        "position_id": pos,
        "fen": FEN,
        "model": model,
        "prompt_format": fmt,
        "difficulty": "easy",
        "t1_model_eval": model_eval,
        "t1_stockfish_eval": truth,
        "t1_absolute_error": None if model_eval is None else abs(model_eval - truth),
        "t1_direction_correct": None,
        "t2_move": move,
        "t2_legal": legal,
        "t2_cpl": cpl,
        "t3_score": t3,
    }


def _write(path: Path, records: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(path)


def _eval_records(model, evals, fmt="eval_only"):
    return [_record(model, i, fmt, model_eval=me, truth=sf)
            for i, (me, sf) in enumerate(zip(evals, TRUTH_EVALS))]


def _compare(tmp_path, base_records, ft_records) -> pd.DataFrame:
    base = _write(tmp_path / "base.jsonl", base_records)
    ft = _write(tmp_path / "ft.jsonl", ft_records)
    return cmp.compare(cmp.load_records(base), cmp.load_records(ft))


def _only_row(table: pd.DataFrame) -> pd.Series:
    assert len(table) == 1
    return table.iloc[0]


# ---------- pairing ----------

def test_only_keys_on_both_sides_are_compared(tmp_path):
    base = _eval_records(BASE, TRUTH_EVALS)
    ft = _eval_records(FT, TRUTH_EVALS)[:6]
    row = _only_row(_compare(tmp_path, base, ft))
    assert row["base_model"] == BASE
    assert row["ft_model"] == FT
    assert row["n_pairs"] == 6


def test_correction_and_control_rows_are_ignored(tmp_path):
    base = _eval_records(BASE, TRUTH_EVALS) + [
        _record(BASE, 0, "eval_only", model_eval=5000, truth=-450, job_type="correction"),
        _record(BASE, 99, "eval_only", model_eval=0, truth=0, job_type="control"),
    ]
    ft = _eval_records(FT, TRUTH_EVALS)
    row = _only_row(_compare(tmp_path, base, ft))
    assert row["n_pairs"] == len(TRUTH_EVALS)
    assert row["base_t1_mae"] == pytest.approx(0)


def test_duplicate_keys_keep_first_record(tmp_path):
    base = _eval_records(BASE, TRUTH_EVALS)
    duplicate = _record(BASE, 0, "eval_only", model_eval=1500, truth=-450)
    ft = _eval_records(FT, TRUTH_EVALS)
    row = _only_row(_compare(tmp_path, base + [duplicate], ft))
    assert row["n_pairs"] == len(TRUTH_EVALS)
    assert row["base_t1_mae"] == pytest.approx(0)


def test_ft_model_without_base_is_skipped(tmp_path):
    base = _eval_records(BASE, TRUTH_EVALS)
    ft = _eval_records(FT, TRUTH_EVALS) + _eval_records("qwen2.5:7b-board-ft", TRUTH_EVALS)
    assert list(_compare(tmp_path, base, ft)["ft_model"]) == [FT]


# ---------- T1 ----------

def test_shrinking_toward_zero_lowers_mae_but_not_rank_correlation(tmp_path):
    """Regression for the 2026-10-22 artefact: small answers beat on error without skill."""
    noise = [300, -200, 150, -400, 250, -100, 350, -300, 200, -250]
    shrunk = [round(n / 100) for n in noise]
    row = _only_row(_compare(tmp_path, _eval_records(BASE, noise), _eval_records(FT, shrunk)))
    assert row["delta_t1_mae"] < 0
    assert row["ft_t1_mae_ratio"] == pytest.approx(row["ft_t1_mae"] / row["zero_mae"])
    assert abs(row["base_t1_spearman"]) < 0.3
    assert abs(row["ft_t1_spearman"]) < 0.3


def test_spearman_matches_metrics_module(tmp_path):
    evals = [-300, -100, -90, 10, 5, 60, 40, 200, 150, 900]
    base = _eval_records(BASE, TRUTH_EVALS)
    ft = _eval_records(FT, evals)
    row = _only_row(_compare(tmp_path, base, ft))
    expected = compute_eval_rank_correlation(pd.DataFrame(ft))["spearman_rho"].iloc[0]
    assert row["ft_t1_spearman"] == pytest.approx(expected)
    assert row["base_t1_spearman"] == pytest.approx(1.0)
    assert row["base_t1_kappa_t50"] == pytest.approx(1.0)


def test_mate_truth_excluded_from_mae(tmp_path):
    mate = cmp.MATE_SCORE_THRESHOLD_CP + 500
    base = _eval_records(BASE, TRUTH_EVALS) + [_record(BASE, 50, "eval_only", model_eval=0, truth=mate)]
    ft = _eval_records(FT, TRUTH_EVALS) + [_record(FT, 50, "eval_only", model_eval=0, truth=mate)]
    row = _only_row(_compare(tmp_path, base, ft))
    assert row["base_t1_mae"] == pytest.approx(0)


# ---------- T2 / T3 ----------

def test_legal_rate_ignores_prompts_that_ask_for_no_move(tmp_path):
    base = [
        _record(BASE, 0, "move_only", move="Nf6", legal=True),
        _record(BASE, 1, "fen_only", move="Nf6", legal=True),
        _record(BASE, 2, "eval_only", model_eval=0, legal=False),
        _record(BASE, 3, "explanation_only", t3=1, legal=False),
    ]
    ft = [
        _record(FT, 0, "move_only", move=None, legal=False),
        _record(FT, 1, "fen_only", move="Nf6", legal=True),
        _record(FT, 2, "eval_only", model_eval=0, legal=False),
        _record(FT, 3, "explanation_only", t3=1, legal=False),
    ]
    row = _only_row(_compare(tmp_path, base, ft))
    assert row["base_t2_legal_rate"] == pytest.approx(1.0)
    assert row["ft_t2_legal_rate"] == pytest.approx(0.5)
    assert row["delta_t2_legal_rate"] == pytest.approx(-0.5)


def test_first_attempt_rate_nan_without_raw_replies(tmp_path):
    """Base re-run saves replies; the fine-tuned run did not (specs/t2-move-source.md)."""
    base = [
        {**_record(BASE, 0, "fen_only", move="Nf6", legal=True),
         "raw_response": "Eval: 0\nMove: Nf6\nExplanation: Equal — level."},
        {**_record(BASE, 1, "fen_only", move="Nf6", legal=True),
         "raw_response": "Eval: 0\nMove: e4\nExplanation: Equal — level.", "raw_move_retry_response": "Nf6"},
    ]
    ft = [_record(FT, 0, "fen_only", move="Nf6", legal=True), _record(FT, 1, "fen_only", move="Nf6", legal=True)]
    row = _only_row(_compare(tmp_path, base, ft))
    assert row["base_t2_first_attempt_legal_rate"] == pytest.approx(0.5)
    assert math.isnan(row["ft_t2_first_attempt_legal_rate"])
    assert math.isnan(row["delta_t2_first_attempt_legal_rate"])


def test_t3_mean_and_cpl_clamp(tmp_path):
    base = [_record(BASE, 0, "fen_only", move="Nf6", legal=True, cpl=5000, t3=2)]
    ft = [_record(FT, 0, "fen_only", move="Nf6", legal=True, cpl=100, t3=0)]
    row = _only_row(_compare(tmp_path, base, ft))
    assert row["base_t2_mean_cpl_clamped"] <= cmp.CPL_CLAMP_CP
    assert row["delta_t3_mean"] == pytest.approx(-2)


def test_t3_uses_whole_word_score(tmp_path):
    """'opinion' must not count as naming the 'pin' theme (F-3, T3-1)."""
    def explained(model, text):
        record = _record(model, 0, "explanation_only", t3=2)
        return {**record, "theme": "pin", "t3_explanation": text, "t3_p1_side_correct": 1}
    base = [explained(BASE, "White — in my opinion the bishop is strong.")]
    ft = [explained(FT, "White — the pin on the knight wins material.")]
    row = _only_row(_compare(tmp_path, base, ft))
    assert row["base_t3_mean"] == pytest.approx(1)
    assert row["ft_t3_mean"] == pytest.approx(2)


def test_undefined_metric_gives_nan_delta_not_zero(tmp_path):
    base = [_record(BASE, 0, "explanation_only", t3=1)]
    ft = [_record(FT, 0, "explanation_only", t3=1)]
    row = _only_row(_compare(tmp_path, base, ft))
    for metric in ("t1_spearman", "t1_mae", "t2_legal_rate"):
        assert math.isnan(row[f"delta_{metric}"]), metric
    assert row["delta_t3_mean"] == 0


# ---------- paths + CLI ----------

def _config(path: Path, results_file: str) -> str:
    path.write_text(yaml.safe_dump({"paths": {"results_file": results_file}}))
    return str(path)


def test_results_path_read_from_config(tmp_path):
    cfg = _config(tmp_path / "c.yaml", "results/x.jsonl")
    assert cmp.resolve_results_path(None, cfg) == "results/x.jsonl"


def test_explicit_path_overrides_config(tmp_path):
    cfg = _config(tmp_path / "c.yaml", "results/x.jsonl")
    assert cmp.resolve_results_path("other.jsonl", cfg) == "other.jsonl"


def test_config_without_results_file_raises(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"paths": {}}))
    with pytest.raises(ValueError):
        cmp.resolve_results_path(None, str(cfg))


def test_main_writes_csv(tmp_path, monkeypatch):
    base = _write(tmp_path / "base.jsonl", _eval_records(BASE, TRUTH_EVALS))
    ft = _write(tmp_path / "ft.jsonl", _eval_records(FT, TRUTH_EVALS))
    out = tmp_path / "out" / "comparison.csv"
    monkeypatch.setattr(sys, "argv", ["cmp", "--base", base, "--ft", ft, "--output", str(out)])
    cmp.main()
    table = pd.read_csv(out)
    assert list(table["ft_model"]) == [FT]
    assert {"base_t1_spearman", "ft_t1_mae_ratio", "zero_mae", "delta_t2_legal_rate"} <= set(table.columns)


def test_main_exits_on_missing_file(tmp_path, monkeypatch):
    ft = _write(tmp_path / "ft.jsonl", _eval_records(FT, TRUTH_EVALS))
    monkeypatch.setattr(sys, "argv", ["cmp", "--base", str(tmp_path / "nope.jsonl"), "--ft", ft])
    with pytest.raises(SystemExit) as exc:
        cmp.main()
    assert exc.value.code == 1


def test_main_exits_when_nothing_pairs(tmp_path, monkeypatch):
    base = _write(tmp_path / "base.jsonl", _eval_records("mistral:7b", TRUTH_EVALS))
    ft = _write(tmp_path / "ft.jsonl", _eval_records(FT, TRUTH_EVALS))
    monkeypatch.setattr(sys, "argv", ["cmp", "--base", base, "--ft", ft,
                                      "--output", str(tmp_path / "c.csv")])
    with pytest.raises(SystemExit) as exc:
        cmp.main()
    assert exc.value.code == 1
