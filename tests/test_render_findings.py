"""Findings tables generated from the metrics (specs/findings-tables.md)."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import render_findings as rf  # noqa: E402

MODELS = ["qwen2.5:7b", "mistral:7b"]


def _write_metrics(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_jobs": 1000, "models_tested": 2, "positions_tested": 40,
        "overall_metrics": {
            "t1_headline": {"metric": "spearman_rho", "value": 0.11, "n": 500},
            "t1_zero_baseline": {"n": 480, "mae": 300.0, "zero_mae": 100.0, "mae_ratio": 3.0},
            "t2_legal_rate": 0.97,
            "t2_blunder_rate": {"n": 400, "rate": 0.42, "ci_low": 0.41, "ci_high": 0.43},
            "t2_best_move_vs_random": {"n": 400, "best_move_rate": 0.102, "random_best_move_rate": 0.088,
                                       "ratio_vs_random": 1.16},
            "puzzle_meta": {"setup_move_rate": 0.115, "n_moves": 400, "tactical_theme_rate": 0.06,
                            "n_tactical_explanations": 300},
            "t3_side_mark": {"n": 450, "accuracy": 0.30, "majority_share": 0.54, "cohen_kappa": 0.07},
        },
        "hypothesis_tests": {
            "H1": {"supported": False, "values": {"easy": 1.36, "medium": 1.37, "hard": 1.37, "extreme": 1.42}},
            "H2": {"supported": False, "values": {"easy": 744.0, "medium": 810.0, "hard": 807.0, "extreme": 792.0}},
            "H3": {"supported": False, "values": {"easy": 0.6, "medium": 0.5, "hard": 0.55, "extreme": 0.5}},
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    pd.DataFrame({"model": MODELS, "n": [10, 10], "spearman_rho": [0.2, float("nan")]}).to_csv(
        directory / "t1_rank_correlation.csv", index=False)
    pd.DataFrame({"model": MODELS, "mae_ratio": [1.5, 2.0]}).to_csv(directory / "t1_error_vs_zero.csv", index=False)
    pd.DataFrame({"model": MODELS, "n": [10, 10], "blunder_rate": [0.5, 0.3], "ci_low": [0.45, 0.25],
                  "ci_high": [0.55, 0.35], "rank_tied": [False, False]}).to_csv(
        directory / "t2_blunder_rate.csv", index=False)
    pd.DataFrame({"model": MODELS, "best_move_rate": [0.1, 0.12], "ratio_vs_random": [1.1, 1.3]}).to_csv(
        directory / "t2_best_move_vs_random.csv", index=False)
    pd.DataFrame({"model": MODELS, "setup_move_rate": [0.11, 0.12], "tactical_theme_rate": [0.05, 0.07]}).to_csv(
        directory / "puzzle_meta_metrics.csv", index=False)
    pd.DataFrame({"model": MODELS, "cohen_kappa": [0.05, 0.08]}).to_csv(
        directory / "t3_side_mark_agreement.csv", index=False)
    pd.DataFrame({"model": MODELS, "t2_legal_attempted_mean": [0.98, 0.93], "t3_score_v2_mean": [0.35, 0.57]}).to_csv(
        directory / "by_model.csv", index=False)
    buckets = {"model": MODELS, "n": [10, 10]}
    for key, count in zip(rf.BUCKET_KEYS, [2, 1, 2, 1, 4]):
        buckets[f"{key}_count"] = [count, count]
    pd.DataFrame(buckets).to_csv(directory / "cpl_buckets.csv", index=False)
    pd.DataFrame({"prompt_format": ["cot", "fen_only", "eval_only"], "n": [5, 5, 5],
                  "t1_spearman_rho": [0.12, 0.10, 0.2], "t2_legal_rate": [0.97, 0.96, float("nan")],
                  "t2_blunder_rate": [0.41, 0.43, float("nan")], "t3_score_v2_mean": [0.5, 0.4, float("nan")]}).to_csv(
        directory / "by_prompt_format.csv", index=False)
    pd.DataFrame({"model_family": ["qwen2.5"], "model_size_b": [7.0], "t1_spearman_rho": [0.2],
                  "t2_blunder_rate": [0.5], "t2_legal_rate": [0.98], "t3_score_v2_mean": [0.35],
                  "n_records": [24000]}).to_csv(directory / "by_model_family.csv", index=False)
    return directory


DOC = """# Findings

Intro text stays.

<!-- table:headline -->
old headline
<!-- /table:headline -->

Middle text stays.

<!-- table:models -->
<!-- /table:models -->
"""


def _sources(tmp_path):
    return rf.Sources(
        metrics=_write_metrics(tmp_path / "metrics"),
        rerun_metrics=tmp_path / "missing_rerun",
        comparison=tmp_path / "missing.csv",
        board_test=tmp_path / "missing_board.csv",
    )


# ---------- block replacement ----------

def test_replace_blocks_only_touches_marked_text():
    out = rf.replace_blocks(DOC, {"headline": "NEW", "models": "TABLE"})
    assert "Intro text stays." in out and "Middle text stays." in out
    assert "old headline" not in out
    assert "<!-- table:headline -->\nNEW\n<!-- /table:headline -->" in out
    assert "<!-- table:models -->\nTABLE\n<!-- /table:models -->" in out


def test_unknown_table_raises():
    with pytest.raises(ValueError, match="nosuch"):
        rf.replace_blocks("<!-- table:nosuch -->\n<!-- /table:nosuch -->", {"headline": "x"})


def test_unterminated_marker_raises():
    with pytest.raises(ValueError, match="headline"):
        rf.replace_blocks("<!-- table:headline -->\nno end", {"headline": "x"})


# ---------- table builders ----------

def test_headline_numbers(tmp_path):
    text = rf.render_all(_sources(tmp_path))["headline"]
    assert "0.11" in text  # rho
    assert "42.0% (41.0–43.0%)" in text  # blunder rate with CI
    assert "8.8%" in text and "1.16×" in text  # random baseline
    assert "11.5%" in text  # setup move
    assert "not available" in text  # first-answer legality needs the re-run


def test_models_table_sorted_by_blunder_rate_with_nan_dash(tmp_path):
    lines = rf.render_all(_sources(tmp_path))["models"].splitlines()
    rows = [line for line in lines if line.startswith("| qwen") or line.startswith("| mistral")]
    assert rows[0].startswith("| mistral")  # 30% blunder rate first
    assert "—" in rows[0]  # NaN rho
    assert "50.0% (45.0–55.0%)" in rows[1]


def test_buckets_pooled_shares(tmp_path):
    text = rf.render_all(_sources(tmp_path))["buckets"]
    assert "40.0%" in text  # blunder bucket: 8 of 20
    assert "20.0%" in text


def test_missing_optional_inputs_give_note(tmp_path):
    tables = rf.render_all(_sources(tmp_path))
    for name in ("first_answer", "finetune"):
        assert "not available" in tables[name]


def test_first_answer_and_finetune_tables(tmp_path):
    sources = _sources(tmp_path)
    rerun = tmp_path / "rerun"
    rerun.mkdir()
    pd.DataFrame({"model": ["qwen2.5:7b"] * 2, "prompt_format": ["fen_only", "move_only"], "n": [100, 100],
                  "first_attempt_legal_rate": [0.1, 0.98], "final_legal_rate": [0.98, 0.98]}).to_csv(
        rerun / "t2_move_source.csv", index=False)
    comparison = tmp_path / "comparison.csv"
    pd.DataFrame({"base_model": ["qwen2.5:7b"], "ft_model": ["qwen2.5:7b-board-ft"],
                  "base_t1_spearman": [0.14], "ft_t1_spearman": [0.06],
                  "base_t2_legal_rate": [0.984], "ft_t2_legal_rate": [0.951],
                  "base_t2_blunder_rate": [0.4], "ft_t2_blunder_rate": [0.45],
                  "base_t3_mean": [0.35], "ft_t3_mean": [0.34]}).to_csv(comparison, index=False)
    board = tmp_path / "board.csv"
    pd.DataFrame({"model": ["qwen2.5:7b", "qwen2.5:7b-board-ft"], "base_model": ["qwen2.5:7b"] * 2,
                  "fine_tuned": [False, True], "mean_recall": [0.05, 0.96], "perfect_rate": [0.0, 0.365]}).to_csv(
        board, index=False)
    tables = rf.render_all(rf.Sources(sources.metrics, rerun, comparison, board))
    assert "10.0%" in tables["first_answer"] and "98.0%" in tables["first_answer"]
    assert "5.0% → 96.0%" in tables["finetune"]
    assert "98.4% → 95.1%" in tables["finetune"]


def test_fmt_nan():
    assert rf.pct(math.nan) == "—"
    assert rf.num(float("nan")) == "—"


# ---------- CLI ----------

def _cli(monkeypatch, tmp_path, doc, *extra):
    sources = _sources(tmp_path)
    argv = ["render", "--metrics", str(sources.metrics), "--rerun-metrics", str(sources.rerun_metrics),
            "--comparison", str(sources.comparison), "--board-test", str(sources.board_test),
            "--doc", str(doc), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return rf.main()


def test_check_fails_on_stale_then_passes_after_render(tmp_path, monkeypatch):
    doc = tmp_path / "FINDINGS.md"
    doc.write_text(DOC)
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, tmp_path, doc, "--check")
    assert exc.value.code == 1
    _cli(monkeypatch, tmp_path, doc)
    _cli(monkeypatch, tmp_path, doc, "--check")  # no exit
    assert "Intro text stays." in doc.read_text()


def test_missing_summary_exits_1(tmp_path, monkeypatch):
    doc = tmp_path / "FINDINGS.md"
    doc.write_text(DOC)
    monkeypatch.setattr(sys, "argv", ["render", "--metrics", str(tmp_path / "empty"), "--doc", str(doc)])
    with pytest.raises(SystemExit) as exc:
        rf.main()
    assert exc.value.code == 1


def test_difficulty_and_formats_tables(tmp_path):
    tables = rf.render_all(_sources(tmp_path))
    assert "| easy | 1.36 | 744 | 0.60 |" in tables["difficulty"]
    assert "not supported" in tables["difficulty"]
    assert "| cot | 0.12 | 97.0% | 41.0% | 0.50 |" in tables["formats"]
    assert "| eval_only | 0.20 | — | — | — |" in tables["formats"]


def test_compute_by_prompt_format():
    from src.metrics import compute_by_prompt_format
    truth = [-300, -100, 0, 100, 300]
    rows = []
    for i, t in enumerate(truth):
        rows.append({"prompt_format": "cot", "t1_model_eval": t, "t1_stockfish_eval": t,
                     "t2_legal_attempted": 1.0, "t2_cpl_clamped": 1000.0 if i == 0 else 10.0, "t3_score_v2": 1.0})
        rows.append({"prompt_format": "eval_only", "t1_model_eval": -t, "t1_stockfish_eval": t,
                     "t2_legal_attempted": float("nan"), "t2_cpl_clamped": float("nan"), "t3_score_v2": float("nan")})
    table = compute_by_prompt_format(pd.DataFrame(rows)).set_index("prompt_format")
    assert table.loc["cot", "t1_spearman_rho"] == pytest.approx(1.0)
    assert table.loc["eval_only", "t1_spearman_rho"] == pytest.approx(-1.0)
    assert table.loc["cot", "t2_blunder_rate"] == pytest.approx(0.2)
    assert table.loc["cot", "t2_legal_rate"] == pytest.approx(1.0)
    assert math.isnan(table.loc["eval_only", "t2_legal_rate"])
    assert table.loc["cot", "n"] == 5


def test_default_docs_cover_findings_and_summary():
    assert rf.DEFAULT_DOCS == ("docs/FINDINGS.md", "docs/SIMPLE_SUMMARY.md")
