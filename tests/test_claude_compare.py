"""Offline Claude comparisons preserve the registered criteria."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import claude_compare as cc

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
HAIKU, OPUS = "claude-haiku:5.5", "claude-opus:5.5-low"


def rows(model=HAIKU, positions=4):
    return pd.DataFrame([
        {"job_id": f"{model}-{position}-{fmt}", "job_type": "standard", "model": model,
         "position_id": position, "prompt_format": fmt, "fen": FEN,
         "t1_model_eval": float(position * 100), "t1_stockfish_eval": float(position * 100),
         "t2_legal": True, "t2_cpl_clamped": 0.0, "t2_eval_before": 0,
         "t2_cpl": 0, "t2_move": "e4", "t2_best_move": "e4",
         "t2_first_attempt_legal": 1.0, "t3_side_claimed": "White" if position else "Equal"}
        for position in range(positions) for fmt in cc.FORMATS
    ])


@pytest.fixture(autouse=True)
def quick_bootstrap(monkeypatch):
    def interval(frame, measure, **kwargs):
        value = measure(frame)
        return (value - 0.001, value + 0.001) if np.isfinite(value) else (np.nan, np.nan)
    monkeypatch.setattr(cc.notation, "bootstrap_ci", interval)


def config(positions=4):
    return {"benchmark": {"sample_per_tier": positions // 4, "max_positions_per_tier": 1000},
            "claude": {"batch_discount": 0.5,
                       "prices_per_mtok": {"api-haiku": {"input": 1.0, "output": 2.0}}}}


@pytest.mark.parametrize("local,claude,left_ci,right_ci,expected", [
    (0.35, 0.30, (0.349, 0.351), (0.299, 0.301), "frontier model plays clearly better"),
    (0.35, 0.31, (0.349, 0.351), (0.309, 0.311), "no clear difference"),
    (0.50, 0.20, (0.10, 0.60), (0.15, 0.25), "no clear difference"),
    (0.50, 0.20, (0.25, 0.60), (0.15, 0.25), "no clear difference"),
])
def test_c1_rule_uses_size_and_strict_nonoverlap(local, claude, left_ci, right_ci, expected):
    assert cc.c1_verdict(local, claude, left_ci, right_ci) == expected


def test_c1_pairs_different_model_tags_on_identical_positions():
    local = rows("gemma4:31b")
    claude = rows()
    local.loc[local.position_id >= 2, "t2_cpl_clamped"] = 1000
    claude = claude[claude.position_id != 3]
    result = cc.c1_result(claude, local, expected_positions=4)
    assert result["comparison"]["n_pairs"] == 3
    assert result["comparison"]["left_value"] == pytest.approx(1 / 3)
    assert result["status"] == "pending"


@pytest.mark.parametrize("missing", ["local", "cpl", "reference"])
def test_c1_incomplete_inputs_are_pending(missing):
    local, claude = rows("gemma4:31b"), rows()
    if missing == "local":
        local = local.iloc[0:0]
    elif missing == "cpl":
        claude["t2_cpl_clamped"] = np.nan
    else:
        claude["t2_eval_before"] = np.nan
    assert cc.c1_result(claude, local, expected_positions=4)["status"] == "pending"


def test_c1_one_unscored_legal_move_keeps_result_pending():
    local, claude = rows("gemma4:31b"), rows()
    claude.loc[(claude.position_id == 1) & (claude.prompt_format == "fen_only"), "t2_cpl_clamped"] = np.nan
    result = cc.c1_result(claude, local, expected_positions=4)
    assert result["comparison"]["n_pairs"] == 3
    assert result["status"] == "pending"


def test_different_fens_with_same_id_are_not_paired():
    local, claude = rows("gemma4:31b"), rows()
    claude.loc[claude.position_id == 1, "fen"] = FEN.replace(" w ", " b ")
    result = cc.c1_result(claude, local, expected_positions=4)
    assert result["comparison"]["n_pairs"] == 3
    assert result["comparison"]["fen_mismatches"] == 1
    assert result["status"] == "pending"


def test_c4_preserves_absolute_rho_rule_even_without_improvement():
    result = cc.c4_result(rows(), expected_positions=4)
    assert result["status"] == "complete"
    assert result["verdict"] == cc.notation.MAJOR_BARRIER
    assert result["primary"]["t1_rho"]["difference"] == 0
    assert "absolute" in " ".join(result["limitations"])


def test_c4_missing_cpl_keeps_available_measurements_pending():
    frame = rows()
    frame["t2_cpl_clamped"] = np.nan
    result = cc.c4_result(frame, expected_positions=4)
    assert result["status"] == "pending"
    assert result["primary"]["first_answer_legal"]["right_value"] == 1


def test_c4_one_unscored_legal_move_is_pending():
    frame = rows()
    frame.loc[(frame.position_id == 1) & (frame.prompt_format == "board_text"), "t2_cpl_clamped"] = np.nan
    assert cc.c4_result(frame, expected_positions=4)["status"] == "pending"


def test_c4_undefined_rho_is_pending_with_null_json():
    frame = rows()
    frame["t1_model_eval"] = 0
    result = cc.json_safe(cc.c4_result(frame, expected_positions=4))
    assert result["status"] == "pending"
    assert result["primary"]["t1_rho"]["right_value"] is None
    json.dumps(result, allow_nan=False)


def test_side_pairing_excludes_invalid_labels_and_missing_truth():
    frame = rows()
    frame.loc[(frame.position_id == 1) & (frame.prompt_format == "board_text"), "t3_side_claimed"] = "maybe"
    frame.loc[(frame.position_id == 2) & (frame.prompt_format == "fen_only"), "t1_stockfish_eval"] = np.nan
    result = cc.c4_result(frame, expected_positions=4)
    assert result["primary"]["side_kappa"]["n_pairs"] == 2


def test_same_counts_different_positions_do_not_complete_c4():
    frame = rows()
    frame.loc[(frame.position_id == 3) & (frame.prompt_format == "board_text"), "position_id"] = 99
    assert cc.c4_result(frame, expected_positions=4)["status"] == "pending"


def test_summary_excludes_pilots_and_discloses_duplicates():
    frame = pd.concat([rows(), rows(OPUS), rows("claude-sonnet:5.5-think", 1), rows().iloc[:1]])
    summary = cc.build_summary(frame, rows("gemma4:31b"), config())
    assert set(summary["variants"]) == {HAIKU, OPUS}
    assert summary["excluded_pilots"]["claude-sonnet:5.5-think"] == 3
    assert summary["variants"][HAIKU]["coverage"]["duplicate_records"] == 1
    assert summary["C2"]["status"] == summary["C3"]["status"] == "not run"


def test_empty_summary_is_strict_json_and_pending():
    result = cc.build_summary(pd.DataFrame(), pd.DataFrame(), config())
    assert all(variant["C1"]["status"] == "pending" for variant in result["variants"].values())
    json.dumps(result, allow_nan=False)


def test_cost_counts_retry_once_and_separate_billed_batches():
    raw = [
        {"tag": HAIKU, "custom_id": "job-r1", "batch_id": "b1", "round": 1, "cost_usd": 0.01},
        {"tag": HAIKU, "custom_id": "job-r2", "batch_id": "b2", "round": 2, "cost_usd": 0.02},
        {"tag": HAIKU, "custom_id": "job-r2", "batch_id": "b2", "round": 2, "cost_usd": 0.02},
        {"tag": HAIKU, "custom_id": "job-r1", "batch_id": "b3", "round": 1, "cost_usd": 0.03},
    ]
    cost = cc.summarize_cost(raw, HAIKU, config())
    assert cost["cost_usd"] == pytest.approx(0.06)
    assert cost["requests"] == 3
    assert cost["duplicate_ledger_records"] == 1


def test_cost_reconstructs_tokens_only_when_prices_are_known():
    raw = [{"tag": HAIKU, "custom_id": "r", "api_model": "api-haiku",
            "input_tokens": 1000, "output_tokens": 1000, "stop_reason": "max_tokens", "round": 1}]
    cost = cc.summarize_cost(raw, HAIKU, config())
    assert cost["cost_usd"] == pytest.approx(0.0015)
    assert cost["first_reply_max_tokens"] == 1
    assert cost["reconstructed_requests"] == 1
    assert cc.summarize_cost(None, HAIKU, config())["cost_usd"] is None


def test_cost_without_variant_transactions_is_unavailable_not_free():
    cost = cc.summarize_cost([{"tag": OPUS, "custom_id": "r", "cost_usd": 1}], HAIKU, config())
    assert cost["status"] == "unavailable"
    assert cost["cost_usd"] is None


def test_read_scored_counts_malformed_without_logging_contents(tmp_path):
    path = tmp_path / "scored.jsonl"
    record = rows().iloc[0].to_dict()
    record["raw_response"] = "Eval: 0\nMove: e4\nExplanation: Equal."
    record["t3_score"] = 0
    path.write_text(json.dumps(record) + "\nnot-json\n")
    frame, audit = cc.read_scored(path)
    assert len(frame) == 1
    assert audit["malformed_lines"] == 1
    assert audit["missing_fen"] == 0


def test_read_scored_missing_measurement_columns_stays_pending(tmp_path):
    path = tmp_path / "scored.jsonl"
    path.write_text(json.dumps({"model": HAIKU, "position_id": 1, "prompt_format": "fen_only"}) + "\n")
    frame, audit = cc.read_scored(path)
    assert audit["missing_fen"] == 1
    assert cc.c4_result(frame, expected_positions=4)["status"] == "pending"


def test_write_report_has_json_and_csv(tmp_path):
    summary = cc.build_summary(rows(), rows("gemma4:31b"), config())
    cc.write_report(summary, tmp_path)
    stored = json.loads((tmp_path / "claude_summary.json").read_text())
    table = pd.read_csv(tmp_path / "claude_comparison.csv")
    assert stored["C2"]["status"] == "not run"
    assert len(table) == 6


def test_cli_stays_offline_and_uses_explicit_paths(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: pytest.fail("Network access"))
    monkeypatch.setattr(cc, "build_provenance", lambda *args, **kwargs: {"git_sha": "test"})
    monkeypatch.setattr(cc, "file_sha256", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(cc, "load_config", lambda path: config())
    monkeypatch.setattr(cc, "read_scored", lambda path: (rows() if "claude" in str(path) else rows("gemma4:31b"), {}))
    output = tmp_path / "metrics"
    monkeypatch.setattr(sys, "argv", ["claude_compare.py", "--results", "claude.jsonl",
                                    "--local-results", "local.jsonl", "--output-dir", str(output)])
    cc.main()
    result = json.loads((output / "claude_summary.json").read_text())
    assert result["provenance"]["claude"]["git_sha"] == "test"
