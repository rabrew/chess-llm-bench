"""Where each final move came from, and first-attempt legality (specs/t2-move-source.md)."""

import json
import math

import pandas as pd
import pytest

from src.metrics import compute_move_source_breakdown, load_results_df, save_metrics

# Black to move: Nf6 is legal, e4 is not.
FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
LEGAL_REPLY = "Eval: 20\nMove: Nf6\nExplanation: White — better development."
ILLEGAL_REPLY = "Eval: 20\nMove: e4\nExplanation: White — better development."
NO_RETRY = object()


def _record(fmt="fen_only", raw=LEGAL_REPLY, retry=NO_RETRY, legal=True, model="qwen2.5:7b",
            pos=1, fen=FEN):
    record = {
        "job_id": f"{model}_{pos}_{fmt}", "job_type": "standard", "position_id": pos, "fen": fen,
        "model": model, "prompt_format": fmt,
        "t1_model_eval": None, "t1_stockfish_eval": 0, "t1_absolute_error": None,
        "t2_move": "Nf6" if legal else None, "t2_legal": legal, "t2_cpl": None, "t3_score": None,
        "difficulty": "easy", "phase": "middlegame", "source": "lichess_puzzles",
        "inference_ms": 100, "t1_direction_correct": None,
        "t3_p1_side_correct": None, "t3_p2_theme_correct": None,
    }
    if raw is not None:
        record["raw_response"] = raw
    if retry is not NO_RETRY:
        record["raw_move_retry_response"] = retry
    return record


def _load(tmp_path, records):
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return load_results_df(str(path), data_dir=str(tmp_path / "no-data"))


def _only(tmp_path, record):
    row = _load(tmp_path, [record]).iloc[0]
    return row["t2_move_source"], row["t2_first_attempt_legal"]


# ---------- labels ----------

def test_legal_first_answer(tmp_path):
    assert _only(tmp_path, _record()) == ("first_answer", 1.0)


def test_retry_rescued_move_is_not_first_attempt(tmp_path):
    assert _only(tmp_path, _record(raw=ILLEGAL_REPLY, retry="Nf6")) == ("isolated_retry", 0.0)


def test_text_rescue_without_retry(tmp_path):
    assert _only(tmp_path, _record(raw=ILLEGAL_REPLY)) == ("text_rescue", 0.0)


def test_failed_retry_with_empty_reply_counts_as_retry(tmp_path):
    assert _only(tmp_path, _record(raw=ILLEGAL_REPLY, retry="", legal=False)) == ("isolated_retry", 0.0)


def test_combined_without_raw_reply_is_unknown(tmp_path):
    source, first = _only(tmp_path, _record(raw=None))
    assert source == "unknown"
    assert math.isnan(first)


@pytest.mark.parametrize("raw", [None, "Nf6"])
@pytest.mark.parametrize("legal", [True, False])
def test_move_only_is_always_first_answer(tmp_path, raw, legal):
    assert _only(tmp_path, _record("move_only", raw=raw, legal=legal)) == ("first_answer", float(legal))


@pytest.mark.parametrize("fmt", ["eval_only", "explanation_only"])
def test_prompts_without_a_move_are_nan(tmp_path, fmt):
    source, first = _only(tmp_path, _record(fmt, legal=False))
    assert pd.isna(source)
    assert math.isnan(first)


@pytest.mark.parametrize("fen", ["not a fen", None])
def test_bad_fen_counts_as_illegal_first_answer(tmp_path, fen):
    assert _only(tmp_path, _record(fen=fen, retry="Nf6")) == ("isolated_retry", 0.0)


def test_no_raw_columns_at_all(tmp_path):
    df = _load(tmp_path, [_record(raw=None), _record("move_only", raw=None, pos=2)])
    assert list(df["t2_move_source"]) == ["unknown", "first_answer"]


# ---------- breakdown table ----------

def _mixed_records():
    return [
        _record(pos=1),
        _record(pos=2, raw=ILLEGAL_REPLY),
        _record(pos=3, raw=ILLEGAL_REPLY, retry="Nf6"),
        _record(pos=4, raw=ILLEGAL_REPLY, retry="", legal=False),
        _record(pos=5, raw=None),
        _record("eval_only", pos=6, legal=False),
        _record(model="mistral:7b", raw=None),
    ]


def test_breakdown_rates_and_shares(tmp_path):
    table = compute_move_source_breakdown(_load(tmp_path, _mixed_records()))
    assert list(zip(table["model"], table["prompt_format"])) == [("qwen2.5:7b", "fen_only")]
    row = table.iloc[0]
    assert row["n"] == 4
    assert row["n_unknown"] == 1
    assert row["first_attempt_legal_rate"] == pytest.approx(0.25)
    assert row["text_rescue_share"] == pytest.approx(0.25)
    assert row["isolated_retry_share"] == pytest.approx(0.5)
    assert row["final_legal_rate"] == pytest.approx(0.75)


def test_breakdown_empty_without_known_rows(tmp_path):
    assert compute_move_source_breakdown(_load(tmp_path, [_record(raw=None)])).empty


def test_save_metrics_writes_table_and_summary(tmp_path):
    out = tmp_path / "metrics"
    save_metrics(_load(tmp_path, _mixed_records()), output_dir=str(out))
    assert (out / "t2_move_source.csv").exists()
    pooled = json.loads((out / "summary.json").read_text())["overall_metrics"]["t2_first_attempt_combined"]
    assert pooled["n"] == 4
    assert pooled["first_attempt_legal_rate"] == pytest.approx(0.25)
    assert pooled["final_legal_rate"] == pytest.approx(0.75)


def test_summary_none_without_raw_replies(tmp_path):
    out = tmp_path / "metrics"
    save_metrics(_load(tmp_path, [_record(raw=None)]), output_dir=str(out))
    assert json.loads((out / "summary.json").read_text())["overall_metrics"]["t2_first_attempt_combined"] is None
    assert not (out / "t2_move_source.csv").exists()
