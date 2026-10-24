"""F-9: before/after board-reading test (specs/board-reading-test.md)."""

import json
import sys
from pathlib import Path

import pytest

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT / "scripts"))
sys.path.insert(0, str(FT_ROOT / "src"))

pytest.importorskip("transformers")
import board_reading_test as brt  # noqa: E402

KINGS = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
POSITION = {"instruction": "List pieces.", "input": f"FEN: {KINGS}"}


def test_prompt_is_identical_for_every_model():
    prompt = brt.build_prompt(POSITION)
    assert prompt == brt.build_prompt(dict(POSITION))
    assert "### Response:\n" in prompt and KINGS in prompt
    assert "Side to move" in prompt  # answer format is spelled out


def test_request_payload_is_raw_and_deterministic():
    payload = brt.build_payload("qwen2.5:7b", "PROMPT")
    assert payload["raw"] is True and payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["seed"] == 42
    assert payload["options"]["stop"] == ["\n###"]


class _FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code != 200:
            raise brt.requests.HTTPError(f"status {self.status_code}")


def test_query_retries_then_succeeds():
    calls = []

    def post(url, json, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise brt.requests.ConnectionError("down")
        return _FakeResponse({"response": "White: King e1"})

    assert brt.query_ollama("m", "p", "http://h", post=post, retry_wait_s=0) == "White: King e1"
    assert len(calls) == 3


def test_query_gives_up_after_retries():
    def post(url, json, timeout):
        raise brt.requests.ConnectionError("down")

    with pytest.raises(brt.requests.ConnectionError):
        brt.query_ollama("m", "p", "http://h", post=post, retry_wait_s=0)


def test_score_row_fields():
    row = brt.score_row("m:ft", "m", True, 3, KINGS, "White: King e1\nBlack: King e8\nSide to move: White", 1.5, "abc")
    assert row["model"] == "m:ft" and row["base_model"] == "m" and row["fine_tuned"] is True
    assert row["idx"] == 3 and row["correct"] is True and row["parsed"] is True
    assert row["digest"] == "abc" and row["seconds"] == 1.5


def test_done_keys_read_existing_rows(tmp_path):
    out = tmp_path / "t0.jsonl"
    out.write_text(json.dumps({"model": "a", "idx": 0}) + "\n" + json.dumps({"model": "a", "idx": 5}) + "\n")
    assert brt.load_done_keys(out) == {("a", 0), ("a", 5)}
    assert brt.load_done_keys(tmp_path / "missing.jsonl") == set()


def test_wilson_interval():
    low, high = brt.wilson_interval(0, 0)
    assert (low, high) == (0.0, 0.0)
    low, high = brt.wilson_interval(100, 100)
    assert high == pytest.approx(1.0) and 0.95 < low < 1.0
    low, high = brt.wilson_interval(50, 100)
    assert low < 0.5 < high


def test_summarise_per_model():
    rows = [
        brt.score_row("a", "a", False, 0, KINGS, "White: King e1\nBlack: King e8\nSide to move: White", 1, "d"),
        brt.score_row("a", "a", False, 1, KINGS, "no idea", 1, "d"),
    ]
    summary = {s["model"]: s for s in brt.summarise(rows)}
    a = summary["a"]
    assert a["n"] == 2 and a["perfect_rate"] == 0.5 and a["unparsed_rate"] == 0.5
    assert a["mean_recall"] == 0.5 and a["side_to_move_acc"] == 1.0


def test_pairs_skip_missing_finetuned_models():
    config = {"models": [
        {"ollama_base": "a", "ollama_ft_tag": "a-board-ft"},
        {"ollama_base": "b", "ollama_ft_tag": "b-board-ft"},
    ]}
    pairs, missing = brt.model_pairs(config, available={"a", "a-board-ft", "b"})
    assert pairs == [("a", "a-board-ft")] and missing == ["b-board-ft"]


def test_format_hint_contains_no_real_squares():
    """A concrete example board gets copied verbatim by base models (seen 2026-10-21)."""
    import re
    assert not re.search(r"\b[a-h][1-8]\b", brt.FORMAT_HINT)


def test_training_style_prompt_matches_training_exactly():
    from finetune_model import INSTRUCTION, format_alpaca_prompt
    assert brt.build_prompt(POSITION, "training") == format_alpaca_prompt(INSTRUCTION, POSITION["input"])
    assert brt.build_prompt(POSITION, "format") == brt.build_prompt(POSITION)


def test_unknown_prompt_style_rejected():
    with pytest.raises(ValueError):
        brt.build_prompt(POSITION, "chatty")


def test_output_paths_depend_on_style():
    from pathlib import Path
    assert brt.output_paths("format") == (
        Path("results/finetuning_experiment/board_t0.jsonl"),
        Path("results/finetuning_experiment/board_t0_summary.csv"),
    )
    assert brt.output_paths("training") == (
        Path("results/finetuning_experiment/board_t0_training_prompt.jsonl"),
        Path("results/finetuning_experiment/board_t0_training_prompt_summary.csv"),
    )


# ---------- --task board+legal (fine-tuning/specs/legal-moves-finetuning.md) ----------

def test_legal_task_training_prompt_uses_legal_instruction():
    from finetune_model import format_alpaca_prompt
    from generate_board_dataset import LEGAL_INSTRUCTION
    assert brt.build_prompt(POSITION, "training", task="board+legal") == format_alpaca_prompt(LEGAL_INSTRUCTION, POSITION["input"])


def test_legal_task_format_prompt_asks_for_legal_moves():
    prompt = brt.build_prompt(POSITION, "format", task="board+legal")
    assert "Legal moves:" in prompt
    assert "Legal moves:" not in brt.build_prompt(POSITION, "format")


def test_legal_task_allows_longer_answers():
    assert brt.build_payload("m", "P", task="board+legal")["options"]["num_predict"] > brt.build_payload("m", "P")["options"]["num_predict"]


def test_legal_task_scores_legal_moves():
    text = "White: King e1\nBlack: King e8\nSide to move: White\nLegal moves: Kd1, Kd2, Ke2, Kf1, Kf2"
    row = brt.score_row("m:ft", "m", True, 0, KINGS, text, 1.0, "d", task="board+legal")
    assert row["legal_found"] is True
    assert row["legal_recall"] == 1.0 and row["legal_precision"] == 1.0
    assert "legal_recall" not in brt.score_row("m", "m", False, 0, KINGS, text, 1.0, "d")


def test_summary_includes_legal_means():
    rows = [brt.score_row("m", "m", False, i, KINGS, "Legal moves: Kd1", 1.0, "d", task="board+legal") for i in range(2)]
    summary = brt.summarise(rows)[0]
    assert summary["mean_legal_recall"] == 0.2
    assert summary["mean_legal_precision"] == 1.0


def test_legal_task_output_paths_are_separate():
    board, legal = brt.output_paths("training"), brt.output_paths("training", task="board+legal")
    assert board != legal
    assert legal[0].name == "board_legal_t0_training_prompt.jsonl"


def test_unknown_task_rejected():
    with pytest.raises(ValueError):
        brt.build_prompt(POSITION, "format", task="moves-only")
