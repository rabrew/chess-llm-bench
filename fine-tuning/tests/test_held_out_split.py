"""F-2: the 200 held-out board positions are never in the training split."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_ROOT / "src"))
sys.path.insert(0, str(FT_ROOT / "scripts"))

from finetune_model import split_held_out  # noqa: E402

HELD_OUT_N = 200


def _records(n):
    return [{"instruction": "List the pieces.", "input": f"FEN: fen-{i}", "output": "..."} for i in range(n)]


def _fens(records):
    return {r["input"] for r in records}


def test_split_sizes_and_disjoint():
    train, held_out = split_held_out(_records(1000))
    assert len(held_out) == HELD_OUT_N
    assert len(train) == 800
    assert not _fens(train) & _fens(held_out)


def test_split_is_deterministic():
    assert split_held_out(_records(500))[1] == split_held_out(_records(500))[1]


def test_duplicate_fen_raises():
    records = _records(300) + [_records(1)[0]]
    with pytest.raises(ValueError):
        split_held_out(records)


def test_real_dataset_held_out_never_trained_or_benchmarked():
    dataset = REPO_ROOT / "data" / "board_dataset.jsonl"
    results = REPO_ROOT / "results" / "evaluations.jsonl"
    if not dataset.exists():
        pytest.skip("real training dataset not present")
    with open(dataset) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    train, held_out = split_held_out(records)
    assert len(held_out) == HELD_OUT_N
    assert not _fens(train) & _fens(held_out)
    if results.exists():
        from generate_board_dataset import load_benchmark_fens
        held_out_fens = {r["input"].replace("FEN: ", "").strip() for r in held_out}
        assert not held_out_fens & load_benchmark_fens(str(results))
