"""T3 phrasing example finder (specs/t3-example-finder.md, T3-2)."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import find_t3_examples as finder  # noqa: E402
from src.evaluator import phrase_mentioned  # noqa: E402

FORK_CUE = "attacks two"


def _rec(theme, text, model="qwen2.5:7b"):
    return {"model": model, "theme": theme, "t3_explanation": text}


def _write(tmp_path, records, name="r.jsonl", extra_lines=()):
    path = tmp_path / name
    lines = [json.dumps(r) for r in records] + list(extra_lines)
    path.write_text("\n".join(lines) + "\n")
    return path


def _cue_row(table, theme, cue):
    rows = table[(table["theme"] == theme) & (table["cue"] == cue)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_phrase_mentioned_is_whole_word():
    assert phrase_mentioned("pin", "The knight is pinned to the king")
    assert not phrase_mentioned("pin", "In my opinion White is better")


def test_gain_counts_only_unmarked_explanations(tmp_path):
    path = _write(tmp_path, [
        _rec("fork", "White — the knight attacks two pieces at once."),
        _rec("fork", "White — a knight fork attacks two pieces."),
        _rec("pin", "White — the bishop pins the knight."),
    ])
    scan = finder.scan([path])
    row = _cue_row(finder.cue_table(scan), "fork", FORK_CUE)
    assert row["gain"] == 1


def test_leak_counts_other_tactical_themes(tmp_path):
    records = [_rec("fork", "White — the knight attacks two pieces.")]
    records += [_rec("pin", "White — the bishop pins it and attacks two pawns.")] * 3
    records += [_rec("skewer", "White — skewer on the diagonal.")]
    table = finder.cue_table(finder.scan([_write(tmp_path, records)]))
    row = _cue_row(table, "fork", FORK_CUE)
    assert row["leak_hits"] == 3
    assert row["leak_rate"] == pytest.approx(3 / 4)


def test_high_leak_fails_threshold_despite_gain(tmp_path):
    records = [_rec("fork", "White — the knight attacks two pieces.")] * (finder.MIN_GAIN + 5)
    records += [_rec("pin", "Black — the queen attacks two pawns.")] * 50
    row = _cue_row(finder.cue_table(finder.scan([_write(tmp_path, records)])), "fork", FORK_CUE)
    assert row["gain"] >= finder.MIN_GAIN
    assert not row["passes"]


def test_low_leak_high_gain_passes(tmp_path):
    records = [_rec("fork", "White — the knight attacks two pieces.")] * (finder.MIN_GAIN + 5)
    records += [_rec("pin", "Black — the bishop pins the rook.")] * 100
    row = _cue_row(finder.cue_table(finder.scan([_write(tmp_path, records)])), "fork", FORK_CUE)
    assert row["passes"]


def test_no_other_themes_gives_nan_leak_and_fails(tmp_path):
    records = [_rec("fork", "White — the knight attacks two pieces.")] * (finder.MIN_GAIN + 5)
    row = _cue_row(finder.cue_table(finder.scan([_write(tmp_path, records)])), "fork", FORK_CUE)
    assert math.isnan(row["leak_rate"])
    assert not row["passes"]


def test_non_tactical_themes_not_scanned(tmp_path):
    scan = finder.scan([_write(tmp_path, [_rec("crushing", "White is winning.")])])
    assert scan.tactical == 0


def test_skips_bad_records_and_malformed_lines(tmp_path):
    path = _write(tmp_path, [_rec("fork", None), {"model": "x"}, _rec("fork", "a fork")],
                  extra_lines=["{not json"])
    scan = finder.scan([path])
    assert scan.scanned == 1


def test_main_writes_report(tmp_path, monkeypatch):
    path = _write(tmp_path, [_rec("fork", "White — the knight attacks two pieces."),
                             _rec("pin", "Black — the bishop pins the rook.")])
    out = tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", ["find", "--results", str(path), "--out", str(out)])
    finder.main()
    report = out.read_text()
    assert "| theme | cue | gain | leak_hits | leak_rate | passes |" in report
    assert "Unmarked explanations to read by hand" in report


def test_main_missing_file_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["find", "--results", str(tmp_path / "nope.jsonl")])
    with pytest.raises(SystemExit) as exc:
        finder.main()
    assert exc.value.code == 1
