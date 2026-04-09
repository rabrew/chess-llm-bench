"""Tests for src/result_writer.py"""

import json
import tempfile
from pathlib import Path

import pytest

from src.result_writer import (
    ResultWriter,
    build_result_record,
    load_results,
    get_completed_job_ids,
)


@pytest.fixture
def tmp_results(tmp_path):
    return str(tmp_path / "results" / "evaluations.jsonl")


@pytest.fixture
def sample_job():
    return {
        "job_id": "job_00001_qwen2_5_7b_pgn+fen_1",
        "job_type": "standard",
        "position_id": 1,
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "model": "qwen2.5:7b",
        "prompt_format": "pgn+fen",
        "difficulty": "medium",
        "phase": "opening",
        "source": "lichess_puzzles",
        "theme": "pin",
        "trial": 1,
    }


@pytest.fixture
def sample_scores():
    return {
        "t1_model_eval": 45,
        "t1_stockfish_eval": 50,
        "t1_absolute_error": 5,
        "t1_direction_correct": True,
        "t2_move": "Nf6",
        "t2_best_move": "Nf6",
        "t2_legal": True,
        "t2_cpl": 0,
        "t3_explanation": "Equal — Both sides are equal.",
        "t3_side_claimed": "Equal",
        "t3_p1_side_correct": 1,
        "t3_p2_theme_correct": 0,
        "t3_score": 1,
    }


@pytest.fixture
def sample_parsed():
    return {
        "eval": 45,
        "move": "Nf6",
        "explanation": "Equal — Both sides are equal.",
        "side_claimed": "Equal",
        "parse_errors": [],
    }


class TestResultWriter:
    def test_write_result(self, tmp_results):
        writer = ResultWriter(tmp_results)
        writer.write_result({"job_id": "test_1", "model": "test"})
        results = load_results(tmp_results)
        assert len(results) == 1
        assert results[0]["job_id"] == "test_1"

    def test_write_multiple_results(self, tmp_results):
        writer = ResultWriter(tmp_results)
        records = [{"job_id": f"job_{i}"} for i in range(5)]
        writer.write_results(records)
        results = load_results(tmp_results)
        assert len(results) == 5

    def test_append_behavior(self, tmp_results):
        writer = ResultWriter(tmp_results)
        writer.write_result({"job_id": "first"})
        writer.write_result({"job_id": "second"})
        results = load_results(tmp_results)
        assert len(results) == 2
        assert results[0]["job_id"] == "first"
        assert results[1]["job_id"] == "second"

    def test_creates_directory(self, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "results.jsonl")
        writer = ResultWriter(deep_path)
        writer.write_result({"job_id": "x"})
        assert Path(deep_path).exists()


class TestBuildResultRecord:
    def test_basic_record(self, sample_job, sample_scores, sample_parsed):
        record = build_result_record(sample_job, sample_parsed, sample_scores, 1500)
        assert record["job_id"] == sample_job["job_id"]
        assert record["model"] == "qwen2.5:7b"
        assert record["model_family"] == "qwen"
        assert record["model_size_b"] == 7
        assert record["inference_ms"] == 1500
        assert record["t1_absolute_error"] == 5
        assert record["t2_legal"] is True
        assert record["t3_score"] == 1

    def test_parse_errors_included(self, sample_job, sample_scores):
        parsed_with_errors = {
            "eval": None,
            "move": None,
            "explanation": None,
            "side_claimed": None,
            "parse_errors": ["Missing Eval field", "Missing Move field"],
        }
        record = build_result_record(sample_job, parsed_with_errors, sample_scores, 500)
        assert "parse_errors" in record
        assert "Missing Eval field" in record["parse_errors"]

    def test_no_parse_errors_key_when_empty(self, sample_job, sample_scores, sample_parsed):
        record = build_result_record(sample_job, sample_parsed, sample_scores, 500)
        assert "parse_errors" not in record

    def test_optional_job_fields(self, sample_scores, sample_parsed):
        minimal_job = {
            "job_id": "test_job",
            "position_id": 1,
            "model": "llama3.2:3b",
        }
        record = build_result_record(minimal_job, sample_parsed, sample_scores, 100)
        assert record["job_type"] == "standard"
        assert record["attempt"] == 1
        assert record["parent_job_id"] is None
        assert record["prompt_format"] == "pgn+fen"

    def test_correction_job_type(self, sample_scores, sample_parsed):
        job = {
            "job_id": "correction_job",
            "position_id": 2,
            "model": "mistral:7b",
            "job_type": "correction",
            "trial": 2,
            "parent_job_id": "parent_001",
        }
        record = build_result_record(job, sample_parsed, sample_scores, 200)
        assert record["job_type"] == "correction"
        assert record["attempt"] == 2
        assert record["parent_job_id"] == "parent_001"


class TestLoadResults:
    def test_returns_empty_for_missing_file(self, tmp_path):
        result = load_results(str(tmp_path / "nonexistent.jsonl"))
        assert result == []

    def test_loads_valid_jsonl(self, tmp_path):
        f = tmp_path / "results.jsonl"
        records = [{"job_id": f"j{i}", "val": i} for i in range(3)]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        result = load_results(str(f))
        assert len(result) == 3
        assert result[0]["job_id"] == "j0"

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text('{"job_id": "a"}\n\n{"job_id": "b"}\n')
        result = load_results(str(f))
        assert len(result) == 2

    def test_skips_malformed_lines(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text('{"job_id": "a"}\nnot json\n{"job_id": "b"}\n')
        result = load_results(str(f))
        assert len(result) == 2


class TestGetCompletedJobIds:
    def test_returns_empty_set_for_missing_file(self, tmp_path):
        ids = get_completed_job_ids(str(tmp_path / "nonexistent.jsonl"))
        assert ids == set()

    def test_returns_job_ids(self, tmp_path):
        f = tmp_path / "results.jsonl"
        records = [{"job_id": f"job_{i}"} for i in range(3)]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        ids = get_completed_job_ids(str(f))
        assert ids == {"job_0", "job_1", "job_2"}

    def test_skips_records_without_job_id(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text('{"model": "x"}\n{"job_id": "j1"}\n')
        ids = get_completed_job_ids(str(f))
        assert ids == {"j1"}
