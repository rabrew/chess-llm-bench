"""Tests for src/utils.py"""

import logging
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils import (
    load_config,
    compute_hash,
    setup_logging,
    ensure_dir,
    ensure_project_dirs,
    get_timestamp,
    get_run_id,
    parse_model_info,
    clamp,
)


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg = {"benchmark": {"random_seed": 42}, "models": ["llama3.2:3b"]}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))
        result = load_config(str(config_file))
        assert result["benchmark"]["random_seed"] == 42
        assert "llama3.2:3b" in result["models"]

    def test_env_override_int(self, tmp_path, monkeypatch):
        cfg = {"benchmark": {"random_seed": 42}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))
        monkeypatch.setenv("CHESS_BENCHMARK_RANDOM_SEED", "99")
        result = load_config(str(config_file))
        assert result["benchmark"]["random_seed"] == 99

    def test_env_override_bool_true(self, tmp_path, monkeypatch):
        cfg = {"correction_loop": {"enabled": False}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))
        monkeypatch.setenv("CHESS_CORRECTION_LOOP_ENABLED", "true")
        result = load_config(str(config_file))
        assert result["correction_loop"]["enabled"] is True

    def test_env_override_bool_false(self, tmp_path, monkeypatch):
        cfg = {"correction_loop": {"enabled": True}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))
        monkeypatch.setenv("CHESS_CORRECTION_LOOP_ENABLED", "0")
        result = load_config(str(config_file))
        assert result["correction_loop"]["enabled"] is False

    def test_env_override_float(self, tmp_path, monkeypatch):
        cfg = {"benchmark": {"threshold": 0.5}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))
        monkeypatch.setenv("CHESS_BENCHMARK_THRESHOLD", "0.9")
        result = load_config(str(config_file))
        assert result["benchmark"]["threshold"] == pytest.approx(0.9)

    def test_env_override_string(self, tmp_path, monkeypatch):
        cfg = {"paths": {"data_dir": "data"}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))
        monkeypatch.setenv("CHESS_PATHS_DATA_DIR", "/new/data")
        result = load_config(str(config_file))
        assert result["paths"]["data_dir"] == "/new/data"

    def test_non_dict_section_not_overridden(self, tmp_path):
        cfg = {"models": ["llama3.2:3b"]}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))
        result = load_config(str(config_file))
        assert result["models"] == ["llama3.2:3b"]


class TestComputeHash:
    def test_deterministic(self):
        h1 = compute_hash("a", "b", "c")
        h2 = compute_hash("a", "b", "c")
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        assert compute_hash("a", "b") != compute_hash("a", "c")

    def test_returns_string(self):
        assert isinstance(compute_hash("x"), str)

    def test_length(self):
        # SHA256 hex digest is 64 chars
        assert len(compute_hash("x")) == 64

    def test_non_string_args(self):
        h = compute_hash(1, 2, 3)
        assert len(h) == 64


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging("test_logger_utils")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = setup_logging("my_test_logger")
        assert logger.name == "my_test_logger"

    def test_with_log_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = setup_logging("test_file_logger", log_file=log_file)
        logger.info("test message")
        assert Path(log_file).exists()

    def test_clears_existing_handlers(self):
        logger = setup_logging("test_clear_logger")
        initial_count = len(logger.handlers)
        logger2 = setup_logging("test_clear_logger")
        assert len(logger2.handlers) == initial_count


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new" / "nested" / "dir"
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert isinstance(result, Path)

    def test_existing_dir_no_error(self, tmp_path):
        result = ensure_dir(tmp_path)
        assert result == tmp_path

    def test_accepts_string(self, tmp_path):
        new_dir = str(tmp_path / "strdir")
        result = ensure_dir(new_dir)
        assert Path(new_dir).exists()


class TestEnsureProjectDirs:
    def test_creates_all_dirs(self, tmp_path):
        config = {
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "jobs_db": str(tmp_path / "jobs/jobs.db"),
                "results_file": str(tmp_path / "results/evaluations.jsonl"),
                "logs_dir": str(tmp_path / "results/logs"),
                "plots_dir": str(tmp_path / "results/plots"),
                "metrics_dir": str(tmp_path / "results/metrics"),
            }
        }
        ensure_project_dirs(config)
        assert (tmp_path / "data").exists()
        assert (tmp_path / "jobs").exists()
        assert (tmp_path / "results").exists()

    def test_empty_paths_uses_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = {}
        ensure_project_dirs(config)  # Should not raise


class TestGetTimestamp:
    def test_returns_string(self):
        ts = get_timestamp()
        assert isinstance(ts, str)

    def test_format(self):
        ts = get_timestamp()
        assert "T" in ts
        assert ts.endswith("Z")


class TestGetRunId:
    def test_returns_string(self):
        run_id = get_run_id()
        assert isinstance(run_id, str)

    def test_format(self):
        run_id = get_run_id()
        assert "_" in run_id
        assert len(run_id) == 15  # YYYYMMDD_HHMMSS


class TestParseModelInfo:
    def test_qwen_model(self):
        info = parse_model_info("qwen2.5:7b")
        assert info["family"] == "qwen"
        assert info["size_b"] == 7
        assert info["size_str"] == "7b"
        assert info["full_tag"] == "qwen2.5:7b"

    def test_llama_model(self):
        info = parse_model_info("llama3.2:3b")
        assert info["family"] == "llama"
        assert info["size_b"] == 3

    def test_mistral_model(self):
        info = parse_model_info("mistral:7b")
        assert info["family"] == "mistral"
        assert info["size_b"] == 7

    def test_phi_model(self):
        info = parse_model_info("phi4:14b")
        assert info["family"] == "phi"
        assert info["size_b"] == 14

    def test_gemma_model(self):
        info = parse_model_info("gemma3:4b")
        assert info["family"] == "gemma"
        assert info["size_b"] == 4

    def test_unknown_family(self):
        info = parse_model_info("wizardlm2:7b")
        assert info["family"] == "unknown"
        assert info["size_b"] == 7

    def test_no_tag(self):
        info = parse_model_info("somemodel")
        assert info["size_b"] == 0
        assert info["size_str"] == ""

    def test_float_size(self):
        info = parse_model_info("solar:10.7b")
        assert info["family"] == "unknown"
        # 10.7b -> size_clean = "10.7" -> float(10.7)
        assert info["size_b"] == pytest.approx(10.7)

    def test_invalid_size(self):
        info = parse_model_info("model:latest")
        assert info["size_b"] == 0


class TestClamp:
    def test_within_range(self):
        assert clamp(50, 0, 100) == 50

    def test_below_min(self):
        assert clamp(-10, 0, 100) == 0

    def test_above_max(self):
        assert clamp(200, 0, 100) == 100

    def test_at_min(self):
        assert clamp(0, 0, 100) == 0

    def test_at_max(self):
        assert clamp(100, 0, 100) == 100

    def test_float_values(self):
        assert clamp(1.5, 0.0, 1.0) == pytest.approx(1.0)
