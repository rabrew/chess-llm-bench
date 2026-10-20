"""Tests for the explicit model registry (specs/model-registry.md)."""

import json

import pandas as pd
import pytest
import yaml

from src.metrics import aggregate_by_model_family, load_results_df
from src.utils import MODEL_REGISTRY, parse_model_info


def _configured_tags() -> list[str]:
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)["models"]


@pytest.mark.parametrize("tag", _configured_tags())
def test_every_configured_model_is_registered(tag):
    info = parse_model_info(tag)
    assert info["family"] and info["family"] != "unknown"
    assert info["size_b"] > 0


def test_effective_and_moe_sizes():
    assert parse_model_info("gemma4:e2b")["size_b"] == 2
    assert parse_model_info("gemma4:e4b")["size_b"] == 4
    mixtral = parse_model_info("mixtral:8x7b")
    assert mixtral["size_b"] == pytest.approx(46.7)
    assert mixtral["active_b"] == pytest.approx(12.9)
    assert mixtral["is_moe"] is True
    assert parse_model_info("mistral:7b")["is_moe"] is False


def test_codellama_not_matched_as_llama():
    assert parse_model_info("codellama:34b")["family"] == "codellama"


def test_previously_merged_models_are_distinct():
    pairs = [("codellama:34b", "yi:34b"), ("deepseek-r1:7b", "wizardlm2:7b"), ("gemma4:e2b", "gemma4:e4b")]
    for a, b in pairs:
        ia, ib = parse_model_info(a), parse_model_info(b)
        assert (ia["family"], ia["size_b"]) != (ib["family"], ib["size_b"])


def test_fine_tuned_suffix():
    info = parse_model_info("qwen2.5:7b-board-ft")
    base = parse_model_info("qwen2.5:7b")
    assert info["family"] == base["family"] and info["size_b"] == base["size_b"]
    assert info["fine_tuned"] is True
    assert base["fine_tuned"] is False


def test_registry_sizes_positive():
    assert all(size > 0 for _, size, _ in MODEL_REGISTRY.values())


def test_load_results_df_replaces_stored_family_and_size(tmp_path):
    path = tmp_path / "r.jsonl"
    rows = [
        {"job_id": "a", "model": "gemma4:e4b", "model_family": "gemma", "model_size_b": 0},
        {"job_id": "b", "model": "yi:34b", "model_family": "unknown", "model_size_b": 34},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    df = load_results_df(str(path), data_dir=str(tmp_path)).set_index("job_id")
    assert df.loc["a", "model_family"] == "gemma4" and df.loc["a", "model_size_b"] == 4
    assert df.loc["b", "model_family"] == "yi"


def test_family_table_one_model_per_row():
    df = pd.DataFrame({
        "model": ["gemma4:e2b", "gemma4:e4b"],
        "model_family": ["gemma4", "gemma4"],
        "model_size_b": [2.0, 4.0],
        "job_id": ["a", "b"],
        "t1_absolute_error": [1, 2], "t2_cpl": [1, 2], "t2_legal": [True, True], "t3_score": [1, 2],
    })
    assert len(aggregate_by_model_family(df)) == 2


def test_family_table_rejects_merged_models():
    df = pd.DataFrame({
        "model": ["a:1b", "b:1b"],
        "model_family": ["x", "x"],
        "model_size_b": [1.0, 1.0],
        "job_id": ["a", "b"],
        "t1_absolute_error": [1, 2], "t2_cpl": [1, 2], "t2_legal": [True, True], "t3_score": [1, 2],
    })
    with pytest.raises(ValueError, match="more than one model"):
        aggregate_by_model_family(df)
