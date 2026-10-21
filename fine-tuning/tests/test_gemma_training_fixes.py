"""Tests for fine-tuning/specs/gemma-training-fixes.md."""

import re
import sys
from pathlib import Path

import pytest
import yaml

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT / "scripts"))

pytest.importorskip("transformers")
from finetune_model import build_load_kwargs  # noqa: E402

CONFIG_PATH = FT_ROOT / "config" / "finetune_config.yaml"
HF_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
VERIFIED_GEMMA4_IDS = {
    "gemma4:e2b": "google/gemma-4-E2B-it",
    "gemma4:e4b": "google/gemma-4-E4B-it",
    "gemma4:26b": "google/gemma-4-26B-A4B-it",
    "gemma4:31b": "google/gemma-4-31B-it",
}
TRAINING = {"max_seq_length": 512}


@pytest.fixture(scope="module")
def entries():
    return yaml.safe_load(CONFIG_PATH.read_text())["models"]


def test_no_unverified_hf_ids():
    assert "VERIFY" not in CONFIG_PATH.read_text()


def test_hf_ids_are_org_slash_name(entries):
    for entry in entries:
        assert HF_ID_PATTERN.match(entry["hf_id"]), entry


def test_gemma4_ids_match_huggingface(entries):
    by_tag = {entry["ollama_base"]: entry["hf_id"] for entry in entries}
    for tag, hf_id in VERIFIED_GEMMA4_IDS.items():
        assert by_tag[tag] == hf_id


def test_every_gemma_entry_uses_eager_attention(entries):
    gemma = [entry for entry in entries if entry["ollama_base"].startswith("gemma")]
    assert gemma
    for entry in gemma:
        assert entry.get("attn_implementation") == "eager", entry


def test_non_gemma_load_kwargs_unchanged():
    entry = {"ollama_base": "qwen2.5:7b", "hf_id": "Qwen/Qwen2.5-7B-Instruct"}
    assert build_load_kwargs(entry, TRAINING, use_grad_ckpt=False, use_cpu_offload=False) == {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "max_seq_length": 512,
        "dtype": None,
        "load_in_4bit": True,
    }


def test_attn_implementation_passed_through():
    entry = {"ollama_base": "gemma3:4b", "hf_id": "google/gemma-3-4b-it", "attn_implementation": "eager"}
    kwargs = build_load_kwargs(entry, TRAINING, use_grad_ckpt=True, use_cpu_offload=True)
    assert kwargs["attn_implementation"] == "eager"
    assert kwargs["use_gradient_checkpointing"] is True
    assert kwargs["device_map"] == "auto"


def test_bad_attn_implementation_rejected():
    entry = {"ollama_base": "gemma3:4b", "hf_id": "google/gemma-3-4b-it", "attn_implementation": "flex"}
    with pytest.raises(ValueError, match="gemma3:4b"):
        build_load_kwargs(entry, TRAINING, use_grad_ckpt=False, use_cpu_offload=False)
