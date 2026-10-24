"""Fine-tuned benchmark run setup (specs/finetuned-benchmark-run.md)."""

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import check_ft_run as chk  # noqa: E402

BASE_CFG = REPO / "config" / "config.yaml"
FT_CFG = REPO / "config" / "config_finetuned.yaml"
BATCH_1 = {
    "llama3.2:3b-board-ft", "gemma3:4b-board-ft", "qwen2.5:7b-board-ft",
    "mistral:7b-board-ft", "deepseek-r1:7b-board-ft",
}  # gemma4:e2b-board-ft excluded, see the spec


def _load(path):
    return yaml.safe_load(path.read_text())


# --- the real config files --------------------------------------------------

def test_ft_config_matches_base_except_models_and_paths():
    assert chk.config_differences(_load(BASE_CFG), _load(FT_CFG)) == []


def test_ft_config_lists_batch_one():
    assert set(_load(FT_CFG)["models"]) == BATCH_1


def test_ft_config_writes_to_separate_files():
    base, ft = _load(BASE_CFG)["paths"], _load(FT_CFG)["paths"]
    for key in ("jobs_db", "results_file", "logs_dir", "plots_dir", "metrics_dir"):
        assert ft[key] != base[key], key
    assert ft["data_dir"] == base["data_dir"]


def test_ft_tags_have_bases_in_base_config():
    assert chk.tag_problems(_load(FT_CFG)["models"], _load(BASE_CFG)["models"]) == []


# --- pure checks --------------------------------------------------------------

def test_config_differences_reports_changed_keys():
    base = {"models": ["a"], "paths": {"x": 1}, "benchmark": {"seed": 42}, "workers": {"count": 1}}
    ft = {"models": ["b"], "paths": {"x": 2}, "benchmark": {"seed": 7}}
    assert chk.config_differences(base, ft) == ["benchmark", "workers"]


def test_tag_problems():
    assert chk.tag_problems(["qwen2.5:7b-board-ft"], ["qwen2.5:7b"]) == []
    problems = chk.tag_problems(["qwen2.5:7b", "phi4:14b-board-ft", "nosuch:1b-board-ft"], ["qwen2.5:7b"])
    assert len(problems) == 3


def test_fen_set_problems():
    assert chk.fen_set_problems("easy", {"a", "b"}, {"a", "b"}) == []
    [msg] = chk.fen_set_problems("easy", {"a", "b"}, {"a", "c"})
    assert "easy" in msg and "1 base-run positions" in msg


def test_base_run_gaps_are_warnings_not_problems():
    assert chk.fen_set_problems("easy", {"a", "b", "c"}, {"a"}) == []
    [msg] = chk.coverage_gaps("easy", {"a", "b", "c"}, {"a"})
    assert "2 of 3" in msg
    assert chk.coverage_gaps("easy", {"a"}, {"a"}) == []


def test_fen_set_problems_empty_base():
    [msg] = chk.fen_set_problems("hard", {"a"}, set())
    assert "no base records" in msg


def test_foreign_models_in_results(tmp_path):
    results = tmp_path / "r.jsonl"
    assert chk.foreign_models(results) == set()  # missing file is fine
    results.write_text('{"model": "qwen2.5:7b-board-ft"}\n{"model": "qwen2.5:7b"}\n\n')
    assert chk.foreign_models(results) == {"qwen2.5:7b"}


# --- launcher -------------------------------------------------------------------

def test_launcher_is_config_driven():
    script = (REPO / "bin" / "run_all.sh").read_text()
    assert "/home/rabrew" not in script
    assert "BENCH_CONFIG" in script and "SKIP_INSTALL" in script
    assert "results/evaluations.jsonl" not in script
    assert "jobs/db/jobs.db" not in script


@pytest.mark.parametrize("step", ["retry_illegal_moves.py --input", "enrich_cpl.py --results"])
def test_launcher_passes_paths_to_post_steps(step):
    assert step in (REPO / "bin" / "run_all.sh").read_text()


# --- base re-run on current code (Issue 18) -----------------------------------

RERUN_CFG = REPO / "config" / "config_base_rerun.yaml"
OUTPUT_PATHS = ("jobs_db", "results_file", "logs_dir", "plots_dir", "metrics_dir")


def test_rerun_config_matches_base_except_models_and_paths():
    assert chk.config_differences(_load(BASE_CFG), _load(RERUN_CFG)) == []


def test_rerun_config_lists_the_bases_of_batch_one():
    assert set(_load(RERUN_CFG)["models"]) == {tag.removesuffix("-board-ft") for tag in BATCH_1}


def test_rerun_config_writes_to_its_own_files():
    rerun = _load(RERUN_CFG)["paths"]
    for other in (_load(BASE_CFG)["paths"], _load(FT_CFG)["paths"]):
        for key in OUTPUT_PATHS:
            assert rerun[key] != other[key], key
        assert rerun["data_dir"] == other["data_dir"]


# --- pieces + legal moves fine-tune (fine-tuning/specs/legal-moves-finetuning.md) ---

LEGAL_CFG = REPO / "config" / "config_legal_ft.yaml"
LEGAL_SUFFIX = "-board-legal-ft"


def test_legal_config_matches_base_except_models_and_paths():
    assert chk.config_differences(_load(BASE_CFG), _load(LEGAL_CFG)) == []


def test_legal_config_lists_batch_one_legal_tags():
    expected = {tag.removesuffix("-board-ft") + LEGAL_SUFFIX for tag in BATCH_1}
    assert set(_load(LEGAL_CFG)["models"]) == expected


def test_legal_config_writes_to_its_own_files():
    legal = _load(LEGAL_CFG)["paths"]
    for other in (_load(BASE_CFG)["paths"], _load(FT_CFG)["paths"], _load(RERUN_CFG)["paths"]):
        for key in OUTPUT_PATHS:
            assert legal[key] != other[key], key


def test_legal_tags_have_bases_in_base_config():
    assert chk.tag_problems(_load(LEGAL_CFG)["models"], _load(BASE_CFG)["models"]) == []


@pytest.mark.parametrize("tag", ["llama3.2:3b-board-ft", "llama3.2:3b-board-legal-ft"])
def test_fine_tuned_suffixes_resolve_to_base(tag):
    from src.utils import base_model_tag, parse_model_info
    assert base_model_tag(tag) == "llama3.2:3b"
    info = parse_model_info(tag)
    assert info["family"] == "llama3"
    assert info["fine_tuned"] is True


def test_base_model_tag_leaves_base_unchanged():
    from src.utils import base_model_tag
    assert base_model_tag("qwen2.5:7b") == "qwen2.5:7b"


def test_legal_finetune_config_matches_batch_one_settings():
    ft = yaml.safe_load((REPO / "fine-tuning" / "config" / "finetune_config.yaml").read_text())
    legal = yaml.safe_load((REPO / "fine-tuning" / "config" / "finetune_config_legal.yaml").read_text())
    differing = {k for k in ft["training"] if ft["training"][k] != legal["training"].get(k)}
    assert differing == {"dataset_path", "max_seq_length"}
    assert legal["training"]["max_seq_length"] == 768
    tags = {m["ollama_ft_tag"] for m in legal["models"]}
    assert tags == {tag.removesuffix("-board-ft") + LEGAL_SUFFIX for tag in BATCH_1}
    hf_ids = {m["ollama_base"]: m["hf_id"] for m in ft["models"]}
    assert all(m["hf_id"] == hf_ids[m["ollama_base"]] for m in legal["models"])


def test_run_batch_honours_ft_config():
    script = (REPO / "fine-tuning" / "scripts" / "run_batch.sh").read_text()
    assert "FT_CONFIG" in script
    assert script.count('--config "$FT_CONFIG"') == 3
