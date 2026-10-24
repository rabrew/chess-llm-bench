"""Provenance block in summary.json (specs/summary-provenance.md, P-2)."""

import hashlib
import json
import subprocess

import pandas as pd
import pytest
import yaml

from src.metrics import save_metrics
from src.provenance import build_provenance

RECORD = {"job_id": "j", "model": "qwen2.5:7b", "position_id": 1}


def _config(tmp_path, weights=None):
    lc0 = {"binary": "/usr/bin/lc0", "nodes": 800}
    if weights is not None:
        lc0["weights"] = str(weights)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"lc0": lc0, "paths": {"results_file": "r.jsonl"}}))
    return path


def _results(tmp_path, n=3):
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps({**RECORD, "job_id": f"j{i}"}) + "\n" for i in range(n)))
    return path


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_git_sha_and_dirty_flag(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("a")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    prov = build_provenance(_config(tmp_path), _results(tmp_path), repo_dir=tmp_path)
    assert prov["git_sha"] == sha
    assert prov["git_dirty"] is False
    (tmp_path / "f.txt").write_text("changed")
    assert build_provenance(_config(tmp_path), _results(tmp_path), repo_dir=tmp_path)["git_dirty"] is True


def test_outside_git_repo_gives_none(tmp_path):
    prov = build_provenance(_config(tmp_path), _results(tmp_path), repo_dir=tmp_path)
    assert prov["git_sha"] is None
    assert prov["git_dirty"] is None


def test_results_hash_and_count(tmp_path):
    results = _results(tmp_path, n=4)
    prov = build_provenance(_config(tmp_path), results, repo_dir=tmp_path)
    assert prov["results_sha256"] == hashlib.sha256(results.read_bytes()).hexdigest()
    assert prov["results_records"] == 4
    assert prov["generated_at"].endswith("+00:00")


def test_missing_results_file(tmp_path):
    prov = build_provenance(_config(tmp_path), tmp_path / "nope.jsonl", repo_dir=tmp_path)
    assert prov["results_sha256"] is None
    assert prov["results_records"] is None


def test_engine_block_hashes_weights(tmp_path):
    weights = tmp_path / "net.pb"
    weights.write_bytes(b"weights")
    engine = build_provenance(_config(tmp_path, weights), _results(tmp_path), repo_dir=tmp_path)["engine"]
    assert engine["name"] == "lc0"
    assert engine["nodes"] == 800
    assert engine["weights_sha256"] == hashlib.sha256(b"weights").hexdigest()


def test_missing_weights_gives_none_hash(tmp_path):
    engine = build_provenance(_config(tmp_path, tmp_path / "gone.pb"), _results(tmp_path), repo_dir=tmp_path)["engine"]
    assert engine["weights_sha256"] is None


def test_config_snapshot_and_versions(tmp_path):
    prov = build_provenance(_config(tmp_path), _results(tmp_path), repo_dir=tmp_path)
    assert prov["config"]["lc0"]["nodes"] == 800
    assert {"pandas", "numpy", "chess"} <= set(prov["packages"])


def _df():
    return pd.DataFrame([{
        "job_id": "j", "job_type": "standard", "model": "qwen2.5:7b", "position_id": 1, "difficulty": "easy", "phase": "opening",
        "source": "lichess_puzzles", "model_family": "qwen2.5", "model_size_b": 7, "inference_ms": 1,
        "t1_model_eval": 0, "t1_stockfish_eval": 0, "t1_absolute_error": 0, "t1_direction_correct": True,
        "t2_legal": True, "t2_cpl": 0, "t3_p1_side_correct": 1, "t3_p2_theme_correct": 0, "t3_score": 1,
    }])


@pytest.mark.parametrize("provenance", [None, {"git_sha": "abc"}])
def test_save_metrics_writes_provenance_only_when_given(tmp_path, provenance):
    save_metrics(_df(), output_dir=str(tmp_path), provenance=provenance)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary.get("provenance") == provenance
