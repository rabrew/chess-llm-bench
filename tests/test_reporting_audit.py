import json
import pandas as pd
import pytest
from scripts.generate_plots import heatmap_values, load_audit_frames, plot_audit
from dashboard import server

def test_heatmap_preserves_identity():
    ordered, matrix = heatmap_values(pd.DataFrame({"model": ["worst", "best"], "score": [0., 1.]}), ["score"])
    assert ordered.model.tolist() == ["best", "worst"]
    assert matrix[:, 0].tolist() == [1., 0.]

def test_missing_quality_chart(tmp_path):
    direction = pd.DataFrame({"model": ["llama3.2:3b"], "cohort": ["all"], "threshold": [100], "kappa": [.1], "spearman_rho": [.2]})
    plot_audit(direction, pd.DataFrame(), tmp_path)
    assert (tmp_path / "direction_diagnostics.png").is_file()
    assert (tmp_path / "move_quality.png").is_file()
    manifest = json.loads((tmp_path / "chart_manifest.json").read_text())
    assert "move_quality.png" in manifest["current_charts"]
    assert "overview_rankings.png" not in manifest["current_charts"]

def test_old_exports_fail(tmp_path):
    with pytest.raises(ValueError, match="regenerate"):
        load_audit_frames(tmp_path)

def test_dashboard_corrected_metrics(tmp_path, monkeypatch):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"model": "llama3.2:3b", "difficulty": "easy", "prompt_format": "cot",
        "fen": "8/8/8/8/8/2k5/8/K7 w - - 0 1", "t1_model_eval": 0, "t1_stockfish_eval": 10000,
        "t1_absolute_error": 10000, "t2_move": "Kb1", "t2_legal": True, "t2_cpl": 5,
        "t2_reference_eval_before": 900, "t2_reference_eval_after": -900,
        "theme": "fork", "t3_explanation": "A fork wins material.", "t3_p1_side_correct": 1, "t3_score": 0}) + "\n")
    monkeypatch.setattr(server, "EVALUATIONS_PATH", str(path))
    models, _, _ = server._compute_metrics()
    assert models[0]["t2_cpl_mean"] == 1800
    assert models[0]["t1_mae_mean"] is None
    assert models[0]["t3_score_mean"] == 2

def test_dashboard_legacy_quality_missing(tmp_path, monkeypatch):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"model": "llama3.2:3b", "prompt_format": "move_only", "t2_move": "e4", "t2_legal": True, "t2_cpl": 500}) + "\n")
    monkeypatch.setattr(server, "EVALUATIONS_PATH", str(path))
    models, _, _ = server._compute_metrics()
    assert models[0]["t2_cpl_mean"] is None
