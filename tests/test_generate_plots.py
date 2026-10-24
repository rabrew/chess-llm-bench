"""ρ as the headline T1 number and registry-driven plot metadata (specs/t1-headline-rho.md)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import generate_plots as gp  # noqa: E402
from src.metrics import aggregate_by_model, compute_eval_rank_correlation, generate_summary  # noqa: E402
from src.utils import MODEL_REGISTRY  # noqa: E402

MODELS = ["llama3.2:3b", "gemma3:4b", "qwen2.5:7b", "llama3.2:3b-board-ft"]


def _by_model(rhos, with_rho=True):
    frame = pd.DataFrame({
        "model": MODELS,
        "t1_direction_correct_mean": [0.4, 0.45, 0.5, 0.42],
        "t2_legal_attempted_mean": [0.98, 0.99, 0.97, 0.7],
        "t2_cpl_clamped_mean": [600.0, 650.0, 620.0, 700.0],
        "t2_cpl_mean": [4000.0, 4100.0, 3900.0, 4500.0],
        "t3_score_v2_mean": [0.4, 0.35, 0.5, 0.45],
        "t3_score_mean": [0.5, 0.45, 0.6, 0.55],
    })
    if with_rho:
        frame["t1_spearman_rho"] = rhos
    return frame


def _metrics_dir(tmp_path, frame):
    frame.to_csv(tmp_path / "by_model.csv", index=False)
    return tmp_path


def _t1_frame():
    truth = [-300, -100, 0, 100, 300]
    rows = []
    for size_b, (model, evals) in enumerate((("a", truth), ("b", [5, 3, 1, -2, 4])), start=1):
        rows += [{"model": model, "position_id": i, "difficulty": "easy", "phase": "opening",
                  "source": "lichess_puzzles", "model_family": model, "model_size_b": size_b,
                  "t1_model_eval": e, "t1_stockfish_eval": t,
                  "job_id": f"{model}{i}", "t1_absolute_error": abs(e - t), "t1_direction_correct": True,
                  "t2_legal": True, "t2_cpl": None, "t3_p1_side_correct": None,
                  "t3_p2_theme_correct": None, "t3_score": None, "inference_ms": 1}
                 for i, (e, t) in enumerate(zip(evals, truth))]
    return pd.DataFrame(rows)


# ---------- metrics side ----------

def test_by_model_has_spearman_matching_metrics_module():
    df = _t1_frame()
    agg = aggregate_by_model(df).set_index("model")
    expected = compute_eval_rank_correlation(df).set_index("model")["spearman_rho"]
    assert agg.loc["a", "t1_spearman_rho"] == pytest.approx(1.0)
    assert agg.loc["b", "t1_spearman_rho"] == pytest.approx(expected["b"])


def test_summary_headline_is_spearman():
    headline = generate_summary(_t1_frame())["overall_metrics"]["t1_headline"]
    assert headline["metric"] == "spearman_rho"
    assert headline["n"] == 10
    assert -1 <= headline["value"] <= 1


# ---------- plot data loading ----------

def test_headline_uses_rho_when_present(tmp_path):
    mdf = gp.load_model_df(_metrics_dir(tmp_path, _by_model([0.1, -0.05, 0.2, 0.0])))
    assert list(mdf["t1_headline_mean"]) == [0.1, -0.05, 0.2, 0.0]
    assert "ρ" in mdf["t1_headline_label"].iloc[0]


def test_headline_falls_back_to_direction_accuracy(tmp_path):
    mdf = gp.load_model_df(_metrics_dir(tmp_path, _by_model(None, with_rho=False)))
    assert list(mdf["t1_headline_mean"]) == [0.4, 0.45, 0.5, 0.42]
    assert "direction" in mdf["t1_headline_label"].iloc[0].lower()


def test_family_and_size_from_registry_including_fine_tuned(tmp_path):
    mdf = gp.load_model_df(_metrics_dir(tmp_path, _by_model([0.1, 0.1, 0.1, 0.1]))).set_index("model")
    assert mdf.loc["llama3.2:3b", "family"] == "LLaMA 3"
    assert mdf.loc["llama3.2:3b-board-ft", "family"] == "LLaMA 3"
    assert mdf.loc["llama3.2:3b-board-ft", "size_b"] == 3
    assert mdf.loc["qwen2.5:7b", "family"] == "Qwen 2.5"


def test_unknown_model_raises(tmp_path):
    frame = _by_model([0.1] * 4)
    frame.loc[0, "model"] = "mystery:9b"
    with pytest.raises(ValueError):
        gp.load_model_df(_metrics_dir(tmp_path, frame))


def test_every_registry_family_has_a_display_name():
    assert {family for family, _, _ in MODEL_REGISTRY.values()} <= set(gp.FAMILY_DISPLAY)


# ---------- plots render ----------

@pytest.mark.parametrize("plot, filename", [
    (gp.plot_overview_rankings, "overview_rankings.png"),
    (gp.plot_family_scaling, "family_scaling.png"),
    (gp.plot_summary_heatmap, "summary_heatmap.png"),
])
def test_plots_render_with_negative_rho(tmp_path, plot, filename):
    mdf = gp.load_model_df(_metrics_dir(tmp_path, _by_model([-0.1, -0.05, 0.2, -0.3])))
    out = tmp_path / "plots"
    out.mkdir()
    plot(mdf, out)
    assert (out / filename).stat().st_size > 0
