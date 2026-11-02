#!/usr/bin/env python3
"""Generate audited charts from explicit diagnostic exports."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config

def load_audit_frames(directory):
    paths = [Path(directory) / name for name in ("direction_diagnostics.csv", "cpl_quality.csv")]
    if not all(path.is_file() for path in paths):
        raise ValueError("Missing audited exports; regenerate using scripts/generate_audit_metrics.py.")
    return tuple(pd.read_csv(path) for path in paths)

def heatmap_values(frame, columns):
    normalized = frame[columns].copy()
    for column in columns:
        values = normalized[column]
        spread = values.max() - values.min()
        normalized[column] = (values - values.min()) / spread if spread > 0 else values.where(values.isna(), .5)
    order = normalized.mean(axis=1).sort_values(ascending=False, na_position="last").index
    return frame.loc[order].reset_index(drop=True), normalized.loc[order].to_numpy(float)

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_audit(direction, quality, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = direction.loc[direction.cohort.eq("all")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for axis, (column, title) in zip(axes, (("kappa", "Direction agreement: Cohen's kappa at ±100 cp"), ("spearman_rho", "Rank correlation (reference |cp| < 9000)"))):
        values = cohort.loc[cohort.threshold.eq(100)].sort_values(column)
        axis.barh(values.model, values[column], color="#31688e")
        axis.axvline(0, color="#555", linewidth=.8)
        axis.set_title(title, fontsize=10)
        axis.tick_params(labelsize=8)
    _save(fig, output_dir / "direction_diagnostics.png")
    fig, axis = plt.subplots(figsize=(10, 7))
    if quality.empty:
        axis.text(.5, .5, "Exact endpoint quality unavailable\nHistorical T1 + CPL cannot recover T2 endpoints.", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        values = quality.sort_values("blunder_rate")
        low = values.blunder_rate - values.blunder_rate_ci_low
        high = values.blunder_rate_ci_high - values.blunder_rate
        axis.errorbar(values.blunder_rate, values.model, xerr=[low.clip(lower=0), high.clip(lower=0)], fmt="o", color="#31688e")
        axis.set_xlabel("P(endpoint-clamped CPL ≥ 1000); position-cluster 95% intervals")
    axis.set_title("Move quality from recorded reference endpoints")
    _save(fig, output_dir / "move_quality.png")
    if not cohort.empty:
        pivot = cohort.pivot(index="model", columns="threshold", values="kappa")
        pivot.columns = [f"kappa_t{threshold}" for threshold in pivot.columns]
        frame = pivot.reset_index()
        columns = list(pivot.columns)
        ordered, matrix = heatmap_values(frame, columns)
        fig, axis = plt.subplots(figsize=(8, max(4, len(ordered) * .3)))
        axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axis.set_yticks(range(len(ordered)), ordered.model, fontsize=8)
        axis.set_xticks(range(len(columns)), columns, fontsize=8)
        axis.set_title("Direction robustness: normalized within each threshold")
        _save(fig, output_dir / "summary_heatmap.png")

    charts = ["direction_diagnostics.png", "move_quality.png"]
    if not cohort.empty:
        charts.append("summary_heatmap.png")
    (output_dir / "chart_manifest.json").write_text(json.dumps({
        "current_charts": charts,
        "note": "Only these charts belong to this export; other files are historical.",
    }, indent=2) + "\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--metrics-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--save-metrics", action="store_true", help="Regenerate audited exports before plotting")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config.get("paths", {})
    metrics_dir = args.metrics_dir or paths.get("metrics_dir", "results/metrics")
    output_dir = args.output_dir or paths.get("plots_dir", "results/plots")
    if args.save_metrics:
        from src.metrics import load_results_df, save_metrics
        frame = load_results_df(paths.get("results_file", "results/evaluations.jsonl"))
        if frame.empty:
            parser.error("No result records to plot")
        save_metrics(frame, metrics_dir)
    try:
        direction, quality = load_audit_frames(metrics_dir)
    except ValueError as error:
        parser.error(str(error))
    plot_audit(direction, quality, output_dir)

if __name__ == "__main__":
    main()
