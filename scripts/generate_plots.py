#!/usr/bin/env python3
"""CLI script to generate visualization plots."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    load_results_df,
    aggregate_by_model,
    aggregate_by_difficulty,
    aggregate_by_phase,
    aggregate_by_source,
    aggregate_by_model_family,
    calculate_hallucination_rate,
    calculate_learning_deltas,
    save_metrics,
)
from src.utils import load_config, setup_logging, ensure_dir


def plot_t1_by_difficulty(df, output_dir):
    """Plot T1 error by difficulty tier."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = aggregate_by_difficulty(df)
    if data.empty:
        return

    difficulty_order = ["easy", "medium", "hard", "extreme"]
    data["difficulty"] = pd.Categorical(
        data["difficulty"], categories=difficulty_order, ordered=True
    )
    data = data.sort_values("difficulty")

    sns.barplot(data=data, x="difficulty", y="t1_absolute_error", hue="model", ax=ax)

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Mean Absolute Error (centipawns)")
    ax.set_title("T1: Centipawn Evaluation Error by Difficulty")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "t1_error_by_difficulty.png", dpi=150)
    plt.close()


def plot_t1_by_phase(df, output_dir):
    """Plot T1 error by game phase."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = aggregate_by_phase(df)
    if data.empty:
        return

    phase_order = ["opening", "middlegame", "endgame"]
    data["phase"] = pd.Categorical(data["phase"], categories=phase_order, ordered=True)
    data = data.sort_values("phase")

    sns.barplot(data=data, x="phase", y="t1_absolute_error", hue="model", ax=ax)

    ax.set_xlabel("Game Phase")
    ax.set_ylabel("Mean Absolute Error (centipawns)")
    ax.set_title("T1: Centipawn Evaluation Error by Phase")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "t1_error_by_phase.png", dpi=150)
    plt.close()


def plot_t2_cpl_by_difficulty(df, output_dir):
    """Plot T2 CPL by difficulty."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = aggregate_by_difficulty(df)
    if data.empty:
        return

    difficulty_order = ["easy", "medium", "hard", "extreme"]
    data["difficulty"] = pd.Categorical(
        data["difficulty"], categories=difficulty_order, ordered=True
    )
    data = data.sort_values("difficulty")

    sns.barplot(data=data, x="difficulty", y="t2_cpl", hue="model", ax=ax)

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Mean Centipawn Loss")
    ax.set_title("T2: Move Quality (CPL) by Difficulty")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "t2_cpl_by_difficulty.png", dpi=150)
    plt.close()


def plot_hallucination_rate(df, output_dir):
    """Plot illegal move (hallucination) rate."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = calculate_hallucination_rate(df)
    if data.empty:
        return

    difficulty_order = ["easy", "medium", "hard", "extreme"]
    data["difficulty"] = pd.Categorical(
        data["difficulty"], categories=difficulty_order, ordered=True
    )
    data = data.sort_values("difficulty")

    sns.barplot(data=data, x="difficulty", y="hallucination_rate", hue="model", ax=ax)

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("T2: Illegal Move (Hallucination) Rate")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(output_dir / "t2_hallucination.png", dpi=150)
    plt.close()


def plot_t3_by_difficulty(df, output_dir):
    """Plot T3 explanation score by difficulty."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = aggregate_by_difficulty(df)
    if data.empty:
        return

    difficulty_order = ["easy", "medium", "hard", "extreme"]
    data["difficulty"] = pd.Categorical(
        data["difficulty"], categories=difficulty_order, ordered=True
    )
    data = data.sort_values("difficulty")

    sns.barplot(data=data, x="difficulty", y="t3_score", hue="model", ax=ax)

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Mean T3 Score (0-2)")
    ax.set_title("T3: Explanation Score by Difficulty")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig(output_dir / "t3_score_by_difficulty.png", dpi=150)
    plt.close()


def plot_t3_by_phase(df, output_dir):
    """Plot T3 score by game phase."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = aggregate_by_phase(df)
    if data.empty:
        return

    phase_order = ["opening", "middlegame", "endgame"]
    data["phase"] = pd.Categorical(data["phase"], categories=phase_order, ordered=True)
    data = data.sort_values("phase")

    sns.barplot(data=data, x="phase", y="t3_score", hue="model", ax=ax)

    ax.set_xlabel("Game Phase")
    ax.set_ylabel("Mean T3 Score (0-2)")
    ax.set_title("T3: Explanation Score by Phase")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig(output_dir / "t3_score_by_phase.png", dpi=150)
    plt.close()


def plot_task_profile_radar(df, output_dir):
    """Plot radar chart comparing models across T1/T2/T3."""
    if df.empty:
        return

    data = aggregate_by_model(df)
    if data.empty:
        return

    # Normalize metrics for radar plot (0-1 scale, higher is better)
    # T1: lower error is better, invert
    max_t1 = data["t1_absolute_error_mean"].max()
    if max_t1 > 0:
        data["t1_norm"] = 1 - (data["t1_absolute_error_mean"] / max_t1)
    else:
        data["t1_norm"] = 1

    # T2: legal rate (already 0-1)
    data["t2_norm"] = data["t2_legal_mean"]

    # T3: score normalized to 0-1
    data["t3_norm"] = data["t3_score_mean"] / 2

    categories = ["Evaluation\n(T1)", "Move Quality\n(T2)", "Explanation\n(T3)"]
    n_cats = len(categories)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = [n / n_cats * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]  # Complete the loop

    for _, row in data.iterrows():
        values = [row["t1_norm"], row["t2_norm"], row["t3_norm"]]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.title("Task Profile Radar")
    plt.tight_layout()
    plt.savefig(output_dir / "task_profile_radar.png", dpi=150)
    plt.close()


def plot_parameter_scaling(df, output_dir):
    """Plot performance vs parameter count by model family."""
    if df.empty:
        return

    data = aggregate_by_model_family(df)
    if data.empty or len(data) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("t1_absolute_error", "T1 Error (lower is better)", True),
        ("t2_cpl", "T2 CPL (lower is better)", True),
        ("t3_score", "T3 Score (higher is better)", False),
    ]

    for ax, (metric, title, invert) in zip(axes, metrics):
        for family in data["model_family"].unique():
            family_data = data[data["model_family"] == family].sort_values(
                "model_size_b"
            )
            ax.plot(
                family_data["model_size_b"],
                family_data[metric],
                marker="o",
                label=family,
            )

        ax.set_xlabel("Model Size (Billions)")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Parameter Scaling Analysis")
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_scaling.png", dpi=150)
    plt.close()


def plot_source_comparison(df, output_dir):
    """Plot performance comparison across position sources."""
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    data = aggregate_by_source(df)
    if data.empty:
        return

    # T1 by source
    sns.barplot(data=data, x="source", y="t1_absolute_error", hue="model", ax=axes[0])
    axes[0].set_title("T1 Error by Source")
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].tick_params(axis="x", rotation=45)

    # T2 by source
    sns.barplot(data=data, x="source", y="t2_legal", hue="model", ax=axes[1])
    axes[1].set_title("T2 Legal Move Rate by Source")
    axes[1].set_ylabel("Legal Rate")
    axes[1].tick_params(axis="x", rotation=45)

    # T3 by source
    sns.barplot(data=data, x="source", y="t3_score", hue="model", ax=axes[2])
    axes[2].set_title("T3 Score by Source")
    axes[2].set_ylabel("Mean Score (0-2)")
    axes[2].tick_params(axis="x", rotation=45)

    # Only show legend on last plot
    for ax in axes[:-1]:
        ax.get_legend().remove()
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "source_comparison.png", dpi=150)
    plt.close()


def plot_correction_delta(df, output_dir):
    """Plot net feedback effect from correction loop."""
    if df.empty:
        return

    data = calculate_learning_deltas(df)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Aggregate by model
    agg = data.groupby("model").agg({
        "delta_correction": "mean",
        "delta_control": "mean",
        "net_feedback_effect": "mean",
    }).reset_index()

    x = np.arange(len(agg))
    width = 0.25

    ax.bar(x - width, agg["delta_correction"], width, label="Correction Delta")
    ax.bar(x, agg["delta_control"], width, label="Control Delta")
    ax.bar(x + width, agg["net_feedback_effect"], width, label="Net Effect")

    ax.set_xlabel("Model")
    ax.set_ylabel("CPL Improvement")
    ax.set_title("Correction Loop: Learning Delta Analysis")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["model"], rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "correction_delta.png", dpi=150)
    plt.close()


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from benchmark results"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Path to results JSONL file (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots (overrides config)",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Also save metric aggregations as CSV/JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)
    logger = logging.getLogger("chess_llm_bench")

    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Get paths
    paths = config.get("paths", {})
    results_file = args.results_file or paths.get(
        "results_file", "results/evaluations.jsonl"
    )
    output_dir = Path(
        args.output_dir or paths.get("plots_dir", "results/plots")
    )

    # Ensure output directory exists
    ensure_dir(output_dir)

    # Load results
    import pandas as pd
    df = load_results_df(results_file)

    if df.empty:
        logger.error(f"No results found in {results_file}")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} results from {results_file}")

    # Generate plots
    print("Generating plots...")

    plot_t1_by_difficulty(df, output_dir)
    print("  - t1_error_by_difficulty.png")

    plot_t1_by_phase(df, output_dir)
    print("  - t1_error_by_phase.png")

    plot_t2_cpl_by_difficulty(df, output_dir)
    print("  - t2_cpl_by_difficulty.png")

    plot_hallucination_rate(df, output_dir)
    print("  - t2_hallucination.png")

    plot_t3_by_difficulty(df, output_dir)
    print("  - t3_score_by_difficulty.png")

    plot_t3_by_phase(df, output_dir)
    print("  - t3_score_by_phase.png")

    plot_task_profile_radar(df, output_dir)
    print("  - task_profile_radar.png")

    plot_parameter_scaling(df, output_dir)
    print("  - parameter_scaling.png")

    plot_source_comparison(df, output_dir)
    print("  - source_comparison.png")

    plot_correction_delta(df, output_dir)
    print("  - correction_delta.png")

    print(f"\nPlots saved to {output_dir}")

    # Save metrics if requested
    if args.save_metrics:
        metrics_dir = paths.get("metrics_dir", "results/metrics")
        save_metrics(df, metrics_dir)
        print(f"Metrics saved to {metrics_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
