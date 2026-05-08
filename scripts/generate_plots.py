#!/usr/bin/env python3
"""Generate benchmark visualisation plots.

Five charts, each telling one clear story:
  1. overview_rankings.png    — all models ranked on every metric (horizontal bars)
  2. family_scaling.png       — does scale help? performance vs parameter count by family
  3. difficulty_profiles.png  — how metrics change from easy → extreme, per family
  4. verbal_vs_mechanical.png — T3 explanation score vs T2 move quality (scatter)
  5. summary_heatmap.png      — all models × all metrics, normalised (heatmap)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, setup_logging, ensure_dir

# ---------------------------------------------------------------------------
# Model metadata — family and parameter count
# ---------------------------------------------------------------------------

MODEL_META = {
    "llama3.2:3b":     {"family": "LLaMA 3",    "size": 3},
    "llama3.1:8b":     {"family": "LLaMA 3",    "size": 8},
    "llama3.3:70b":    {"family": "LLaMA 3",    "size": 70},
    "gemma3:4b":       {"family": "Gemma 3",    "size": 4},
    "gemma3:12b":      {"family": "Gemma 3",    "size": 12},
    "gemma4:e2b":      {"family": "Gemma 4",    "size": 2},
    "gemma4:e4b":      {"family": "Gemma 4",    "size": 4},
    "gemma4:26b":      {"family": "Gemma 4",    "size": 26},
    "gemma4:31b":      {"family": "Gemma 4",    "size": 31},
    "qwen2.5:7b":      {"family": "Qwen 2.5",   "size": 7},
    "qwen2.5:14b":     {"family": "Qwen 2.5",   "size": 14},
    "qwen2.5:32b":     {"family": "Qwen 2.5",   "size": 32},
    "deepseek-r1:7b":  {"family": "DeepSeek-R1","size": 7},
    "deepseek-r1:14b": {"family": "DeepSeek-R1","size": 14},
    "mistral:7b":      {"family": "Mistral",    "size": 7},
    "mixtral:8x7b":    {"family": "Mistral",    "size": 47},
    "solar:10.7b":     {"family": "Solar",      "size": 11},
    "phi4:14b":        {"family": "Phi-4",      "size": 14},
    "wizardlm2:7b":    {"family": "WizardLM",   "size": 7},
    "codellama:34b":   {"family": "CodeLlama",  "size": 34},
    "yi:34b":          {"family": "Yi",         "size": 34},
    "command-r:35b":   {"family": "Command-R",  "size": 35},
}

FAMILY_ORDER = [
    "LLaMA 3", "Gemma 3", "Gemma 4", "Qwen 2.5",
    "DeepSeek-R1", "Mistral", "Solar", "Phi-4",
    "WizardLM", "CodeLlama", "Yi", "Command-R",
]

FAMILY_COLORS = {
    "LLaMA 3":    "#1f77b4",
    "Gemma 3":    "#ff7f0e",
    "Gemma 4":    "#2ca02c",
    "Qwen 2.5":   "#d62728",
    "DeepSeek-R1":"#9467bd",
    "Mistral":    "#8c564b",
    "Solar":      "#e377c2",
    "Phi-4":      "#bcbd22",
    "WizardLM":   "#17becf",
    "CodeLlama":  "#aec7e8",
    "Yi":         "#ffbb78",
    "Command-R":  "#98df8a",
}

DIFFICULTY_ORDER = ["easy", "medium", "hard", "extreme"]

# Short display names for models (strip redundant tag info)
def short_name(m: str) -> str:
    return m.replace("deepseek-r1", "ds-r1").replace("command-r", "cmd-r")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_df(metrics_dir: Path) -> pd.DataFrame:
    """Load by_model.csv, annotate with family/size.

    Adds aliases so the rest of the plotting code can reference the headline
    metric as ``t2_cpl_headline_mean`` and ``t3_score_headline_mean``
    regardless of whether the underlying file has the new (clamped /
    rescored) columns or only the legacy raw ones.
    """
    path = metrics_dir / "by_model.csv"
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found at {path}. Run with --save-metrics first.")
    df = pd.read_csv(path)
    df["family"] = df["model"].map(lambda m: MODEL_META.get(m, {}).get("family", "Other"))
    df["size_b"] = df["model"].map(lambda m: MODEL_META.get(m, {}).get("size", 0))
    df["display"] = df["model"].map(short_name)
    # Headline legality: prefer t2_legal_attempted (only counts move-asking
    # prompts) when available; falls back to legacy column for older files.
    df["t2_legal_headline_mean"] = df.get(
        "t2_legal_attempted_mean", df.get("t2_legal_mean")
    )
    df["illegal_rate"] = 1 - df["t2_legal_headline_mean"]
    # Headline CPL: prefer the clamped column if available (Lichess convention,
    # ±1000 cp). Falls back to the raw uncapped column for older metrics files.
    df["t2_cpl_headline_mean"] = df.get("t2_cpl_clamped_mean", df["t2_cpl_mean"])
    df["t2_cpl_headline_label"] = (
        "CPL (clamped ±1000 cp)" if "t2_cpl_clamped_mean" in df.columns
        else "CPL (raw)"
    )
    df["t3_score_headline_mean"] = df.get("t3_score_v2_mean", df["t3_score_mean"])
    df["t3_score_headline_label"] = (
        "T3 score (v2 theme matcher)" if "t3_score_v2_mean" in df.columns
        else "T3 score (legacy)"
    )
    return df


def load_difficulty_df(metrics_dir: Path) -> pd.DataFrame:
    """Load by_difficulty.csv, annotate with family/size."""
    path = metrics_dir / "by_difficulty.csv"
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found at {path}.")
    df = pd.read_csv(path)
    df["family"] = df["model"].map(lambda m: MODEL_META.get(m, {}).get("family", "Other"))
    df["difficulty"] = pd.Categorical(df["difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
    return df


# ---------------------------------------------------------------------------
# Plot 1 — Overview rankings
# ---------------------------------------------------------------------------

def plot_overview_rankings(mdf: pd.DataFrame, output_dir: Path) -> None:
    """Three horizontal bar charts: T1 direction acc, T2 CPL, T3 score.

    T2 uses clamped CPL (±1000 cp, Lichess convention). T3 uses the v2 theme
    matcher (handles camelCase Lichess labels). See specs/artefact-fixes.md.
    """

    # For CPL we invert so that "longer bar = better" is consistent across all panels.
    # We display a "moves quality" score = max_cpl - cpl so highest score = best model.
    mdf = mdf.copy()
    cpl_col = "t2_cpl_headline_mean"
    t3_col = "t3_score_headline_mean"
    max_cpl = mdf[cpl_col].max()
    mdf["t2_quality_score"] = max_cpl - mdf[cpl_col]

    metrics = [
        ("t1_direction_correct_mean", "T1: Direction Accuracy\n(who is winning?)", True,
         "Higher = better  ↑", 0.33, "Chance (33%)"),
        ("t2_quality_score",          "T2: Move Quality\n(higher bar = fewer centipawns lost)", True,
         "Higher = better  ↑  (clamped CPL shown on bar)", None, None),
        (t3_col,                      "T3: Explanation Score\n(verbal reasoning quality)", True,
         "Higher = better  ↑", None, None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle("Model Rankings Across All Three Tasks", fontsize=16, fontweight="bold", y=1.01)

    legend_patches = [
        mpatches.Patch(color=FAMILY_COLORS.get(f, "#999"), label=f)
        for f in FAMILY_ORDER if f in mdf["family"].values
    ]

    for ax, (col, title, higher_better, note, refval, reflabel) in zip(axes, metrics):
        needed = list(dict.fromkeys(["display", "family", col, cpl_col]))
        data = mdf[needed].dropna(subset=[col]).copy()
        data = data.sort_values(col, ascending=False)  # best at top in every panel

        colors = [FAMILY_COLORS.get(f, "#999999") for f in data["family"]]

        bars = ax.barh(data["display"], data[col], color=colors, edgecolor="white", linewidth=0.5)

        # Value labels — show the raw interpretable number, not the inverted score
        for bar, (_, row) in zip(bars, data.iterrows()):
            if col == "t2_quality_score":
                raw = row[cpl_col]
                fmt = f"{raw:.0f} CPL"
            elif col == "t1_direction_correct_mean":
                fmt = f"{row[col]:.1%}"
            else:
                fmt = f"{row[col]:.3f}"
            ax.text(
                bar.get_width() + ax.get_xlim()[1] * 0.005,
                bar.get_y() + bar.get_height() / 2,
                fmt, va="center", ha="left", fontsize=7.5, color="#333"
            )

        if refval is not None:
            ax.axvline(refval, color="#e74c3c", linestyle="--", linewidth=1.5, label=reflabel)
            ax.legend(fontsize=8, loc="lower right")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(note, fontsize=9, color="#555")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        if higher_better:
            ax.set_xlim(0, data[col].max() * 1.18)
        else:
            ax.set_xlim(0, data[col].max() * 1.12)

    fig.legend(
        handles=legend_patches, title="Model Family", title_fontsize=9,
        fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04),
        frameon=True, edgecolor="#ccc",
    )

    plt.tight_layout()
    out = output_dir / "overview_rankings.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Plot 2 — Family scaling
# ---------------------------------------------------------------------------

def plot_family_scaling(mdf: pd.DataFrame, output_dir: Path) -> None:
    """Performance vs parameter count, one line per model family."""

    panels = [
        ("t1_direction_correct_mean", "T1 Direction Accuracy",       True,  0.33, "Chance (33%)"),
        ("t2_cpl_headline_mean",      "T2 Move Quality (clamped CPL)", False, None, None),
        ("t3_score_headline_mean",    "T3 Explanation Score",        True,  None, None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Does Scale Help? Performance vs Parameter Count by Model Family",
        fontsize=14, fontweight="bold", y=1.02
    )

    families_present = sorted(mdf["family"].unique(), key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99)

    for ax, (col, ylabel, higher_better, refval, reflabel) in zip(axes, panels):
        for family in families_present:
            fdata = mdf[mdf["family"] == family].sort_values("size_b")
            if fdata.empty:
                continue
            color = FAMILY_COLORS.get(family, "#999")
            ax.plot(
                fdata["size_b"], fdata[col],
                marker="o", linewidth=2, markersize=7,
                color=color, label=family, zorder=3,
            )
            # Label the last point with model size
            for _, row in fdata.iterrows():
                ax.annotate(
                    f"{int(row['size_b'])}B",
                    (row["size_b"], row[col]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, color=color, alpha=0.85
                )

        if refval is not None:
            ax.axhline(refval, color="#e74c3c", linestyle="--", linewidth=1.2,
                       label=reflabel, zorder=2)

        ax.set_xscale("log")
        ax.set_xlabel("Parameter Count (B, log scale)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        note = "↑ higher is better" if higher_better else "↓ lower is better"
        ax.set_title(f"{ylabel}\n({note})", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)

        # Tick only at decade boundaries to avoid crowding
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xticks([2, 4, 7, 11, 14, 26, 32, 47, 70])
        ax.set_xticklabels(["2", "4", "7", "11", "14", "26", "32", "47", "70"], fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Model Family", title_fontsize=9,
        fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12),
        frameon=True, edgecolor="#ccc"
    )

    plt.tight_layout()
    out = output_dir / "family_scaling.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Plot 3 — Difficulty profiles by family
# ---------------------------------------------------------------------------

def plot_difficulty_profiles(ddf: pd.DataFrame, output_dir: Path) -> None:
    """How each metric changes easy→extreme, averaged per family.

    T2 uses clamped CPL where available; T3 uses v2 score; T1 uses
    absolute error excl. mate-truth rows where available. All three changes
    fall through to legacy columns if the metrics file is older.
    """

    cpl_col = "t2_cpl_clamped" if "t2_cpl_clamped" in ddf.columns else "t2_cpl"
    t3_col = "t3_score_v2" if "t3_score_v2" in ddf.columns else "t3_score"
    t1_col = "t1_abs_error_excl_mate" if "t1_abs_error_excl_mate" in ddf.columns else "t1_absolute_error"
    cpl_label = "T2 Move Quality (clamped CPL)" if cpl_col == "t2_cpl_clamped" else "T2 Move Quality (CPL)"
    t1_label = "T1 Eval Error (abs, excl. mate)" if t1_col == "t1_abs_error_excl_mate" else "T1 Eval Error (abs)"

    panels = [
        ("t2_legal", "T2 Legal Move Rate",  True,  "% legal moves produced"),
        (cpl_col,    cpl_label,             False, "centipawn loss vs Stockfish"),
        (t3_col,     "T3 Explanation Score", True, "0–2 score (v2 theme matcher)"),
        (t1_col,     t1_label,              False, "centipawns (lower = better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle(
        "Performance vs Difficulty — Averaged by Model Family",
        fontsize=14, fontweight="bold", y=1.01
    )

    families_present = sorted(ddf["family"].unique(), key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99)

    for ax, (col, title, higher_better, ylabel) in zip(axes, panels):
        for family in families_present:
            fdata = ddf[ddf["family"] == family].groupby("difficulty", observed=True)[col].agg(["mean", "std"]).reindex(DIFFICULTY_ORDER)
            if fdata["mean"].isna().all():
                continue
            color = FAMILY_COLORS.get(family, "#999")
            ax.plot(
                DIFFICULTY_ORDER, fdata["mean"],
                marker="o", linewidth=2, markersize=6,
                color=color, label=family, zorder=3
            )
            ax.fill_between(
                DIFFICULTY_ORDER,
                fdata["mean"] - fdata["std"].fillna(0),
                fdata["mean"] + fdata["std"].fillna(0),
                alpha=0.10, color=color
            )

        arrow = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{title}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Difficulty Tier", fontsize=9)
        ax.set_ylabel(f"{ylabel}  ({arrow})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Model Family", title_fontsize=9,
        fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04),
        frameon=True, edgecolor="#ccc"
    )

    plt.tight_layout()
    out = output_dir / "difficulty_profiles.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Plot 4 — Verbal reasoning vs mechanical play (scatter)
# ---------------------------------------------------------------------------

def plot_verbal_vs_mechanical(mdf: pd.DataFrame, output_dir: Path) -> None:
    """T3 explanation quality (y) vs T2 CPL (x). Shows the decoupling."""

    fig, ax = plt.subplots(figsize=(12, 8))

    families_present = sorted(mdf["family"].unique(), key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99)

    cpl_col = "t2_cpl_headline_mean"
    t3_col = "t3_score_headline_mean"

    for family in families_present:
        fdata = mdf[mdf["family"] == family]
        color = FAMILY_COLORS.get(family, "#999")
        sizes = (fdata["size_b"] * 3).clip(30, 400)
        ax.scatter(
            fdata[cpl_col], fdata[t3_col],
            s=sizes, color=color, alpha=0.85, edgecolors="white",
            linewidth=0.8, label=family, zorder=3
        )
        for _, row in fdata.iterrows():
            ax.annotate(
                short_name(row["model"]),
                (row[cpl_col], row[t3_col]),
                textcoords="offset points", xytext=(6, 3),
                fontsize=7, color=color, alpha=0.9
            )

    # Quadrant dividers at medians
    med_cpl = mdf[cpl_col].median()
    med_t3  = mdf[t3_col].median()
    ax.axvline(med_cpl, color="#aaa", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(med_t3,  color="#aaa", linestyle="--", linewidth=1, alpha=0.7)

    # Quadrant labels
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin + 5,  med_t3 + (ymax - med_t3) * 0.6,  "Good explainer\nBetter play",    fontsize=8, color="#555", style="italic")
    ax.text(med_cpl + 5, med_t3 + (ymax - med_t3) * 0.6, "Good explainer\nWorse play",    fontsize=8, color="#555", style="italic")
    ax.text(xmin + 5,  ymin + (med_t3 - ymin) * 0.1,    "Poor explainer\nBetter play",    fontsize=8, color="#555", style="italic")
    ax.text(med_cpl + 5, ymin + (med_t3 - ymin) * 0.1,  "Poor explainer\nWorse play",     fontsize=8, color="#555", style="italic")

    ax.set_xlabel("T2 Move Quality — Mean CPL  (← lower is better)", fontsize=10)
    ax.set_ylabel("T3 Explanation Score  (higher is better →)", fontsize=10)
    ax.set_title(
        "Verbal Reasoning vs Mechanical Play\n"
        "Bubble size = parameter count · Median lines divide quadrants",
        fontsize=12, fontweight="bold"
    )
    ax.legend(title="Model Family", fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = output_dir / "verbal_vs_mechanical.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Plot 5 — Summary heatmap
# ---------------------------------------------------------------------------

def plot_summary_heatmap(mdf: pd.DataFrame, output_dir: Path) -> None:
    """Normalised model × metric matrix. Green = good, red = bad.

    Uses headline (clamped CPL, v2 T3) columns where available.
    """

    cols = {
        "t1_direction_correct_mean": "T1\nDirection\nAccuracy",
        "t2_legal_headline_mean":             "T2\nLegal Move\nRate",
        "t2_cpl_headline_mean":      "T2 Move\nQuality\n(clamped CPL)",
        "t3_score_headline_mean":    "T3\nExplanation\nScore",
    }
    higher_better = {
        "t1_direction_correct_mean": True,
        "t2_legal_headline_mean":             True,
        "t2_cpl_headline_mean":      False,
        "t3_score_headline_mean":    True,
    }

    data = mdf[["model", "family", "size_b"] + list(cols.keys())].dropna().copy()

    # Normalise each column 0–1 so heatmap colours are comparable
    norm = data.copy()
    for col, hb in higher_better.items():
        col_min, col_max = data[col].min(), data[col].max()
        if col_max > col_min:
            norm[col] = (data[col] - col_min) / (col_max - col_min)
            if not hb:
                norm[col] = 1 - norm[col]
        else:
            norm[col] = 0.5

    # Compute overall rank (mean of normalised scores), sort
    norm["overall"] = norm[list(cols.keys())].mean(axis=1)
    data["overall"] = norm["overall"]
    data = data.sort_values("overall", ascending=False).reset_index(drop=True)
    norm = norm.loc[data.index].reset_index(drop=True)

    matrix = norm[list(cols.keys())].values
    raw    = data[list(cols.keys())].values

    n_models = len(data)
    fig, ax = plt.subplots(figsize=(8, max(9, n_models * 0.38)))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(list(cols.values()), fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(
        [f"{row['model']}  [{row['family']}  {int(row['size_b'])}B]"
         for _, row in data.iterrows()],
        fontsize=8
    )

    # Annotate cells with raw values
    fmt_fns = {
        "t1_direction_correct_mean": lambda v: f"{v:.1%}",
        "t2_legal_headline_mean":             lambda v: f"{v:.1%}",
        "t2_cpl_headline_mean":      lambda v: f"{v:.0f}",
        "t3_score_headline_mean":    lambda v: f"{v:.3f}",
    }
    for i in range(n_models):
        for j, col in enumerate(cols.keys()):
            val = raw[i, j]
            txt = fmt_fns[col](val)
            cell_brightness = matrix[i, j]
            text_color = "black" if 0.25 < cell_brightness < 0.75 else ("white" if cell_brightness < 0.25 else "black")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                 label="Normalised score  (green = better, red = worse)")

    ax.set_title(
        "All Models × All Metrics\n(normalised; sorted by overall rank, best at top)",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")

    plt.tight_layout()
    out = output_dir / "summary_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Plot 6 — Per-family deep-dives (one plot per family)
# ---------------------------------------------------------------------------

# Families with 3+ models get their own plot.
# Smaller families are grouped into "Others".
_SOLO_FAMILIES = {"LLaMA 3", "Gemma 3", "Gemma 4", "Qwen 2.5", "DeepSeek-R1", "Mistral"}

# Within each family plot, use these tab colours for individual models
_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
]


def _family_filename(name: str) -> str:
    return "family_" + name.lower().replace(" ", "_").replace("-", "_").replace(".", "") + ".png"


def plot_per_family(ddf: pd.DataFrame, mdf: pd.DataFrame, output_dir: Path) -> None:
    """One 2×2 subplot per model family: each metric vs difficulty, one line per model."""

    cpl_col = "t2_cpl_clamped" if "t2_cpl_clamped" in ddf.columns else "t2_cpl"
    t3_col = "t3_score_v2" if "t3_score_v2" in ddf.columns else "t3_score"
    t1_col = "t1_abs_error_excl_mate" if "t1_abs_error_excl_mate" in ddf.columns else "t1_absolute_error"
    cpl_label = "T2: Move Quality (clamped CPL)" if cpl_col == "t2_cpl_clamped" else "T2: Move Quality (CPL)"
    t1_label = "T1: Eval Error (abs, excl. mate)" if t1_col == "t1_abs_error_excl_mate" else "T1: Eval Error (absolute)"
    t3_label = "T3: Explanation Score (v2 matcher)" if t3_col == "t3_score_v2" else "T3: Explanation Score"

    panels = [
        ("t2_legal", "T2: Legal Move Rate",       True,  "fraction of legal moves"),
        (cpl_col,    cpl_label,                   False, "centipawn loss vs Stockfish"),
        (t3_col,     t3_label,                    True,  "score"),
        (t1_col,     t1_label,                    False, "centipawns (lower = better)"),
    ]

    ddf = ddf.copy()
    ddf["plot_family"] = ddf["family"].apply(
        lambda f: f if f in _SOLO_FAMILIES else "Others"
    )
    mdf = mdf.copy()
    mdf["plot_family"] = mdf["family"].apply(
        lambda f: f if f in _SOLO_FAMILIES else "Others"
    )

    for family_name in sorted(ddf["plot_family"].unique()):
        fam_diff = ddf[ddf["plot_family"] == family_name]
        fam_model = mdf[mdf["plot_family"] == family_name]
        models = sorted(fam_diff["model"].unique(), key=lambda m: MODEL_META.get(m, {}).get("size", 0))

        if not models:
            continue

        model_colors = {m: _TAB10[i % len(_TAB10)] for i, m in enumerate(models)}

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
        axes = axes.flatten()

        subtitle = "  ·  ".join(
            f"{short_name(m)}  ({MODEL_META.get(m, {}).get('size', '?')}B)"
            for m in models
        )
        fig.suptitle(
            f"{family_name} models\n{subtitle}",
            fontsize=12, fontweight="bold", y=1.03,
        )

        for ax, (col, title, higher_better, ylabel) in zip(axes, panels):
            for model in models:
                row_data = fam_diff[fam_diff["model"] == model]
                ys = [
                    row_data.loc[row_data["difficulty"] == d, col].mean()
                    if d in row_data["difficulty"].values else float("nan")
                    for d in DIFFICULTY_ORDER
                ]
                size_b = MODEL_META.get(model, {}).get("size", 0)
                ax.plot(
                    DIFFICULTY_ORDER, ys,
                    marker="o", linewidth=2.2, markersize=7,
                    color=model_colors[model],
                    label=f"{short_name(model)} ({size_b}B)",
                )

            arrow = "↑ better" if higher_better else "↓ better"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Difficulty tier", fontsize=9)
            ax.set_ylabel(f"{ylabel}  ({arrow})", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8, loc="best")

        # Add a small per-model summary bar inset on the T3 panel (axes[2]).
        # Uses the headline T3 column so the inset shares the v2 matcher when
        # available.
        ax_t3 = axes[2]
        ax_inset = ax_t3.inset_axes([0.65, 0.04, 0.33, 0.40])
        t3_inset_col = (
            "t3_score_headline_mean" if "t3_score_headline_mean" in fam_model.columns
            else "t3_score_mean"
        )
        vals  = [fam_model.loc[fam_model["model"] == m, t3_inset_col].values for m in models]
        vals  = [float(v[0]) if len(v) else float("nan") for v in vals]
        names = [short_name(m) for m in models]
        ys    = list(range(len(models)))
        ax_inset.barh(ys, vals,
                      color=[model_colors[m] for m in models],
                      edgecolor="white", linewidth=0.4)
        ax_inset.set_yticks(ys)
        ax_inset.set_yticklabels(names, fontsize=6)
        valid_vals = [v for v in vals if not np.isnan(v)]
        if valid_vals:
            ax_inset.set_xlim(0, max(valid_vals) * 1.25)
        ax_inset.set_title("Overall T3", fontsize=6.5, pad=2)
        ax_inset.tick_params(axis="x", labelsize=6)
        ax_inset.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        out = output_dir / _family_filename(family_name)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {out.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualisation plots")
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--metrics-dir", default=None,
                        help="Directory containing metrics CSVs (overrides config)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write plots (overrides config)")
    parser.add_argument("--save-metrics", action="store_true",
                        help="Re-compute and save metrics CSVs before plotting")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    import logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    config = load_config(args.config)
    paths  = config.get("paths", {})

    metrics_dir = Path(args.metrics_dir or paths.get("metrics_dir", "results/metrics"))
    output_dir  = Path(args.output_dir  or paths.get("plots_dir",   "results/plots"))
    ensure_dir(output_dir)

    # Optionally re-compute metrics CSVs from the raw JSONL first
    if args.save_metrics:
        print("Re-computing metrics from evaluations.jsonl …")
        from src.metrics import load_results_df, save_metrics
        results_file = paths.get("results_file", "results/evaluations.jsonl")
        df_raw = load_results_df(results_file)
        if df_raw.empty:
            print(f"ERROR: no results in {results_file}")
            sys.exit(1)
        save_metrics(df_raw, str(metrics_dir))
        print(f"  Metrics saved to {metrics_dir}\n")

    # Load pre-computed metrics
    print("Loading metrics CSVs …")
    try:
        mdf = load_model_df(metrics_dir)
        ddf = load_difficulty_df(metrics_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Re-run with --save-metrics to generate them.")
        sys.exit(1)

    print(f"  {len(mdf)} models × {len(ddf)} model-difficulty rows\n")

    print("Generating plots …")
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.titlepad": 10,
    })

    print("Overview charts …")
    plot_overview_rankings(mdf, output_dir)
    plot_family_scaling(mdf, output_dir)
    plot_difficulty_profiles(ddf, output_dir)
    plot_verbal_vs_mechanical(mdf, output_dir)
    plot_summary_heatmap(mdf, output_dir)

    print("\nPer-family deep-dives …")
    plot_per_family(ddf, mdf, output_dir)

    print(f"\nAll plots written to {output_dir}/")


if __name__ == "__main__":
    main()
