"""Render the write-up's tables from the metrics files (specs/findings-tables.md).

Only text between <!-- table:NAME --> and <!-- /table:NAME --> markers is replaced.

Usage (from the repo root):
    python scripts/render_findings.py            # rewrite the tables in FINDINGS and SIMPLE_SUMMARY
    python scripts/render_findings.py --check    # exit 1 if any table is stale
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import COMBINED_PROMPT_FORMATS, CPL_BUCKETS  # noqa: E402

BUCKET_KEYS = [key for key, _, _ in CPL_BUCKETS]
BUCKET_LABELS = {
    "perfect": "Essentially perfect (< 25)",
    "strong": "Strong (25–100)",
    "decent": "Decent (100–500)",
    "very_weak": "Very weak (500–1000)",
    "blunder": "Blunder (≥ 1000)",
}
NOT_AVAILABLE = "_Table not available yet: its input file has not been generated._"
DASH = "—"
DEFAULT_DOCS = ("docs/FINDINGS.md", "docs/SIMPLE_SUMMARY.md")
BLOCK = re.compile(r"<!-- table:(?P<name>[\w-]+) -->\n.*?<!-- /table:(?P=name) -->", re.S)
OPEN = re.compile(r"<!-- table:(?P<name>[\w-]+) -->")


@dataclass
class Sources:
    metrics: Path
    rerun_metrics: Path
    comparison: Path
    board_test: Path


# ---------- formatting ----------

def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def pct(value) -> str:
    return DASH if _missing(value) else f"{100 * value:.1f}%"


def num(value, places: int = 2) -> str:
    return DASH if _missing(value) else f"{value:.{places}f}"


def pct_ci(rate, low, high) -> str:
    if _missing(rate):
        return DASH
    return f"{pct(rate)} ({100 * low:.1f}–{100 * high:.1f}%)"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


# ---------- tables ----------

def _first_answer_pooled(rerun_dir: Path) -> str:
    table = _read_csv(rerun_dir / "t2_move_source.csv")
    if table is None:
        return "not available (needs the base re-run)"
    combined = table[table["prompt_format"].isin(COMBINED_PROMPT_FORMATS)]
    rate = (combined["first_attempt_legal_rate"] * combined["n"]).sum() / combined["n"].sum()
    return f"{pct(rate)} (5 re-run models, combined prompts)"


def headline_table(summary: dict, rerun_dir: Path) -> str:
    m = summary["overall_metrics"]
    blunder, best, setup, side = m["t2_blunder_rate"], m["t2_best_move_vs_random"], m["puzzle_meta"], m["t3_side_mark"]
    rows = [
        ["Records / models / positions", f"{summary['total_jobs']:,} / {summary['models_tested']} / {summary['positions_tested']:,}"],
        ["T1: Spearman ρ, model eval vs engine (pooled)", num(m["t1_headline"]["value"])],
        ["T1: eval error ÷ error of always answering 0", num(m["t1_zero_baseline"]["mae_ratio"])],
        ["T2: legal move, final (after re-asking with the legal-move list)", pct(m["t2_legal_rate"])],
        ["T2: legal move, model's own first answer", _first_answer_pooled(rerun_dir)],
        ["T2: blunder rate (clamped CPL ≥ 1000), 95% CI", pct_ci(blunder["rate"], blunder["ci_low"], blunder["ci_high"])],
        ["T2: plays the engine's best move", f"{pct(best['best_move_rate'])} vs {pct(best['random_best_move_rate'])} for a random legal move ({num(best['ratio_vs_random'])}×)"],
        ["T2: plays the human setup move (the move that led into the puzzle)", pct(setup["setup_move_rate"]) if setup else DASH],
        ["T3: names the better side", f"{pct(side['accuracy'])} vs {pct(side['majority_share'])} for always naming the usual side (κ = {num(side['cohen_kappa'])})"],
        ["T3: names a tactical theme of the puzzle", pct(setup["tactical_theme_rate"]) if setup else DASH],
    ]
    return _table(["Measure", "Value"], rows)


def models_table(metrics_dir: Path) -> str:
    frames = {
        "rho": pd.read_csv(metrics_dir / "t1_rank_correlation.csv")[["model", "spearman_rho"]],
        "zero": pd.read_csv(metrics_dir / "t1_error_vs_zero.csv")[["model", "mae_ratio"]],
        "blunder": pd.read_csv(metrics_dir / "t2_blunder_rate.csv")[["model", "blunder_rate", "ci_low", "ci_high"]],
        "best": pd.read_csv(metrics_dir / "t2_best_move_vs_random.csv")[["model", "ratio_vs_random"]],
        "setup": pd.read_csv(metrics_dir / "puzzle_meta_metrics.csv")[["model", "setup_move_rate"]],
        "side": pd.read_csv(metrics_dir / "t3_side_mark_agreement.csv")[["model", "cohen_kappa"]],
        "by_model": pd.read_csv(metrics_dir / "by_model.csv")[["model", "t2_legal_attempted_mean", "t3_score_v2_mean"]],
    }
    merged = frames["blunder"]
    for name, frame in frames.items():
        if name != "blunder":
            merged = merged.merge(frame, on="model", how="left")
    merged = merged.sort_values(["blunder_rate", "model"])
    rows = [
        [r.model, num(r.spearman_rho), num(r.mae_ratio), pct(r.t2_legal_attempted_mean),
         pct_ci(r.blunder_rate, r.ci_low, r.ci_high), f"{num(r.ratio_vs_random)}×",
         pct(r.setup_move_rate), num(r.cohen_kappa), num(r.t3_score_v2_mean)]
        for r in merged.itertuples()
    ]
    header = ["Model", "T1 ρ", "T1 error ÷ zero", "Legal (final)", "Blunder rate (95% CI)",
              "Best move ×random", "Setup move", "Side κ", "T3 (0–2)"]
    return _table(header, rows)


def buckets_table(metrics_dir: Path) -> str:
    buckets = pd.read_csv(metrics_dir / "cpl_buckets.csv")
    total = buckets["n"].sum()
    rows = [[BUCKET_LABELS[key], pct(buckets[f"{key}_count"].sum() / total)] for key in BUCKET_KEYS]
    return _table(["Clamped CPL bucket", "Share of legal moves"], rows)


def first_answer_table(rerun_dir: Path) -> str:
    table = _read_csv(rerun_dir / "t2_move_source.csv")
    if table is None:
        return NOT_AVAILABLE
    combined = table[table["prompt_format"].isin(COMBINED_PROMPT_FORMATS)].copy()
    rows = []
    for model, group in combined.groupby("model", sort=True):
        weights = group["n"] / group["n"].sum()
        rows.append([model, pct((group["first_attempt_legal_rate"] * weights).sum()),
                     pct((group["final_legal_rate"] * weights).sum())])
    return _table(["Model", "First answer legal", "Final (after re-asking with the list)"], rows)


def family_table(metrics_dir: Path) -> str:
    family = pd.read_csv(metrics_dir / "by_model_family.csv").sort_values(["model_family", "model_size_b"])
    rows = [
        [r.model_family, f"{r.model_size_b:g}B", num(r.t1_spearman_rho), pct(r.t2_blunder_rate),
         pct(r.t2_legal_rate), num(r.t3_score_v2_mean)]
        for r in family.itertuples()
    ]
    return _table(["Family", "Size", "T1 ρ", "Blunder rate", "Legal (final)", "T3 (0–2)"], rows)


def _arrow(before, after, fmt) -> str:
    return f"{fmt(before)} → {fmt(after)}"


def finetune_table(comparison_path: Path, board_path: Path) -> str:
    comparison, board = _read_csv(comparison_path), _read_csv(board_path)
    if comparison is None or board is None:
        return NOT_AVAILABLE
    recall = board.set_index("model")
    rows = []
    for r in comparison.sort_values("base_model").itertuples():
        base_board = recall.loc[r.base_model] if r.base_model in recall.index else None
        ft_board = recall.loc[r.ft_model] if r.ft_model in recall.index else None
        board_cell = DASH if base_board is None or ft_board is None else _arrow(base_board.mean_recall, ft_board.mean_recall, pct)
        perfect_cell = DASH if base_board is None or ft_board is None else _arrow(base_board.perfect_rate, ft_board.perfect_rate, pct)
        rows.append([
            r.base_model, board_cell, perfect_cell,
            _arrow(r.base_t1_spearman, r.ft_t1_spearman, num),
            _arrow(r.base_t2_legal_rate, r.ft_t2_legal_rate, pct),
            _arrow(r.base_t2_blunder_rate, r.ft_t2_blunder_rate, pct),
            _arrow(r.base_t3_mean, r.ft_t3_mean, num),
        ])
    header = ["Model", "Board recall", "Perfect boards", "T1 ρ", "Legal (final)", "Blunder rate", "T3 (0–2)"]
    return _table(header, rows)


DIFFICULTY_TIERS = ("easy", "medium", "hard", "extreme")


def _verdict(test: dict) -> str:
    return "supported" if test.get("supported") else "not supported"


def difficulty_table(summary: dict) -> str:
    tests = summary.get("hypothesis_tests", {})
    if not all(key in tests for key in ("H1", "H2", "H3")):
        return NOT_AVAILABLE
    h1, h2, h3 = (tests[key]["values"] for key in ("H1", "H2", "H3"))
    rows = [[tier, num(h1.get(tier)), num(h2.get(tier), 0), num(h3.get(tier))] for tier in DIFFICULTY_TIERS]
    rows.append(["Expected trend holds?", _verdict(tests["H1"]), _verdict(tests["H2"]), _verdict(tests["H3"])])
    return _table(["Tier", "T1 relative error (H1)", "Clamped CPL (H2)", "T3 v2 (H3)"], rows)


def formats_table(metrics_dir: Path) -> str:
    table = _read_csv(metrics_dir / "by_prompt_format.csv")
    if table is None:
        return NOT_AVAILABLE
    rows = [
        [r.prompt_format, num(r.t1_spearman_rho), pct(r.t2_legal_rate), pct(r.t2_blunder_rate), num(r.t3_score_v2_mean)]
        for r in table.itertuples()
    ]
    return _table(["Prompt format", "T1 ρ", "Legal (final)", "Blunder rate", "T3 (0–2)"], rows)


def render_all(sources: Sources) -> dict[str, str]:
    summary = json.loads((sources.metrics / "summary.json").read_text())
    return {
        "headline": headline_table(summary, sources.rerun_metrics),
        "models": models_table(sources.metrics),
        "buckets": buckets_table(sources.metrics),
        "first_answer": first_answer_table(sources.rerun_metrics),
        "family": family_table(sources.metrics),
        "finetune": finetune_table(sources.comparison, sources.board_test),
        "difficulty": difficulty_table(summary),
        "formats": formats_table(sources.metrics),
    }


# ---------- document blocks ----------

def replace_blocks(text: str, tables: dict[str, str]) -> str:
    """Replace each marked block with its rendered table.

    Raises:
        ValueError: for an unknown table name or an opening marker without a close.
    """
    closed = {match.group("name") for match in BLOCK.finditer(text)}
    for match in OPEN.finditer(text):
        name = match.group("name")
        if name not in closed:
            raise ValueError(f"Unterminated table marker: {name}")
        if name not in tables:
            raise ValueError(f"Unknown table in document: {name}")
    return BLOCK.sub(lambda m: f"<!-- table:{m.group('name')} -->\n{tables[m.group('name')]}\n<!-- /table:{m.group('name')} -->", text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render write-up tables from the metrics files.")
    parser.add_argument("--metrics", default="results/metrics")
    parser.add_argument("--rerun-metrics", default="results/finetuning_experiment/base_rerun/metrics")
    parser.add_argument("--comparison", default="results/finetuning_experiment/comparison.csv")
    parser.add_argument("--board-test", default="results/finetuning_experiment/board_t0_summary.csv")
    parser.add_argument("--doc", action="append", help=f"Document to update (repeatable; default {', '.join(DEFAULT_DOCS)})")
    parser.add_argument("--check", action="store_true", help="Exit 1 if any table is stale; write nothing")
    args = parser.parse_args()

    sources = Sources(Path(args.metrics), Path(args.rerun_metrics), Path(args.comparison), Path(args.board_test))
    if not (sources.metrics / "summary.json").exists():
        print(f"ERROR: {sources.metrics / 'summary.json'} not found", file=sys.stderr)
        sys.exit(1)
    tables = render_all(sources)
    stale = []
    for doc in [Path(d) for d in (args.doc or DEFAULT_DOCS)]:
        text = doc.read_text()
        rendered = replace_blocks(text, tables)
        if rendered == text:
            continue
        if args.check:
            stale.append(str(doc))
        else:
            doc.write_text(rendered)
            print(f"Updated tables in {doc}")
    if stale:
        print(f"Stale tables in: {', '.join(stale)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
