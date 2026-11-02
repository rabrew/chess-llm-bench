"""Auditable diagnostics over stored records; no model or engine calls."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

THRESHOLDS = (0, 50, 100, 200)
CLASSES = ("W", "B", "E")
MOVE_FORMATS = ("cot", "fen_only", "pgn+fen", "move_only")
BUCKET_EDGES = (0, 50, 100, 300, 1000, np.inf)
BUCKET_NAMES = ("0-49", "50-99", "100-299", "300-999", ">=1000")


def _classes(values, threshold):
    return np.where(values > threshold, "W", np.where(values < -threshold, "B", "E"))


def direction_baselines(frame):
    """Reference priors on unique positions, with sample size and denominator."""
    if "t1_stockfish_eval" not in frame:
        return pd.DataFrame()
    reference = frame.dropna(subset=["t1_stockfish_eval"])
    identity = next((key for key in ("position_id", "fen") if key in reference), None)
    if identity:
        conflicts = reference.groupby(identity).t1_stockfish_eval.nunique()
        if (conflicts > 1).any():
            raise ValueError("Conflicting reference evaluations for the same position")
        reference = reference.drop_duplicates(identity)
    rows = []
    for threshold in THRESHOLDS:
        labels = _classes(reference.t1_stockfish_eval, threshold)
        priors = {label: float(np.mean(labels == label)) if len(labels) else np.nan for label in CLASSES}
        majority = max(priors, key=priors.get) if len(labels) else None
        rows.append(dict(threshold=threshold, n=len(labels), majority_class=majority,
                         majority_baseline_acc=priors.get(majority),
                         **{f"pct_{label}": prior for label, prior in priors.items()}))
    return pd.DataFrame(rows)


def _agreement(predicted, truth):
    accuracy = float(np.mean(predicted == truth))
    expected = sum(np.mean(predicted == label) * np.mean(truth == label) for label in CLASSES)
    recalls = [np.mean(predicted[truth == label] == label) for label in CLASSES if (truth == label).any()]
    return dict(accuracy=accuracy, kappa=(accuracy - expected) / (1 - expected) if expected < 1 else np.nan,
                balanced_accuracy=float(np.mean(recalls)))


def _direction_rows(model, cohort, subset):
    paired = subset.dropna(subset=["t1_model_eval", "t1_stockfish_eval"])
    finite = np.isfinite(paired.t1_model_eval) & np.isfinite(paired.t1_stockfish_eval)
    paired = paired.loc[finite]
    ranks = paired.loc[paired.t1_stockfish_eval.abs() < 9000]
    rho = np.nan
    if len(ranks) > 1 and ranks.t1_model_eval.nunique() > 1 and ranks.t1_stockfish_eval.nunique() > 1:
        rho = float(spearmanr(ranks.t1_model_eval, ranks.t1_stockfish_eval).statistic)
    for threshold in THRESHOLDS:
        agreement = _agreement(_classes(paired.t1_model_eval, threshold), _classes(paired.t1_stockfish_eval, threshold)) if len(paired) else dict(accuracy=np.nan, kappa=np.nan, balanced_accuracy=np.nan)
        labels = _classes(paired.t1_stockfish_eval, threshold)
        majority = max((np.mean(labels == label) for label in CLASSES), default=np.nan) if len(labels) else np.nan
        yield dict(model=model, cohort=cohort, threshold=threshold, n=len(paired), matched_majority_baseline=majority,
                   spearman_rho=rho, spearman_n=len(ranks), **agreement)


def direction_diagnostics(frame):
    if not {"model", "t1_model_eval", "t1_stockfish_eval"}.issubset(frame.columns):
        return pd.DataFrame()
    rows = []
    for model, subset in frame.groupby("model"):
        rows.extend(_direction_rows(model, "all", subset))
        if "fen" in subset:
            turns = subset.fen.map(lambda fen: fen.split()[1] if isinstance(fen, str) and len(fen.split()) > 1 else None)
            for side in ("w", "b"):
                rows.extend(_direction_rows(model, side, subset.loc[turns == side]))
    return pd.DataFrame(rows)


def classify_regimes(frame):
    """Classify only explicit metadata or one-record evidence; never infer rollout."""
    regimes = pd.Series("unknown", index=frame.index)
    if {"t2_move", "t2_legal"}.issubset(frame.columns):
        evidence = frame.t2_move.notna() & frame.t2_legal.eq(False)
        regimes.loc[evidence] = "pre_rescue_evidence"
    if "t2_rescue_enabled" in frame:
        regimes.loc[frame.t2_rescue_enabled.eq(True)] = "harness_assisted"
        regimes.loc[frame.t2_rescue_enabled.eq(False)] = "single_shot"
        if "prompt_format" in frame:
            instrumented = frame.t2_rescue_enabled.notna()
            regimes.loc[instrumented & frame.prompt_format.eq("move_only")] = "legal_list_assisted"
    if "prompt_format" in frame:
        regimes.loc[~frame.prompt_format.isin(MOVE_FORMATS)] = "not_applicable"
    return regimes


def regime_split(frame):
    if "model" not in frame:
        return pd.DataFrame()
    classified = frame.copy()
    classified["regime"] = classify_regimes(frame)
    if "prompt_format" not in classified:
        classified["prompt_format"] = "unknown"
    classified["move_prompt"] = classified.prompt_format.isin(MOVE_FORMATS)
    legal = classified.get("t2_legal", pd.Series(np.nan, index=classified.index))
    classified["legality"] = legal.where(classified.move_prompt)
    return classified.groupby(["regime", "model", "prompt_format"], dropna=False).agg(
        n=("model", "size"), move_prompt_n=("move_prompt", "sum"),
        measured_n=("legality", "count"), legality=("legality", "mean")).reset_index()


def _cluster_intervals(subset, columns, rng, replicates):
    identity = next((key for key in ("position_id", "fen") if key in subset), None)
    groups = subset.groupby(identity, dropna=False) if identity else subset.groupby(np.arange(len(subset)))
    sums = groups[columns].sum().to_numpy(float)
    counts = groups[columns].count().to_numpy(float)
    samples = []
    for _ in range(replicates):
        indices = rng.integers(0, len(sums), size=len(sums))
        total = counts[indices].sum(axis=0)
        samples.append(np.divide(sums[indices].sum(axis=0), total, out=np.full(len(columns), np.nan), where=total > 0))
    return np.nanquantile(samples, [.025, .975], axis=0)


def cpl_diagnostics(frame, replicates=500, seed=42):
    """Position-cluster bootstrap within each model; pairwise CI overlap is descriptive."""
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    if "t2_cpl_clamped" not in frame:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    quality, buckets = [], []
    for model, subset in frame.groupby("model"):
        subset = subset.loc[subset.t2_cpl_clamped.notna()].copy()
        if "t2_legal" in subset:
            subset = subset.loc[subset.t2_legal.eq(True)]
        if subset.empty:
            continue
        if (subset.t2_cpl_clamped < 0).any():
            raise ValueError("CPL must be nonnegative")
        subset["blunder_rate"] = (subset.t2_cpl_clamped >= 1000).astype(float)
        columns = [column for column in ("t2_cpl_clamped", "t2_wp_loss", "blunder_rate") if column in subset]
        bounds = _cluster_intervals(subset, columns, rng, replicates)
        statuses = subset.get("t2_endpoint_status", pd.Series("unclassified", index=subset.index))
        measured_n = int(statuses.eq("recorded").sum())
        convention_n = int(statuses.eq("best_move_zero_convention").sum())
        row = dict(model=model, n=len(subset), measured_endpoint_n=measured_n,
                   convention_zero_n=convention_n, unclassified_quality_n=len(subset)-measured_n-convention_n,
                   seed=seed, bootstrap_replicates=replicates,
                   interval_method="position_cluster_percentile", rank_tied=False)
        for offset, column in enumerate(columns):
            row[column] = float(subset[column].mean())
            row[column + "_median"] = float(subset[column].median())
            row[column + "_ci_low"], row[column + "_ci_high"] = bounds[:, offset]
        quality.append(row)
        labels = pd.cut(subset.t2_cpl_clamped, BUCKET_EDGES, right=False, labels=BUCKET_NAMES)
        for bucket in BUCKET_NAMES:
            count = int((labels == bucket).sum())
            buckets.append(dict(model=model, bucket=bucket, n=count, fraction=count / len(subset)))
    for row in quality:
        overlaps = [other["model"] for other in quality if other is not row and row["blunder_rate_ci_low"] <= other["blunder_rate_ci_high"] and other["blunder_rate_ci_low"] <= row["blunder_rate_ci_high"]]
        row["rank_tied"] = bool(overlaps)
        row["overlapping_ci_models"] = json.dumps(overlaps)
    quality_columns = ["model", "n", "measured_endpoint_n", "convention_zero_n", "unclassified_quality_n", "seed", "bootstrap_replicates", "interval_method", "rank_tied", "overlapping_ci_models"]
    for column in ("t2_cpl_clamped", "t2_wp_loss", "blunder_rate"):
        quality_columns.extend([column, column + "_median", column + "_ci_low", column + "_ci_high"])
    return pd.DataFrame(quality, columns=quality_columns), pd.DataFrame(buckets, columns=["model", "bucket", "n", "fraction"])


def analysis_provenance(collection=None):
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
                                  check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), analysis_git_sha=revision,
                collection=collection or {"status": "unverified", "t1_truth_engine": "unknown", "t2_truth_engine": "unknown"},
                legacy_field_note="stockfish_* field names do not establish engine provenance",
                regime_note="No historical rollout timestamp inferred from last illegal move")


def publication_columns(frame):
    """Label superseded columns at export without breaking in-memory consumers."""
    legacy = ("t1_absolute_error", "t1_direction_correct", "t2_cpl", "t2_legal",
              "t2_legal_attempted", "t3_score", "t3_p2_theme_correct", "hallucination_rate")
    corrected = ("clamped", "v2", "correct_t", "move_prompt")
    names = {column: column + "_legacy" for column in frame
             if any(column == old or column.startswith(old + "_") for old in legacy)
             and not any(marker in column for marker in corrected)}
    return frame.rename(columns=names)


def json_safe(value):
    """Produce interoperable JSON, with missing numbers explicitly null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
