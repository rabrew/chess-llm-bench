"""Pre-flight check for the fine-tuned T1–T3 run (specs/finetuned-benchmark-run.md).

Confirms the fine-tuned models will be asked exactly what their base models were
asked, in exactly the same way. Exits 1 on any problem.

Usage:
    python scripts/check_ft_run.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fine-tuning" / "scripts"))

from src.utils import load_config, parse_model_info  # noqa: E402

FT_SUFFIX = "-board-ft"
ALLOWED_DIFFERENCES = {"models", "paths"}
TIERS = ("easy", "medium", "hard", "extreme")


def config_differences(base: dict, ft: dict) -> list[str]:
    """Top-level keys, other than models and paths, whose values differ."""
    keys = (set(base) | set(ft)) - ALLOWED_DIFFERENCES
    return sorted(key for key in keys if base.get(key) != ft.get(key))


def tag_problems(ft_tags: list[str], base_tags: list[str]) -> list[str]:
    problems = []
    for tag in ft_tags:
        if not tag.endswith(FT_SUFFIX):
            problems.append(f"{tag}: not a {FT_SUFFIX} tag")
            continue
        try:
            parse_model_info(tag)
        except ValueError as exc:
            problems.append(f"{tag}: {exc}")
            continue
        base = tag.removesuffix(FT_SUFFIX)
        if base not in base_tags:
            problems.append(f"{tag}: base {base} is not in the base config")
    return problems


def fen_set_problems(label: str, sampled: set[str], recorded: set[str]) -> list[str]:
    """Fail when the base run used positions this sample does not contain (different sampling)."""
    if not recorded:
        return [f"{label}: no base records to compare with"]
    not_sampled = recorded - sampled
    if not_sampled:
        return [f"{label}: {len(not_sampled)} base-run positions are not in this sample"]
    return []


def coverage_gaps(label: str, sampled: set[str], recorded: set[str]) -> list[str]:
    """Positions in the sample the base run never completed. Compare on the shared positions only."""
    gap = sampled - recorded
    return [f"{label}: base run has no record for {len(gap)} of {len(sampled)} positions"] if gap else []


def foreign_models(results_path: Path) -> set[str]:
    """Models in the fine-tuned results file that are not fine-tuned."""
    if not results_path.exists():
        return set()
    found = set()
    with open(results_path) as handle:
        for line in handle:
            if line.strip():
                found.add(json.loads(line)["model"])
    return {model for model in found if not model.endswith(FT_SUFFIX)}


def recorded_fens(results_path: Path, models: set[str]) -> dict[tuple[str, str], set[str]]:
    """(base model, tier) -> FENs that model was benchmarked on."""
    fens: dict[tuple[str, str], set[str]] = defaultdict(set)
    with open(results_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["model"] in models and record.get("job_type", "standard") == "standard":
                fens[(record["model"], record["difficulty"])].add(record["fen"])
    return fens


def position_problems(base_config: dict, bases: set[str]) -> tuple[list[str], list[str]]:
    from src.data_loader import DataLoader

    bench = base_config["benchmark"]
    loader = DataLoader(base_config["paths"]["data_dir"])
    recorded = recorded_fens(Path(base_config["paths"]["results_file"]), bases)
    problems, warnings = [], []
    for tier in TIERS:
        sample = loader.sample(loader.load_tier(tier), count=bench["max_positions_per_tier"], seed=bench["random_seed"])
        sampled = {position["fen"] for position in sample}
        for base in sorted(bases):
            label, base_fens = f"{base}/{tier}", recorded.get((base, tier), set())
            problems += fen_set_problems(label, sampled, base_fens)
            warnings += coverage_gaps(label, sampled, base_fens)
    return problems, warnings


def modelfile_problems(ft_tags: list[str]) -> list[str]:
    from import_finetuned_to_ollama import base_modelfile, modelfile_settings

    problems = []
    for tag in ft_tags:
        base = tag.removesuffix(FT_SUFFIX)
        if modelfile_settings(base_modelfile(tag)) != modelfile_settings(base_modelfile(base)):
            problems.append(f"{tag}: Modelfile settings differ from {base} (re-import it)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-config", default="config/config.yaml")
    parser.add_argument("--ft-config", default="config/config_finetuned.yaml")
    args = parser.parse_args()

    base_config, ft_config = load_config(args.base_config), load_config(args.ft_config)
    ft_tags = ft_config["models"]
    bases = {tag.removesuffix(FT_SUFFIX) for tag in ft_tags}

    problems = [f"config key differs: {key}" for key in config_differences(base_config, ft_config)]
    problems += tag_problems(ft_tags, base_config["models"])
    problems += [f"non-fine-tuned rows in results: {m}" for m in sorted(foreign_models(Path(ft_config["paths"]["results_file"])))]
    problems += modelfile_problems(ft_tags)
    position_errors, warnings = position_problems(base_config, bases)
    problems += position_errors

    for warning in warnings:
        print(f"WARN  {warning}")
    for problem in problems:
        print(f"FAIL  {problem}")
    print("Pre-flight check:", "FAILED" if problems else f"OK ({len(ft_tags)} models)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
