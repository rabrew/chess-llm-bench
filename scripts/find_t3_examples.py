"""Find plain-English phrasing the models use for tactical themes (T3-2).

For each candidate cue in IDEA_CUES, reports how many unmarked explanations of
that theme contain it (gain) and how often it appears in explanations of other
tactical themes (leak, a false-positive proxy), plus a hand-reading sample.
Spec: specs/t3-example-finder.md.

Usage (from the repo root):
    python scripts/find_t3_examples.py --results results/evaluations.jsonl \
        --results results/finetuning_experiment/evaluations_base_rerun.jsonl
"""

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import NON_TACTICAL_THEMES, phrase_mentioned, theme_mentioned  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("chess_llm_bench")

SEED = 42
DEFAULT_OUT = "results/t3_examples.md"
MIN_GAIN = 20
MAX_LEAK_RATE = 0.02
EXAMPLES_PER_CUE = 3
SAMPLE_PER_THEME = 15
SAMPLE_TEXT_CHARS = 400

# Candidate plain-English descriptions (hand-written 2026-10-20). Candidates
# only: T3-3 keeps a cue only when the data supports it.
IDEA_CUES = {
    "pin": ["can't move", "cannot move", "unable to move", "stuck", "in front of the king", "shielding the king"],
    "fork": ["attacks two", "attacking two", "attacks both", "attacking both", "two pieces at once",
             "at the same time", "simultaneously"],
    "skewer": ["behind it", "piece behind", "in line with", "through to the"],
    "backRankMate": ["first rank", "eighth rank", "1st rank", "8th rank", "no escape", "trapped king",
                     "boxed in", "own pawns"],
    "mate": ["no escape", "no legal moves", "can't escape", "cannot escape", "king is trapped",
             "forced win", "unstoppable"],
    "mateIn1": ["no escape", "no legal moves", "can't escape", "cannot escape", "king is trapped",
                "ends the game"],
    "mateIn2": ["no escape", "can't escape", "cannot escape", "king is trapped", "forced win", "unstoppable"],
    "hangingPiece": ["unprotected", "not protected", "not defended", "for free", "free piece",
                     "can be taken", "en prise", "wins a piece"],
    "discoveredAttack": ["moves out of the way", "uncover", "unleash", "opens the line", "opening the line"],
    "deflection": ["lure", "draw away", "draws away", "pull away", "distract", "drag"],
    "sacrifice": ["gives up", "giving up", "give up", "in exchange for", "offers the"],
    "advancedPawn": ["about to queen", "close to promoting", "seventh rank", "7th rank", "2nd rank",
                     "second rank", "will queen"],
    "kingsideAttack": ["king's side", "g-file", "h-file", "around the king", "near the king", "weak king"],
    "trappedPiece": ["no squares", "nowhere to go", "no safe squares", "no escape"],
    "exposedKing": ["king is open", "unsafe", "king safety", "lack of cover", "no pawn cover", "open king"],
}
ALL_CUES = sorted({cue for cues in IDEA_CUES.values() for cue in cues})


@dataclass
class ScanResult:
    scanned: int = 0
    tactical: int = 0
    explanations_by_theme: Counter = field(default_factory=Counter)
    cue_hits_by_theme: dict = field(default_factory=lambda: defaultdict(Counter))
    gain: Counter = field(default_factory=Counter)
    gain_examples: dict = field(default_factory=lambda: defaultdict(list))
    unmarked: dict = field(default_factory=lambda: defaultdict(list))


def iter_explanations(path: Path):
    """Yield (model, theme, text) for records with string theme and explanation."""
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"{path}:{line_no}: skipping malformed JSON ({exc})")
                continue
            theme, text = record.get("theme"), record.get("t3_explanation")
            if isinstance(theme, str) and isinstance(text, str):
                yield record.get("model"), theme, text


def _record_unmarked(result: ScanResult, model: str, theme: str, text: str, cues_present: set) -> None:
    result.unmarked[theme].append((model, text))
    for cue in IDEA_CUES.get(theme, []):
        if cue in cues_present:
            result.gain[(theme, cue)] += 1
            result.gain_examples[(theme, cue)].append((model, text))


def scan(paths: list[Path]) -> ScanResult:
    result = ScanResult()
    for path in paths:
        for model, theme, text in iter_explanations(path):
            result.scanned += 1
            if theme in NON_TACTICAL_THEMES:
                continue
            result.tactical += 1
            result.explanations_by_theme[theme] += 1
            cues_present = {cue for cue in ALL_CUES if phrase_mentioned(cue, text)}
            for cue in cues_present:
                result.cue_hits_by_theme[cue][theme] += 1
            if not theme_mentioned(theme, text):
                _record_unmarked(result, model, theme, text, cues_present)
    return result


def _cue_row(result: ScanResult, theme: str, cue: str) -> dict:
    other_explanations = result.tactical - result.explanations_by_theme[theme]
    leak_hits = sum(n for other, n in result.cue_hits_by_theme[cue].items() if other != theme)
    leak_rate = leak_hits / other_explanations if other_explanations else math.nan
    gain = result.gain[(theme, cue)]
    passes = gain >= MIN_GAIN and not math.isnan(leak_rate) and leak_rate <= MAX_LEAK_RATE
    return {"theme": theme, "cue": cue, "gain": gain, "leak_hits": leak_hits,
            "leak_rate": leak_rate, "passes": passes}


def cue_table(result: ScanResult) -> pd.DataFrame:
    """One row per (theme, cue) with gain, leak and whether it passes the T3-3 thresholds."""
    rows = [_cue_row(result, theme, cue) for theme, cues in IDEA_CUES.items() for cue in cues]
    return pd.DataFrame(rows).sort_values(["passes", "gain"], ascending=False).reset_index(drop=True)


def _diverse_examples(examples: list, rng: random.Random) -> list:
    """Up to EXAMPLES_PER_CUE examples, one per model where possible."""
    by_model = defaultdict(list)
    for model, text in examples:
        by_model[model].append(text)
    models = sorted(by_model, key=str)
    rng.shuffle(models)
    return [(model, rng.choice(by_model[model])) for model in models[:EXAMPLES_PER_CUE]]


def _table_lines(table: pd.DataFrame) -> list[str]:
    lines = ["## Cue table", "", "| theme | cue | gain | leak_hits | leak_rate | passes |", "|---|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.theme} | {row.cue} | {row.gain} | {row.leak_hits} | {row.leak_rate:.4f} | {row.passes} |")
    return lines + [""]


def _example_lines(result: ScanResult, table: pd.DataFrame, rng: random.Random) -> list[str]:
    lines = ["## Examples for cues that pass", ""]
    for row in table[table["passes"]].itertuples():
        lines.append(f"**{row.theme} / {row.cue}** (gain {row.gain})")
        for model, text in _diverse_examples(result.gain_examples[(row.theme, row.cue)], rng):
            lines.append(f"- `{model}`: {text.strip()[:SAMPLE_TEXT_CHARS]}")
        lines.append("")
    return lines


def _sample_lines(result: ScanResult, rng: random.Random) -> list[str]:
    lines = ["## Unmarked explanations to read by hand", ""]
    for theme in sorted(result.unmarked, key=lambda t: -len(result.unmarked[t])):
        pool = result.unmarked[theme]
        lines.append(f"### {theme} ({len(pool)} unmarked)")
        for model, text in rng.sample(pool, min(SAMPLE_PER_THEME, len(pool))):
            lines.append(f"- `{model}`: {text.strip()[:SAMPLE_TEXT_CHARS]}")
        lines.append("")
    return lines


def build_report(result: ScanResult) -> str:
    table = cue_table(result)
    rng = random.Random(SEED)
    unmarked_total = sum(len(v) for v in result.unmarked.values())
    header = [
        "# T3 phrasing examples", "",
        f"- Explanations scanned: {result.scanned:,}",
        f"- Tactical-theme explanations: {result.tactical:,}",
        f"- Unmarked tactical explanations: {unmarked_total:,}",
        f"- A cue passes with gain ≥ {MIN_GAIN} and leak rate ≤ {MAX_LEAK_RATE}", "",
    ]
    return "\n".join(header + _table_lines(table) + _example_lines(result, table, rng) + _sample_lines(result, rng))


def main() -> None:
    parser = argparse.ArgumentParser(description="Find plain-English T3 phrasing in saved explanations.")
    parser.add_argument("--results", action="append", required=True, help="Results JSONL (repeatable)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Report path (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    setup_logging()

    paths = [Path(p) for p in args.results]
    missing = [p for p in paths if not p.exists()]
    if missing:
        logger.error(f"Results file not found: {', '.join(map(str, missing))}")
        sys.exit(1)

    result = scan(paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(result))
    logger.info(f"Wrote {out}: {result.scanned:,} scanned, {result.tactical:,} tactical")


if __name__ == "__main__":
    main()
