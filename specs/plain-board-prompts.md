# Spec: Plain-Board Prompts (the notation check)

**Builds on:** [isolated-prompt-formats.md](isolated-prompt-formats.md), [retry-illegal-moves.md](retry-illegal-moves.md), [t2-move-source.md](t2-move-source.md)
**Created:** 2026-10-31, before any results

---

## Goal

Every benchmark prompt so far gave the position as a FEN string. A fair objection: the models may
fail because FEN is hard to read, not because they can't play chess. This experiment gives the base
models the **same positions in plain form**, as a board diagram and a plain-English piece list with
no FEN anywhere, and asks the same three questions. If the scores stay near chance, notation was
not the reason the models failed. If they rise, that is a finding about input format.

## Success criteria (fixed before running)

**Primary comparison:** `board_text` against a `fen_only` control run **in the same run**, on the
same positions and models, paired on (model, position). Running the control again keeps the code,
Ollama version and settings identical, and saves raw replies, so every model gets a first-answer
legality figure (the base run has it for 5 models only).

Measures (pooled over models, with seeded bootstrap 95% CIs, 1,000 resamples, seed 42):

| # | Measure | "Notation not the barrier" if `board_text`... |
|---|---|---|
| N1 | Blunder rate (clamped CPL ≥ 1,000, final legal moves) | is less than 5 points lower than `fen_only` |
| N2 | T1 Spearman ρ against the engine | stays below 0.20 |
| N3 | Side mark, Cohen's κ against the majority baseline | stays below 0.20 |
| N4 | First-answer legality (combined prompt, before any rescue) | stays below 25% |

**Decision rule:**
- All four hold: **notation is not the main barrier.** The FINDINGS discussion says so with these numbers.
- N1 drops by 5 points or more with non-overlapping CIs, or N2 reaches 0.30: **notation was a
  major barrier.** This is reported as a finding, and the main conclusion is revised.
- Anything else: **mixed**. Each measure is reported on its own, with no overall claim.

Thresholds: 5 points is about the gap between the two best models (Gemma 4 31B/26B, 36–37%) and the
rest (40–45%). ρ 0.20 is about the best current model (0.25); 0.30 would beat every model.

**Secondary (reported, not judged):** `board_text+legal` against `board_text`. Does listing the
legal moves raise first-answer legality, and does it change the blunder rate? Also each model's
paired differences, and the share of illegal first answers that are moves for the wrong side.

## Inputs / Outputs

**Inputs**
- The 4,000 benchmark positions, through the existing job pipeline.
- `config/config_plain_board.yaml`: the 22 base models **minus llama3.3:70b** (loading it on the PC
  crashes the desktop), the three formats below, and its own jobs database and results paths.

**Outputs**
- `results/plain_board/evaluations.jsonl`: one record per job, in the usual format, with raw replies.
- `results/plain_board/metrics/`: the usual metrics from `generate_plots.py --save-metrics`, plus
  `plain_board_comparison.csv` (per model and pooled, per measure and format) and
  `plain_board_summary.json` (N1–N4 values, CIs, the verdict, seed, sample and provenance).

## Steps / Logic

### 1. Sample
- `benchmark.sample_per_tier: 125` (new key). After the usual `max_positions_per_tier: 1000` draw with
  seed 42, the job generator takes a second seeded draw of 125 per tier from those 1,000 (from sorted
  ids, `random.Random(42)`), so every position is one of the benchmark positions. 500 positions in all.
- Unset or 0: no second draw, so every existing config behaves as before.

### 2. Plain board text (`src/board_text.py`, new)
`fen_to_board_text(fen)` returns, with no FEN or FEN fragment:

```
Board (White at the bottom, rank 8 at the top):
8 | r . . q k . . r
7 | p p . . . p p p
...
1 | R . . Q K . . R
    a b c d e f g h
Upper case = White, lower case = Black. K king, Q queen, R rook, B bishop, N knight, P pawn, . empty.

White pieces: King e1, Queen d1, Rooks a1 h1, ...
Black pieces: King e8, ...
Side to move: White
Castling still possible: White king side, White queen side, Black king side
En passant: none
```
- The piece lists reuse the wording of `fine-tuning/src/board_state.py` (`fen_to_board_description`), moved
  or imported so there is one implementation.
- Castling reads "none" when no rights are left; en passant names the target square, e.g. "e3".

### 3. Prompt formats (`src/llm_client.py`)
- **`board_text`:** the combined three-task prompt (`build_prompt`, the `fen_only` wording), with
  "Current position (FEN): …" replaced by the board text. Questions and response format are unchanged.
- **`board_text+legal`:** the same, plus a `Legal moves: …` line (SAN, sorted).
- **`fen_only`:** unchanged (control).
- **Move rescue for the plain formats:** if the move is illegal and no legal move is found in the
  reply, the re-ask uses `build_board_move_prompt(fen)`, the `move_only` wording with the board text
  and legal moves in place of the FEN. A FEN never reaches the model in a plain format, even on retry.
  The system prompt for that re-ask drops the word "FEN".
- Moves are still answered in SAN, and `extract_move_from_text` also accepts UCI (`e2e4`).

### 4. Metrics (`src/metrics.py`)
- Add `board_text` and `board_text+legal` to `MOVE_PROMPT_FORMATS` and `COMBINED_PROMPT_FORMATS`, so
  legality, move source and first-answer legality count them.
- CPL is filled in by the existing `enrich_cpl.py` (Lc0, 800 nodes). Every position is a benchmark
  position, so its "before" eval is already cached.

### 5. Comparison (`scripts/plain_board_compare.py`, new)
- Pairs records on (model, position_id) across formats; only pairs where both sides have a value count.
- Computes N1–N4 and the secondary measures per model and pooled, with bootstrap CIs over positions.
- Applies the decision rule and writes the CSV and summary JSON.

### 6. Timing probe, then the run
1. Probe: 20 positions × 3 models (llama3.2:3b, qwen2.5:14b, mixtral:8x7b) × 3 formats. It logs
   the time per job and the prompt token count reported by Ollama (`prompt_eval_count`).
2. Estimate the full run: 500 positions × 21 models × 3 formats = 31,500 jobs, plus retries.
3. If the estimate is over 60 hours, cut to 60 per tier (240 positions) **before** starting, and
   record the change in this spec. The criteria do not change.
4. Run in a `tmux` session with the existing workers; it resumes from the jobs database.

## Edge cases

- **Prompt longer than the context window:** the probe checks `prompt_eval_count` against `num_ctx`
  for the longest prompt (most pieces and legal moves). If any prompt would be cut off, stop and
  raise `num_ctx` for that model before the run. Never run truncated.
- **No legal moves** (mate or stalemate): benchmark positions are puzzle positions and always have
  moves; `board_text+legal` writes "Legal moves: none" if one ever appears.
- **Promoted pieces** (two queens): the piece list shows both, as in `board_state.py`.
- **A model replies with a FEN or a square list instead of a move:** handled as now; no legal move
  means the rescue runs.
- **Missing pairs:** a failed job on either side drops that pair from that measure, and the count of
  dropped pairs is reported per model.
- **`sample_per_tier` larger than the tier:** take the whole tier, as `DataLoader.sample` does.

## Dependencies

- python-chess, pandas, numpy, Ollama on the PC (RTX 5080, 16 GB VRAM), Lc0 for CPL. No scipy.
- No new packages.

## Reproducibility

- Seed 42 for both sampling draws and the bootstrap.
- Models run at Ollama's default temperature, as in the base run (move re-asks at 0, as before), so
  outputs are not bit-for-bit repeatable; the same-run `fen_only` control handles that.
- `plain_board_summary.json` records the git SHA, the config snapshot, the model digests from
  `ollama list` and the sample size actually used.

## Tests (`tests/test_plain_board.py`)

- `fen_to_board_text`: start position diagram is exact; a position with Black to move, no castling
  rights and an en passant square; promoted pieces listed; the output never contains the FEN's
  placement field.
- Prompt builders: `board_text` prompt has no FEN; `board_text+legal` lists exactly python-chess's
  legal moves; questions and response format match `fen_only`.
- Worker: an illegal move on a plain format re-asks with the board move prompt, never the FEN one
  (fake client records prompts).
- Sampling: `sample_per_tier` gives 125 per tier, all inside the 1,000, the same on every call; unset
  leaves the old behaviour.
- Comparison: pairing drops unmatched rows; the decision rule returns each of the three verdicts on
  hand-built inputs; bootstrap is repeatable with the seed.
