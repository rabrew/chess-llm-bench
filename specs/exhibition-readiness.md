# Spec: Young Scientist Exhibition Readiness

**Project:** PR09426, *Do LLMs Reason? Benchmarking Logical Performance via Chess*
**Created:** 2026-10-20
**Supersedes as the master list:** the defect register in [metric-defect-remediation.md](metric-defect-remediation.md) (D1–D11). The detail for each D-item stays there. This file tracks status and adds the issues found on 2026-10-20.

---

## Goal

Make every number and claim shown at the exhibition defensible. That means three things:

1. Fix or clearly disclose every known measurement problem.
2. Re-run the post-processing (no new model runs unless an item says otherwise) to get corrected numbers.
3. Run the fine-tuning experiment (section F) on a clean, leak-free setup, scored with the same corrected metrics.
4. Only then write the final conclusion, combining the base benchmark and the fine-tuning results, and line it up with the submitted proposal.

**The base-benchmark findings are interim until section F is done.** They describe how the models behave *before* fine-tuning. The final answer to "reason or pattern match?" depends on what fine-tuning changes. For example, does teaching a model to read the board fix its moves, or only its board descriptions?

**Done when:** every P0 and P1 box below is ticked, including section F, or has a written reason for being left open, and FINDINGS, SIMPLE_SUMMARY, README and the two explainer pages are regenerated from the corrected `summary.json`.

## Inputs / Outputs

**Inputs**
- `results/evaluations.jsonl`: 526,662 records, **only on the Linux PC**
- `data/{easy,medium,hard,extreme}.json`: the 4,000 positions, **only on the Linux PC**
- The Lichess puzzle CSV, needed only for items T3-4 and T2-3
- The submitted proposal (project PR09426), for the claim alignment in section W
- For section F: base-model weights from HuggingFace, and the training config in `fine-tuning/config/`

**Outputs**
- Corrected CSVs in `results/metrics/`, plus `summary.json` with a provenance block
- Rewritten `docs/FINDINGS.md`, `docs/SIMPLE_SUMMARY.md`, `README.md`
- Updated explainer pages (technical and plain-English)
- This file with every box ticked or explained

## Steps / Logic: the to-do list

Priorities: **P0** = a claim is wrong or can't be defended until this is done. **P1** = it materially changes a number or ranking. **P2** = presentation and polish.

Every code change follows the CLAUDE.md order: spec, then tests, then code, then commit and push.

### 0. Access (blocks almost everything)

- [ ] **A-1 · Get the Linux PC reachable (P0).** On 2026-10-20 it timed out on both the LAN and the Tailscale address. The Tailscale key had already expired by 2026-10-04. Run `sudo tailscale up` and turn off key expiry in the Tailscale admin console.
- [ ] **A-2 · Back up the raw data (P0).** Copy `results/evaluations.jsonl` and `data/*.json` somewhere other than the PC, such as an external drive or cloud storage, not git. If they are lost, there are no results at all.

### T1. Evaluation (who is winning, by how much)

- [x] **T1-1 · Chance-corrected direction metrics (D1). Code done.** Commits `0131404` (spec), `7b29c38` (tests) and `eef16c3` (code) add the majority-class baseline, Cohen's κ, balanced accuracy and Spearman ρ. See [direction-baseline-kappa.md](direction-baseline-kappa.md).
- [x] **T1-2 · Run T1-1 on the real data (P0).** *Done: metrics regenerated in `882144b` and `72f63fe`; the Finding 3 rewrite is part of W-8.* `python scripts/generate_plots.py --save-metrics`. Then read `direction_agreement.csv`. If κ ≤ 0.1 for most models at ±100 and ±200, Finding 3 has to say the models barely beat "always say even".
- [ ] **T1-3 · Perspective-bug check (D2, P0).** Split direction accuracy by side to move. If Llama-3.3-70B or Gemma4-31B is right when White is to move and inverted when Black is to move, they are reporting from the side-to-move's view. Fix the parser and recompute. Also hand-check the sign on 50 raw responses for each of those two models. Do this before making any claim like "70B is worse than random".
- [ ] **T1-4 · Lead with Spearman ρ (D2, P1).** Rank-correlation needs no threshold. Make it the main T1 measure and move the four-threshold table to a robustness section.
- [x] **T1-5 · Correct the ±16,000 cp claim (D9, P1).** *Done in `a4aa928`; `tests/test_docs_engine_claims.py` (`fffe1ed`) stops it returning.* Measure the largest |truth| in `data/*.json`. If it is 10,000, the Lc0 cap, fix A1 and A2 in FINDINGS.

### T2. Move selection

- [ ] **T2-1 · Report the blunder rate as the main T2 number (D5, P1).** The share of legal moves with CPL ≥ 1000, per model, with bootstrap 95% confidence intervals. Write `cpl_buckets.csv` with the five buckets for each model. Keep the mean and median of clamped CPL next to it.
- [x] **T2-2 · Split legality by worker version (D10, P0).** *Superseded 2026-10-24 by [t2-move-source.md](t2-move-source.md): there is no pre-rescue population (every run had the rescue), so legality is split by move source instead, and first-attempt legality comes from saved raw replies. Issues 18 (corrected) and 20.* Original text: Records with a stored *illegal* move must come from before the rescue code existed. Use them to find the cutover time, label every record `pre_rescue` or `post_rescue`, and report legality separately for each. Never quote the pooled 97.85% again. If the pre-rescue group has at least 1,000 records for a model, that model gets a real first-attempt legality figure.
- [ ] **T2-3 · Correct the best-move random baseline (P1).** "About 3.6% by chance" uses an average of 28 legal moves. Replace it with the mean of 1 ÷ (number of legal moves) across the actual positions. That needs only the FENs and python-chess.
- [x] **T2-4 · Fix the `move_only` 96.5% figure (D10 step 7, P1).** *Resolved 2026-10-24: 96.5% (0.9647 on 87,795 standard records) is legal ÷ all `move_only` rows, counting blanked moves as illegal, which is the corrected Issue-16 definition. "100% by construction" holds only for the old metric that skipped blanks (legal ÷ non-null = 1.0000). The figure stands, but it means "picked a legal move from the legal-move list the prompt supplies" (Issue 20). The write-up must say so.* Original text: After the cutover, `move_only` legality is 100% by construction. Work out where 96.5% came from and correct it.
- [ ] **T2-6 · Clamped CPL mixes two different engine readings (P1, new 2026-10-20).** `scripts/enrich_cpl.py` stores the raw `t2_cpl` as Lc0-800-node(before) − Lc0-800-node(after). `load_results_df` then rebuilds "eval after" as `t1_stockfish_eval − t2_cpl`, but `t1_stockfish_eval` is the *no-search* network reading of the starting position, not the 800-node one. Wherever those two disagree, `t2_cpl_clamped` and `t2_wp_loss` are off. Fix: re-evaluate only the 4,000 starting positions at 800 nodes (a few minutes of GPU, no LLM runs), store that as `t2_eval_before`, and rebuild from it. Then re-check the disaster-rate and bucket figures.
- [ ] **T2-5 · Engine-strength sensitivity check (D7, P2, optional).** Re-score 1,000 positions with Lc0 at a higher node count and show the model ranking doesn't change. This is GPU time on the PC, not new model runs.

### T3. Explanation

- [x] **T3-1 · Whole-word theme matching (P0).** *Done: `dd51e11` / `8c137bb` / `d076777`.* The current substring test credits "mate" inside "estimate", "approximately" and "material", "pin" inside "opinion", and "even" inside "eventually". Checked against `score_t3` on 2026-10-20. Match whole words plus the endings -s, -es, -d, -ed and -ing, so "forks" and "mated" count but "material" doesn't. Apply it in both `src/evaluator.py` and `src/metrics.py::rescore_t3_theme`. Keep a single shared matcher, with no copy-pasted logic. This is post-processing only.
- [ ] **T3-2 · Gather real phrasing examples (P1).** Run `find_t3_examples.py` (now in the session scratchpad; move it into `scripts/` with a test if it's kept) on `evaluations.jsonl`. It reports:
  - (a) marks given only via a match inside a longer word
  - (b) unmarked answers that use a plain-English description of the idea
  - (c) 15 unmarked explanations per tactical theme, to read by hand
- [ ] **T3-3 · Add plain-English descriptions to the synonym lists (P1).** For example, "can't move without exposing the king" for pin, or "attacks two pieces at once" for fork. Use only examples from T3-2, not guesses. Add a test for every new phrase, plus a test for each false positive it must not trigger.
- [ ] **T3-4 · Primary-theme problem (P1).** `dataset_builder.py` keeps only `themes[0]`. For about two thirds of records, that label is `crushing` or `advantage`, so the "idea" mark only checks for words like "winning" or "better". At minimum, report P2 separately for tactical themes only. Better: join back to the Lichess CSV on FEN to recover the full theme list, with no model re-run.
- [ ] **T3-5 · Chance-correct the side mark (P1).** P1 uses ±50 cp, but models say "Equal" only 7.4% of the time, so they lose the mark on close positions almost automatically. Report the side mark with the same baseline and κ treatment as T1-1.
- [ ] **T3-6 · Use v2 scores in H3, H4 and the family table (D4, P1).** These still use pre-audit columns. `by_model_family.csv` also still has the buggy 64.6%-style legality.

### M. Models, setup and reproducibility

- [ ] **M-1 · Llama-3.3-70B context window (P0, new).** `src/llm_client.py` gives 70B models `num_ctx: 512`, against 1024 for every other model, with `num_predict: 400`. The `move_only` prompt is about 240 tokens of system prompt plus the move list. That is a rough estimate (characters ÷ 4), not a tokenizer count. Prompt plus answer can exceed 512, and Ollama then quietly drops the oldest tokens. Llama-3.3-70B is also the model with the strangest T1 results. To check: re-send about 50 stored prompts at `num_ctx` 512 and 1024 and compare the answers. This is a **small re-run**, about an hour. If they differ, mark Llama-3.3-70B's results as affected, or re-run that model.
- [ ] **M-2 · Thinking mode switched off partway through (P0, new).** `think: false` for DeepSeek-R1 and Gemma 4 was added in the catch-up commit `1e799bb` (9 April), during collection. FINDINGS Finding 6 credits DeepSeek-R1's T1 results to chain-of-thought reasoning. If thinking was off for most of its records, that explanation is wrong. To check: label records by timestamp against the cutover, and compare response length and `inference_ms` before and after. Rewrite or remove Finding 6 to match.
- [ ] **M-3 · Temperature was not fixed for most prompts (P1, new).** Only `move_only` and the illegal-move retries use `temperature=0`. The other five prompt formats use Ollama's default, so model answers are not deterministic. The proposal says "fixed random seed to guarantee complete reproducibility", but the seed only controls puzzle sampling. Either measure the run-to-run spread by re-asking a 200-record sample three times, or reword the claim to say the dataset and scoring are reproducible, not the model outputs.
- [x] **M-4 · Model family and size parsing (D3, P0 for any family or scaling claim).** *Done: explicit `MODEL_REGISTRY` that raises on unknown tags (`b6b1bc6` / `e80ea2c` / `f68da96`).* Eight of 22 models fall into family "unknown", and `e2b`, `e4b` and `8x7b` parse as 0B, so different models are merged. Fix `parse_model_info` (exact-tag lookup, raise on unknown), then regenerate H4 and the family table.

### P. Provenance and hygiene

- [x] **P-1 · Correct the engine description everywhere (D9, P0).** *Done: `a4aa928`, guarded by `fffe1ed`.* README, FINDINGS, SIMPLE_SUMMARY, `benchmark.md` and `benchmark-detailed.md` say "Stockfish 17 at depth 22". The proposal correctly says Lc0. First confirm with the check commands in D9, then state the actual engine, the search budget and the reason (GPU throughput).
- [ ] **P-2 · Provenance block in `summary.json` (D8, P1).** Generated-at time, git SHA, engine versions, config snapshot. Generate doc numbers from `summary.json`, never type them by hand.
- [x] **P-3 · Remove the stale `learning_deltas.csv` (D11, P1),** *Done: `b4e26f8`. `results/metrics/` is tracked (small, corrected outputs); `.gitignore` agrees as of 2026-10-24.* and make `save_metrics()` delete it when there is no correction data. Decide whether `results/metrics/` is tracked or ignored, and make `.gitignore` agree.

### F. Fine-tuning experiment (in scope: part of the final conclusion)

Detailed design: [fine-tuning/specs/chess-board-finetuning.md](../fine-tuning/specs/chess-board-finetuning.md). These items make sure its results can be trusted and compared fairly with the base benchmark.

- [x] **F-1 · Training data must not overlap the test positions (resolved 2026-10-21).** The original concern was wrong: `data/{easy,medium,hard,extreme}.json` is the 5.8M-position Lichess pool, and the benchmark used only 4,000 of those positions. The training set was built with `--exclude-benchmark`, which skips every FEN in `results/evaluations.jsonl`. `fine-tuning/tests/test_board_dataset.py::TestBenchmarkExclusion` checks the real data and finds 0 of the 20,000 training FENs among the 4,000 benchmarked ones (commit `3672733`).
- [ ] **F-2 · Keep a held-out set separate from the benchmark (P1).** `split_held_out` takes 200 records from the training pool for per-epoch board accuracy. After F-1 that pool is benchmark-free. Confirm the held-out positions are also excluded from training (they are split off, so check the test covers this).
- [ ] **F-3 · Score fine-tuned models with the corrected metrics (P0).** `compare_finetuning_results.py` must use the post-audit columns: T1 κ, baseline and Spearman ρ (T1-1); blunder rate with confidence intervals (T2-1); legality split by regime (T2-2); and whole-word T3 (T3-1). Otherwise base and fine-tuned models are compared on numbers already known to mislead.
- [ ] **F-4 · Same model settings for base and fine-tuned runs (P0).** Use the same prompts, `num_ctx`, `num_predict`, thinking flag, temperature and rescue behaviour. Fix M-1, M-2 and M-3 *before* the fine-tuned runs, so that base and fine-tuned answers are collected the same way. Where the base run used old settings, re-run the base models on the fine-tuning subset too, so the comparison is like for like.
- [x] **F-5 · Record which models were fine-tuned and why (P1).** *Done: `97d2617`, Model Selection in `fine-tuning/specs/chess-board-finetuning.md`.* One model per family/size is enough. State the choice and keep the base version of each model in the same comparison table.
- [ ] **F-6 · Decide the comparison before seeing the results (P1).** Write down, before the runs, what result would count as "fine-tuning helped reasoning" and what would count as "it only improved board reading". For example: board-description accuracy (T0) goes up, but blunder rate and T1 κ don't change. That points to pattern matching.
- [x] **F-8 · Board accuracy ignores invented pieces (P0).** *Done: recall, precision, perfect-board rate and mean extra pieces (`952f646` / `e58069a` / `cc449dc`).* `verify_board_description` in `src/board_state.py` returns `accuracy = correct_pieces / total_pieces`. Extra pieces are listed in `extra_pieces` but don't lower the score, so a model that names every real piece and adds fake ones still scores 1.0. Report three numbers instead: piece recall (the current score), the share of boards with no errors at all (the existing `correct` flag), and the average number of extra pieces per board. The held-out figures logged during training (Qwen 98.9% then 99.5%, Mistral 99.8% then 100%, logged 2026-10-20) used the old score. The generated text wasn't saved, so they can't be re-marked; F-9 replaces them. Do this after the current training batch, so the export step isn't affected.
- [x] **F-9 · Before-and-after board-reading test (P0).** *Done 2026-10-21: `results/finetuning_experiment/board_t0.jsonl` (benchmark prompt) and `board_t0_training_prompt.jsonl` (training prompt), 200 held-out positions, base and fine-tuned.* Rebuild the 200 held-out positions (`split_held_out`, seed 42, n=200). Ask each base model and its `-board-ft` version through Ollama, with the same settings, and mark the answers with the F-8 score. Save every answer to `results/finetuning_experiment/board_t0.jsonl`, so it can be re-marked later. This gives the missing "before" number, measures the compressed (q4_k_m) model that is actually benchmarked, and recovers the Llama 3.2 figure, which was never saved to a file.
- [ ] **F-7 · Track experiments (P1).** Each training run gets an entry in `runs/` or `experiments/` with config, git SHA, seed, data hash and metrics, as the CLAUDE.md ML rules require. Keep model weights and GGUF files out of git.

### W. Write-up and claims (after the numbers are corrected and section F is done)

- [ ] **W-1 · Align with the proposal (P0).**
  - The proposal says 4 task prompts. The data has 6 prompt formats (3 combined plus 3 isolated). Explain this.
  - It says 1,000 positions. It is 1,000 *per tier*, 4,000 in total.
  - It says reproducible. Reword this to match M-3.
- [ ] **W-2 · Reframe the main argument (P0).** Scaling alone can't separate reasoning from pattern matching, because bigger models also memorise more. Present three lines of evidence together:
  1. flat across size
  2. flat across difficulty (H1 and H2 not supported)
  3. "perfect or disaster" move quality

  Present these as **interim findings about the base models**, not the final conclusion. The final conclusion waits for section F, and should say whether fine-tuning changed any of the three. Whatever the result, phrase it as "more consistent with", not "proves".
- [ ] **W-3 · Lead with the bimodal move-quality result (D6, P1)** and move legality down the list.
- [ ] **W-4 · Contamination caveat (P1).** Lichess puzzles are public and may be in the training data. Say this openly. It strengthens the result, because the models still fail.
- [ ] **W-5 · Scope statement (P1).** Open-source models up to 70B, on chess only. No claims about commercial models or about LLM reasoning in general.
- [ ] **W-6 · Prompt-format comparison (P1).** Does `cot` beat `fen_only`? Does the PGN history help? This is post-processing only, and it speaks directly to "reasoning or pattern matching". Check against M-2 first.
- [ ] **W-7 · Audit section as a contribution (P2).** Show how naive metrics misled before the corrections: legality 64.6% vs 97.85%, CPL 4,735 vs 702, the ±50 threshold, and now T1 κ and T3 substrings. That is strong evidence of scientific method.
- [ ] **W-8 · Regenerate every document and page (P2)** from the corrected `summary.json`: FINDINGS, SIMPLE_SUMMARY, README, plots, the technical explainer and the plain-English guide.

### Optional (only if time allows)

- [ ] **O-2** Commercial-model comparison ([commercial-models.md](commercial-models.md))
- [ ] **O-3** Judge-model marking of T3 explanations (Option 2), validated against a hand-marked sample

## Suggested order

1. A-1, A-2
2. P-1, T1-2, T1-3, M-1, M-2
3. T2-2, T3-1, M-4
4. T3-2 → T3-3
5. F-1 (before any training), then F-4, then the fine-tuning runs, then F-3
6. Remaining P1 items
7. W-1 to W-8

Do the cheapest high-impact checks (M-1, M-2, T1-3) early, because their results may change which findings survive.

## Edge cases

| Situation | What to do |
|---|---|
| The PC or its data is lost | Stop. Nothing can be recomputed. Present the April metrics with every caveat in this file disclosed. |
| T1 κ ≈ 0 for most models | Rewrite Finding 3 as "the models do not reliably beat always-say-even". Drop the "play vs reason" decoupling claim. |
| A perspective bug is confirmed (T1-3) | Fix the parser, recompute T1 and T3-side, and redo every ranking that uses them. |
| The 70B context overflow is confirmed (M-1) | Flag Llama-3.3-70B as affected in every table, or re-run that one model at `num_ctx` 1024 (about 24k jobs). |
| The pre- and post-thinking split can't be separated (M-2) | Remove the CoT explanation from Finding 6 and say the thinking setting changed mid-run. |
| The pre-rescue group is too small for a model (T2-2) | Publish no first-attempt legality figure for that model. Say so; don't extrapolate. |
| Fine-tuning improves board reading but not moves or evaluation | A legitimate result. Report it as evidence that knowing where the pieces are doesn't by itself produce good play. |
| Fine-tuning can't be finished in time | Present the base findings clearly labelled as interim, and fine-tuning as work in progress with whatever partial results exist. |
| Not enough time before the exhibition | Do every P0 item. Disclose open P1 items in a "Limitations" section rather than dropping them silently. |

## Dependencies

- The Linux PC with Ollama (only for M-1, M-3 and the optional T2-5), Python 3.11, and the packages in `requirements.txt`
- python-chess, for T2-3 and T1-3
- The Lichess puzzle CSV, for T3-4 (full themes) only
- Fine-tuning: Unsloth, TRL, datasets and huggingface-hub (already in `requirements.txt`), plus the RTX 5080 on the PC
- No new LLM runs are needed for the base benchmark except M-1 (small), the optional M-3 sample, and anything an edge case triggers. Section F needs its own runs by design.
