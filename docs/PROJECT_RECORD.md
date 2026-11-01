# Project record: experiments, decisions, corrections and remaining work

**Project:** Ryan Brew, PR09426 — *Do LLMs Reason? Benchmarking Logical Performance via Chess*

**Updated:** 1 November 2026. This records work supported by the repository and
the current handover session. It is a consolidated audit trail, not a claim to
capture unrecorded activity or to attribute every implementation to Ryan.
Ryan directed the research and authorized execution; Claude Code and Codex
assisted with implementation, review, analysis and documentation.

## 1. Research question and what the measurements establish

The project investigates how reliably language models can evaluate a chess
position, select a move, and explain who is better. It also tests whether input
notation and board-focused fine-tuning account for weak performance.

Three tasks are measured:

| Task | Measurement | Essential qualification |
|---|---|---|
| T1: numerical evaluation | Association/error against the reference engine; comparison with a constant-zero answer | Correlation and calibration are different properties |
| T2: move choice | First-answer legality, assisted final legality, CPL, blunders and engine-best-move rate versus random | Assistance and legal/scored-output selection must be disclosed |
| T3: explanation | Better-side agreement and theme keyword scoring | Keyword matches do not establish sound reasoning |

The reference uses Lc0: a network evaluation for T1 and an 800-node search for
T2. Fields named `stockfish_*` are legacy names, not proof that Stockfish supplied
the current reference. The project's blunder definition is clamped CPL ≥ 1,000.

Behavioural results can support a claim about measured chess competence. They
cannot uniquely identify a model's internal mechanism. “LLMs do not reason” is
not established by these experiments. Pattern-based performance is an
interpretation, which must be separated from the measurements.

## 2. Experiments completed or attempted

### Experiment 1 — base benchmark

- 22 local models, approximately 2B–70B, across 12 families.
- 4,000 positions, sampled across four Lichess puzzle-rating tiers.
- Six prompt formats and 526,662 saved evaluations.
- Pooled T1 correlation: **0.110**. Non-mate mean absolute error: **142.74 cp**,
  versus **46.96 cp** for always answering zero (3.04× baseline error).
- Engine-best-move rate: **10.22%**, versus **8.82%** for uniformly random legal
  moves on the evaluated positions (1.16×).
- Blunder rate among final legal/scored moves: **41.86%**. Corrected
  position-cluster 95% interval: **40.74–42.97%**.
- Better-side accuracy: **30.34%**, versus **54.21%** for the majority guess;
  side agreement κ approximately **0.067**.
- Gemma4:31b/26b have the lowest blunder point estimates in the base benchmark,
  approximately 35.8%/37.3%. Adjacent ranks are not automatically significant.
- There is no consistent monotonic benefit of size across all tested families;
  Gemma provides counterexamples to any blanket claim that size never helps.

**Status:** useful completed evidence of unreliable performance under the
harness. Retain the answers; regenerate legacy reporting with corrected methods.

### Experiment 2 — five-model rerun with raw replies

- Repeated five local models on the same benchmark while preserving raw replies.
- Their unaided first-answer legality on combined prompts is approximately
  **9.3%**, while assisted final legality is approximately **97%**.
- The system can recover a legal move from elsewhere in a reply or re-ask with
  the complete legal-move list. Final legality therefore includes assistance.
- The original other 17 models do not acquire first-answer measurements merely
  by reanalysing records that lack the required raw replies.

**Status:** completed and valuable for exposing the assistance hidden by the
headline final-legality figure. Do not generalize its 9.3% to all 22 models.

### Experiment 3 — pieces-only fine-tuning

- Five 3B–7B models received LoRA training on board descriptions from 20,000
  positions outside the benchmark.
- Reported held-out piece recall rose to approximately **89–99%**. Perfect-board
  rates remained much lower and depended on model/prompt; recall alone does not
  establish complete board understanding.
- Matched pooled final legality fell **95.88% → 76.89%**.
- Legal, scored, non-blunder answers per matched move request fell
  **56.21% → 43.96%**.
- On identical legal/scored pairs, blunders were **42.36% → 42.77%**.

**Status:** a useful negative result about this intervention. Training changed
board-description behaviour but did not improve reliable overall play. It did
not overwrite baseline answers or invalidate the research dataset. The success
criteria for this intervention were written after its results and must be
labelled retrospective/exploratory.

### Experiment 4 — pieces plus legal-move fine-tuning

- The same five models were trained to produce piece lists and legal moves.
- Registered learning prerequisite: held-out legal-list precision and recall
  both at least 90%.
- Observed recall approximately **65–81%** and precision **52–60%**: every model
  failed that prerequisite.
- Matched pooled final legality fell **95.88% → 66.90%**.
- Legal, scored, non-blunder answers per request fell **56.21% → 39.23%**.
- On identical legal/scored pairs, blunders were **41.04% → 41.29%**.
- Gemma3:4b's apparent separate-subset blunder improvement, 42.95% → 31.81%,
  shrinks to **35.28% → 31.56%** on the same 4,671 pairs. Its final legality
  fell **98.74% → 29.45%** and successful non-blunder responses fell
  **56.33% → 20.08%** of matched requests.

**Status:** the intervention failed. Whether *accurately* learning legal moves
would transfer remains unanswered. A new training attempt must pass its
held-out prerequisite before a costly downstream benchmark. Preserve the failed
experiment rather than replacing it or changing its success criteria.

### Experiment 5 — engine-strength robustness

- Rescored a 1,000-position sample at 8,000 rather than 800 Lc0 nodes.
- Model-ranking correlation approximately **0.97**; aggregate blunder rate
  approximately **42.0% → 42.1%** on that sample.

**Status:** supports the broad move-quality result's robustness to search
budget. It does not validate every individual reference move or adjacent rank.

### Experiment 6 — notation/plain-board test

- Same 500 positions, 21 local models, three formats: FEN, plain board, and plain
  board with a legal-move list. The unsafe-to-load 70B model was excluded.
- Collection was paused after CodeLlama, with 14 of 21 model passes finished.
- Ryan subsequently authorized “cap and resume”. At **00:41 GMT on 1 November**,
  the run resumed using the handoff command and append-only logging.
- Pipeline finished at **13:42:45 GMT**, with terminal **`EXIT 0`**.
- **31,487 records**: 19 models have 1,500 each, WizardLM has 1,491 and
  DeepSeek-R1:7b has 1,496. The 13 missing records remain disclosed.
- All **30,625 legal moves** have finite nonnegative CPL. Metrics and plots
  were generated. There are 487 positions shared across every model/format;
  individual measures use their own explicitly reported pairs.

Final registered primary comparison:

| Criterion | FEN → plain board | Condition for “not the main barrier” | Holds? |
|---|---:|---|---|
| N1: paired legal-move blunders | 42.2542% → 41.8761% | Improvement below 5 percentage points | Yes |
| N2: evaluation correlation | 0.1214 → 0.1672 | Plain-board ρ below 0.20 | Yes |
| N3: side agreement | 0.0802 → 0.0886 | Plain-board κ below 0.20 | Yes |
| N4: first-answer legality | 21.1861% → 25.5149% | Plain-board legality below 25% | No |

**Registered verdict: “mixed”.** Neither replace it with a preferred narrative
nor present an earlier 13/14/17-model snapshot as final. Formatting improved
legality, while the pooled blunder improvement was small and evaluation/side
agreement remained weak. The plain-board legality interval crosses 25%; the
registered rule uses the specified point estimate, not a rewritten criterion.

Evidence: [summary](../results/plain_board/metrics/plain_board_summary.json),
[CSV](../results/plain_board/metrics/plain_board_comparison.csv), and
[registered specification](../specs/plain-board-prompts.md). Results commit:
`c4541da`. Provenance records completed collection, input hashes, missingness and
the analysis-only omission of redundant dataset lookups.

### Experiment 7 — Claude commercial models

- Haiku 5.5 and Opus 5.5 low-effort each completed 500 positions × three formats
  (1,500 saved answers per variant).
- Sonnet produced only 48 retained pilot replies; the full thinking comparison
  was deferred by Ryan. **C2/C3 are “not run”.**
- Authorized offline enrichment scored all **3,048 legal Claude/pilot moves**
  at 800 Lc0 nodes, with original response identities/payloads preserved.
- Full-variant FEN first-answer legality: **88.8% Haiku; 93.8% Opus**.
- FEN evaluation correlation approximately **0.49; 0.58**, respectively.
- Full 500-position FEN blunder rates: **29.0%; 19.2%**. Engine-best-move rates:
  **17.8%; 26.8%**, versus an 8.04% random-legal baseline.
- Haiku had 41 capped initial replies. Complete saved-record coverage does not
  mean every reply was untruncated.

Final C1 uses **498 matched legal/scored pairs**, so its rates differ slightly
from the full 500-position descriptive rates:

| C1 comparison | Blunder rates | Registered outcome |
|---|---|---|
| Haiku versus Gemma4:31b | 28.9157% versus 34.5382% | **No clear difference**: reduction exceeds 5 points but confidence intervals overlap |
| Opus versus Gemma4:31b | 19.2771% versus 34.5382% | **“frontier model plays clearly better”**: 15.2610-point reduction and non-overlapping intervals |

Both full Claude variants receive C4's registered **“notation was a major
barrier”** label. This is triggered by an *absolute* plain-board correlation
threshold, even though paired correlation decreases from FEN. Report the exact
label and that limitation together; it does not demonstrate a formatting gain.

Existing recorded Claude expenditure totals **$12.2555896**, including retries
and Sonnet pilots. Offline scoring/comparison added no API expenditure.
Limitations include public-puzzle contamination, unknown parameter counts,
default API sampling settings, caps, and the unrun Sonnet comparison.

The Claude branch remains separate. Latest compared outputs are committed at
[`9ea9871`](https://github.com/rabrew/chess-llm-bench/tree/9ea9871/results/claude/metrics).
Final integration/testing and publication updates remain outstanding.

### GPT illustration and OpenAI API setup pilot

- A historical tool-disabled GPT illustration used 12 positions: 10 legal first
  moves, 12 legal final moves and six correct side answers. It is **illustrative
  only**, not a comparable benchmark.
- Ryan requested a new ordinary commercial-model benchmark, explored full-run
  costs, and chose a smaller initial scope rather than spending thousands.
- Ryan installed an OpenAI key on the PC using a hidden SSH prompt. The private
  credential is outside git, with owner-only permissions. Its value was not
  displayed, logged or committed.
- A new setup/cost pilot was specified, tested, implemented and independently
  reviewed before execution: **GPT-6.1 Sol and GPT-6 Luna, 20 shared positions
  each, five per tier, plain-board prompts, no tools or move rescue**.
- Ryan explicitly approved **40 paid requests with a $2 total ceiling**.
- The offline plan reserved approximately **$1.1682** conservatively. The pilot
  uses sequential standard Responses API requests, API defaults and a 4,000-token
  output cap including reasoning. It is not a full performance experiment.
- **At this document's initial creation, the pilot is still running.** Sol's 20
  replies are saved; Luna is in progress. Caps/missing fields remain visible.
  Final counts, costs and interpretation must come from the completed summary,
  not from a partial conversational snapshot.

Sources: [pilot spec](../specs/openai-pilot.md),
[configuration](../config/config_openai_pilot.yaml),
[runner](../scripts/openai_pilot.py), and
[tests](../tests/test_openai_pilot.py). Spec/config commit `522ea04` precedes
implementation commit `525fd9b`.

## 3. Measurement problems found and their consequences

| Problem | Correction/disposition | Does it require new model answers? |
|---|---|---|
| Repeated responses treated as independent for uncertainty | Position-cluster bootstrap; wider, more honest intervals | No; reanalyse existing answers |
| Non-move prompts counted as failed move requests | Restrict denominator to move-requesting prompts | No |
| Assisted final moves conflated with unaided ability | Separate first answers, recovery and legal-list retries | Only where original raw replies are absent |
| Random baseline computed as inverse mean move count | Use mean of per-position random-best probabilities | No |
| Mate/reference mismatch inflated CPL | Store matched search reference and use declared clamping | Re-score engine values where needed; not necessarily new LLM answers |
| Fine-tuning's legal-only selection made failures look beneficial | Matched legal/scored pairs plus success per all move requests | No |
| T3 theme matcher matched substrings such as “mate” in “material” | Whole-word scoring and corrected comparisons | No |
| Board shown precedes the actual puzzle | Retain valid evaluation/move measures for the shown board; withdraw theme/difficulty inference | Yes, for actual puzzle-solving/theme/difficulty questions |
| PGN format contained no history | Treat as repeated FEN, not a history test | Yes, to test genuine history |
| 70B prompts exceeded their context in affected formats | Disclose/exclude affected outcomes or rerun on suitable hardware | Only for claims requiring those affected outcomes; never load 70B locally |
| Thinking settings differed historically | Disclose configuration confounding; withdraw unsupported causal claims | A controlled run is needed for a causal thinking comparison |

The corrected base blunder point estimate remained **41.86%**. Its old interval
was **41.69–42.02%**, versus **40.74–42.97%** after clustering. The main effect
was honest uncertainty, not a reversal of the base result. A formerly reported
35.36% all-format failure rate was not an applicable move-failure measure; the
correct requested-move failure rate is approximately **3.04%**.

## 4. Engineering corrections and good practices actually applied

The correction work includes:

- Explicit pending states for incomplete experiments; consistent validity rules,
  paired coverage and missingness; strict JSON instead of nonfinite JSON values.
- Position-cluster uncertainty and descriptive interval-overlap labels, without
  declaring overlapping ranks equivalent.
- Dashboard CPL, theme/error, T3 scaling, provenance and cache freshness fixes;
  heatmap row/colour alignment corrections.
- Dataset/FEN cache invalidation, square-zero king handling, deterministic
  parallel validation selection and explicit dataset-source opt-ins.
- Engine EOF/deadline handling and precomputation of previously null reference
  values rather than treating mere key existence as completion.
- Worker dry runs use isolated state rather than writing to the live queue;
  training failure cleanup preserves pre-existing checkpoints/exports.
- GPU enrichment fails closed while an Ollama model is loaded or status is
  unknown; a bounded read-only idle handover avoids forced unloading.
- Terminal `EXIT 0` evidence is required before declaring pipeline completion;
  old successful log lines followed by new progress are insufficient.
- Summary-driven rendering for notation, Claude and the GPT illustration,
  including missing-input notices and a read-only consistency check.
- API pilot transport has no automatic paid retries; durable reservations/raw
  replies, locking and plan hashes prevent accidental duplicate submissions.
  Independent review caught a missing-ledger resend defect before paid use.
  Regression tests cover missing, empty and valid-prefix ledger loss.

The required workflow for new substantive changes is **spec → tests → code →
independent review/validation → commit and push**. Registered criteria remain
fixed after results. Preserve unrelated working-tree changes; one writer owns
each file. Development tests use CPU-only temporary fixtures/mocked services,
never the live benchmark queue or training runs.

Verification recorded in this session:

- Corrected implementation baseline: **1,132 tests passed** earlier.
- Main completion/renderer work: **1,164 tests passed**.
- Claude worktree: **1,105 passed, one skipped** for a missing real-data fixture.
- OpenAI implementation: **1,196 full-suite tests passed**, then the final
  **34-test focused suite passed** after two extra ledger-loss cases were added.
  This is not a claim that a 1,198-test full suite was run.
- Required Ruff scope and findings renderer checks passed. The remaining
  python-chess deprecation warning was pre-existing.

Important remaining practice improvements: finish regenerating legacy reports,
complete branch integration and final independent checks, and replace temporary
analysis dependencies with durable reproducibility artifacts where appropriate.
Tests passing establish implementation checks, not scientific validity by
themselves.

## 5. Ryan's operational decisions and constraints

- Preserve research answers, datasets, checkpoints and active runtime state.
- Never reboot the remotely accessed PC; never load 70B models locally.
- Do not edit files used by a running job or merge source changes over it.
- Ask before paid API/cloud work; do not print, log or commit credentials.
- Commit and push completed steps; do not discard existing user work.
- Reduce overnight noise: Ryan approved a GPU power limit of **250 W**, reduced
  from 360 W. The limit was verified and is not automatically restored.
- Resume experiment 6 only after explicit authorization; that authorization was
  given and execution is now complete.
- Finish experiment 6 before merging the Claude worktree. Do not remove the
  worktree until its work is merged and preserved.
- Aim to finish research with time for writing before approximately mid-December.
  A precise campaign end date has not yet been agreed.

## 6. Newly reported free allowance and next commercial-model direction

Ryan supplied this allowance wording on 1 November:

- Up to **250,000 tokens/day shared across** GPT-5.4, GPT-5.2, GPT-5.1, GPT-5,
  GPT-4.1, GPT-4o, o1 and o3.
- Up to **2,500,000 tokens/day shared across** GPT-5.4-mini/nano,
  GPT-5-mini/nano, GPT-4.1-mini/nano, GPT-4o-mini, o3-mini and o4-mini.

This is a **user-reported account allowance**, not independently verified billing
entitlement. Exact project enrolment, shared external usage, reset rules and
over-quota billing still need confirmation. Output usage includes reasoning;
do not add the reasoning count twice when calculating total tokens or cost.

Ryan's latest direction is to **prioritise GPT-5.4 after the current pilot,
retain Luna, and leave mini/nano out**. Sol/Luna are not in the pasted free list.
The approved paid pilot may finish, but further paid Luna expansion has not
been authorized. No GPT-5.4 requests or automatic daily campaign have started.

Proposed next setup: reuse the fixed 500-position cohort, preserve raw replies
and first-answer scoring, add durable daily token accounting and conservative
stopping below the free quota, and verify entitlement before making a zero-cost
claim. A local counter alone cannot see another application's use of the shared
allowance. This proposal is not an implemented or running daily campaign.

## 7. What is good, unfinished, or needs a new experiment

| Work | Disposition |
|---|---|
| Base evaluation/move results | Retain, with corrected uncertainty and denominator/assistance disclosures |
| Five-model raw-reply rerun | Retain; essential evidence about unaided versus assisted performance |
| Pieces-only fine-tuning | Retain as a limited negative intervention result, with retrospective criteria disclosed |
| Legal-move fine-tuning | Retain failed intervention; retrain successfully before claiming to test accurate-rule transfer |
| Engine-strength check | Retain as broad robustness evidence |
| Plain-board collection/scoring/comparison | Complete; registered result mixed; final publication integration remains |
| Claude Haiku/Opus comparison | Complete; matched C1 supports Opus, not a clear Haiku advantage |
| Sonnet thinking comparison | Not run; optional, requires a separate approved budget and controlled design |
| Actual puzzle themes and difficulty | New experiment on the actual puzzle positions |
| Genuine move-history effect | New experiment with real histories |
| GPT 12-position illustration | Illustrative only |
| OpenAI 40-request pilot | Running at initial record creation; report final completeness/caps/cost before expansion |
| GPT-5.4 daily free-quota benchmarking | Requested direction; entitlement confirmation, spec/tests/code and execution remain |

The outstanding publication work includes final experiment 6/7 sections and
generated tables in `FINDINGS.md`, the HTML write-up pages, updated conclusions,
README/checklist statuses, full branch integration checks, master fast-forward
and eventual Claude-worktree removal. Renderer support exists; that is not the
same as every final page already being updated.

## 8. Where to verify the record

- [Project findings](FINDINGS.md), [post-fix conclusions](CONCLUSIONS.md),
  [numbered issues](issues-log.md) and [handoff](HANDOFF.md). Earlier snapshots
  in those documents may precede the final experiment 6/7 results recorded here.
- [Analysis correction specification](../specs/analysis-review-corrections.md),
  [completion specification](../specs/experiment-completion.md),
  [completion safeguards](../specs/completion-safeguards.md),
  [table rendering specification](../specs/experiment-tables.md).
- Main correction/conclusion commits: `c06d3da`, `326e0a3`; completion safeguards
  `91f14ba`; table renderer `d855a7a`; capped resumption record `4b6f990`;
  notation comparison `c4541da`.
- Claude scored-response/validation commit `a141909`; comparison implementation
  `99f73b8`; initial reports `1da3d14`; matched final reports `9ea9871` on the
  separate `claude-models` branch.
- OpenAI specification/config `522ea04` and implementation `525fd9b`. Pilot
  outputs live under `results/openai_pilot/`; no credential belongs there.

For subsequent updates, append dated facts or explicitly supersede old status.
Do not rewrite registered criteria, erase failed experiments, promote an interim
result to final, or present a proposed action as completed.
