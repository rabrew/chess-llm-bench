# Spec: T1–T3 Benchmark Run for the Fine-Tuned Models

**Closes:** F-4 for batch 1 in [exhibition-readiness.md](exhibition-readiness.md)
**Created:** 2026-10-22

## Goal

Run the full T1, T2 and T3 benchmark on the batch-1 fine-tuned models (five of six; see "Gemma 4 exclusion"). The point is to see whether
board-reading training changes evaluation, move choice or explanation. The models are:

- llama3.2:3b-board-ft
- gemma3:4b-board-ft
- qwen2.5:7b-board-ft
- mistral:7b-board-ft
- deepseek-r1:7b-board-ft
- gemma4:e2b-board-ft

**Decision (2026-10-22):** the fine-tuned models are run with *exactly the base models' settings*. That
means the same positions, prompts, chat template, system prompt, Ollama sampling defaults, thinking flag,
`num_ctx`, `num_predict`, illegal-move retry and Lc0 CPL scoring. Base and fine-tuned records then differ
only in the weights, and they compare directly with the base records already in
`results/evaluations.jsonl`.

The base runs for these six models were collected after the `think: false` cutover of 9 April, so M-2
does not affect them. M-3 (default temperature) applies equally to both sides. It is handled in the
write-up, not by re-running.

## Gemma 4 exclusion (found 2026-10-22)

gemma4:e2b-board-ft is left out of this run, so it is five models, not six.

- The trained adapter reads boards correctly in HuggingFace, with or without Unsloth (plain transformers +
  PEFT, bf16).
- The same merged weights converted to GGUF fail in llama.cpp and in Ollama, even at BF16 with no
  quantization. They open with `<|channel>thought` and misread the FEN. `verify_export` recall is 0.13.
- The untrained base also behaves differently between HuggingFace and llama.cpp on the raw training
  prompt.

So llama.cpp's Gemma 4 conversion or inference does not yet match HuggingFace for these weights. This is
not a training or template problem. The checkpoint (`checkpoint-2476`) is kept, and the model can be
re-tried after an Ollama or llama.cpp update by running `reexport_gguf.py` and then `verify_export.py`.
Batch 2's Gemma 4 E4B will probably hit the same problem; `verify_export` will catch it.

## Problems this fixes

1. **The fine-tuned Modelfiles do not match their bases.**
   - `import_finetuned_to_ollama.py` writes only `FROM`, `num_ctx` and `num_predict`.
   - Through `/api/chat`, five of the fine-tuned models therefore get **no chat template**
     (`{{ .Prompt }}`): the system prompt is dropped, and so are Ollama's default sampling settings. For
     example, Gemma uses temperature 1, top_k 64 and top_p 0.95; a bare Modelfile gets Ollama's 0.8.
   - gemma3's fine-tuned model got Unsloth's template, which is different again.
2. **`bin/run_all.sh` hard-codes everything.** It hard-codes `config/config.yaml`,
   `results/evaluations.jsonl`, `jobs/db/jobs.db` and `cd /home/rabrew/Desktop/chess-llm-bench`, so it
   cannot run a second configuration.
3. **Its setup step can break the training environment.** It runs `pip install -r requirements.txt`,
   and that file installs **unpinned `unsloth` from git HEAD**.

## Inputs / Outputs

- **`config/config_finetuned.yaml`:** identical to `config/config.yaml` except for:
  - `models`: the six `-board-ft` tags
  - `paths.jobs_db`: `jobs/db/jobs_ft.db`
  - `paths.results_file`: `results/finetuning_experiment/evaluations_ft.jsonl`
  - `paths.logs_dir`: `results/finetuning_experiment/logs`
  - `paths.plots_dir` and `paths.metrics_dir`: `results/finetuning_experiment/plots` and `/metrics`
- **Outputs:**
  - `evaluations_ft.jsonl`, the same record schema as the base results
  - `evaluations_ft_retried.jsonl`, from the illegal-move retry
  - CPL filled in place by `enrich_cpl.py --results`

## Steps / Logic

1. **Matching Modelfiles.** `import_finetuned_to_ollama.py` builds the fine-tuned Modelfile from
   `ollama show <base> --modelfile`, via `build_ft_modelfile(base_text, gguf_path)`:
   - Every top-level `FROM` line is replaced by a single `FROM <ft gguf>`.
   - Everything else is kept unchanged (`TEMPLATE`, `SYSTEM`, `PARAMETER`, `RENDERER`, `PARSER`,
     `LICENSE`), including `FROM`-looking lines inside `"""` blocks.
   - `num_ctx` and `num_predict` are no longer added, because the benchmark and board test pass them per
     request.
   - All six batch-1 models are re-imported. The weights are unchanged, so no retraining is needed.
     `verify_export.py` is re-run on each to confirm the board test still passes, because raw mode should
     ignore the template.
2. **Config-driven launcher.** `bin/run_all.sh`:
   - takes `BENCH_CONFIG` (default `config/config.yaml`) and reads the results file and jobs DB from it;
   - `cd`s to its own repo instead of a fixed path;
   - skips the pip install when `SKIP_INSTALL=1`;
   - passes the paths to `retry_illegal_moves.py --input/--output` and to
     `enrich_cpl.py --results --config`.

   With no environment variables set, its behaviour on the base config is unchanged.
3. **Pre-flight check** (`scripts/check_ft_run.py`). Runs before any job and exits non-zero on failure.
   - **Config:** `config_finetuned.yaml` equals `config.yaml` on every key except `models` and `paths`.
   - **Tags:** every fine-tuned tag is a registered `-board-ft` model whose base is in `config.yaml`.
   - **Positions:** for each tier, every FEN the base model has in `results/evaluations.jsonl` is in the
     sample from `DataLoader.sample(seed 42, 1000)`. If not, the sampling differs and the run fails.
     Sampled positions the base run never completed are printed as **warnings**. On 2026-10-22 these
     were: deepseek-r1:7b easy 170, llama3.2:3b medium 9 and hard 1, qwen2.5:7b extreme 1. Base versus
     fine-tuned comparisons must use only the positions both sides answered (paired).
   - **Modelfiles:** each fine-tuned Modelfile equals its base's once the `FROM` lines are removed and the
     directives are sorted. Ollama may reorder `PARAMETER` lines when it saves a model.
   - **Results file:** the fine-tuned results file has no rows for a non-`-board-ft` model.
4. **Run.** `BENCH_CONFIG=config/config_finetuned.yaml SKIP_INSTALL=1 bin/run_all.sh`, in tmux session
   `ftbench`. The per-model `NUM_PARALLEL` and worker counts match on the base name inside the tag
   (`deepseek-r1`, `e2b`, `7b`…), so each fine-tuned model gets its base's settings.

## Edge cases

| Case | Behaviour |
|---|---|
| Base Modelfile has no `FROM` | `ValueError`. Something is wrong with the base. |
| Base Modelfile has two `FROM` lines (model plus projector) | Both are replaced by the one fine-tuned GGUF. The board benchmark is text only. |
| A `FROM` inside a `"""` template block | Kept |
| Base not installed in Ollama | `import` fails with the `ollama show` error |
| Positions differ from the base run | The pre-flight check fails and the run does not start |
| Run interrupted | `run_all.sh` resumes per model and tier from the results count, as for the base run |
| Ollama restart by `run_all.sh` | Same as the base run: it kills `ollama serve` and starts its own with tuned `NUM_PARALLEL` |

## Dependencies

- Ollama 0.20.
- Lc0 for CPL (config `lc0` section, same nodes as the base run).
- No new Python packages.

## Reproducibility

- Positions are seed 42, 1,000 per tier.
- Each model's Ollama digest is in `ollama list` at run time.
- The run is logged under `results/finetuning_experiment/logs`.
- The base and fine-tuned answers are not deterministic, because of the default temperature (M-3). This
  is reported, and both sides were sampled the same way.

## Estimated time

From the base runs' active hours: llama3.2 2.0, gemma3 2.5, qwen 1.7, mistral 2.5, deepseek 4.2 and
gemma4 1.3. That is about 14 h, plus the retry and CPL enrichment.

## Base re-run on current code (added 2026-10-23, issues-log Issue 18)

**Why:** the base records for the five batch-1 models (March and April) predate the move rescue, so base
T2 legality is first-attempt while fine-tuned T2 legality is after rescue. The claim above, that base and
fine-tuned share the same illegal-move retry, was wrong, and F-4 is not met. Re-running the bases on the
current code gives both sides:
- the same worker, rescue, prompts and Modelfiles
- stored raw replies (`raw_response`)

**Config:** `config/config_base_rerun.yaml` is identical to `config/config.yaml` except for:
- `models`: `llama3.2:3b`, `gemma3:4b`, `qwen2.5:7b`, `mistral:7b`, `deepseek-r1:7b` (the bases of the
  fine-tuned tags)
- `paths`: `jobs/db/jobs_base_rerun.db` and
  `results/finetuning_experiment/evaluations_base_rerun.jsonl`, with logs, plots and metrics under
  `results/finetuning_experiment/base_rerun/`

`results/evaluations.jsonl` is never written. The original base records stay as the published 22-model
benchmark. The fine-tuning comparison uses
`compare_finetuning_results.py --base results/finetuning_experiment/evaluations_base_rerun.jsonl`.

**Run:** `BENCH_CONFIG=config/config_base_rerun.yaml SKIP_INSTALL=1 bin/run_all.sh`. It is the same size
as the fine-tuned run: 120k jobs, about 8 h, plus retry and CPL enrichment (about 10 h).

**Tests** (`tests/test_ft_run_setup.py`):
- The re-run config matches the base config except for `models` and `paths`
- Its models are exactly the bases of the batch-1 fine-tuned tags
- Every output path differs from both the base and the fine-tuned configs, and `data_dir` is shared
