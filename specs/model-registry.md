# Spec: Explicit Model Registry

**Closes:** D3 in [metric-defect-remediation.md](metric-defect-remediation.md), and M-4 in [exhibition-readiness.md](exhibition-readiness.md)
**Created:** 2026-10-20

## Goal

`parse_model_info` guesses a model's family from a 5-entry prefix map and its size by
stripping "b" from the tag. As a result:

- 8 of 22 models are labelled family `unknown`
- `gemma4:e2b`, `gemma4:e4b` and `mixtral:8x7b` parse as 0B
- the family table merges different models into one row: codellama + yi, deepseek-r1:7b + wizardlm2:7b, and gemma4:e2b + e4b

The family and size were also **stored in every result record** at collection time, so
the stored values are wrong too.

Replace the guessing with an explicit registry that fails loudly on an unknown tag, and
recompute family and size from the tag when loading results.

## Inputs / Outputs

- **Input:** an Ollama tag such as `"gemma4:e4b"`, optionally with the `-board-ft` suffix
  used for fine-tuned models.
- **Output:** the same dict shape as now, plus three new keys:
  `{family, size_str, size_b, full_tag, active_b, is_moe, fine_tuned}`.
- `load_results_df` overwrites the stored `model_family` and `model_size_b` with
  registry values. The raw JSONL is never modified.

## Steps / Logic

1. `MODEL_REGISTRY` in `src/utils.py` maps each exact base tag to (family, total size,
   active size). The family is the model **generation**, so that different generations
   never share a row:

   | Family | Tags |
   |---|---|
   | `llama3` | llama3.2:3b, llama3.1:8b, llama3.3:70b |
   | `gemma3` | gemma3:4b, gemma3:12b |
   | `gemma4` | gemma4:e2b (2B effective), gemma4:e4b (4B effective), gemma4:26b, gemma4:31b |
   | `qwen2.5` | qwen2.5:7b, 14b, 32b, 72b |
   | `deepseek-r1` | deepseek-r1:7b, deepseek-r1:14b |
   | single-model families | mistral:7b, wizardlm2:7b, phi4:14b, solar:10.7b, codellama:34b, yi:34b, command-r:35b |
   | `mixtral` (MoE) | mixtral:8x7b: 46.7B total, 12.9B active, `is_moe=True` |

   gemma4:26b is also MoE: 26B total, 4B active (HuggingFace name `gemma-4-26B-A4B`, added 2026-10-21).

2. `parse_model_info` strips `-board-ft` (setting `fine_tuned=True`), then looks up the
   base tag. An unknown tag raises `ValueError` naming the tag and the registry location.
3. `load_results_df` recomputes `model_family` and `model_size_b` from `model` for every
   row. A tag not in the registry raises, so a mislabelled row is never silently kept.
4. `aggregate_by_model_family` asserts that each output row comes from exactly one model.
   This stops any future merge.

## Edge cases

| Case | Behaviour |
|---|---|
| Unknown tag (a new model, or a typo) | `ValueError`. Add the model to the registry. |
| Fine-tuned tag `X-board-ft` | Same family and size as X, with `fine_tuned=True` |
| MoE model | `size_b` = total parameters, `active_b` = active parameters. Scaling analyses should leave MoE models out. |
| Results with no `model` column (synthetic test frames) | `load_results_df` leaves family and size alone |

## Dependencies

None new.

## Verification

- Every tag in `config/config.yaml` resolves to a known family with a non-zero size
- `gemma4:e2b` gives 2, `gemma4:e4b` gives 4, and `mixtral:8x7b` gives 46.7 total, 12.9 active, MoE
- `codellama:34b` is not in the `llama3` family
- An unknown tag raises `ValueError`
- A `-board-ft` tag resolves like its base and sets `fine_tuned`
- `load_results_df` replaces wrong stored family and size values
- The family table has one model per row
