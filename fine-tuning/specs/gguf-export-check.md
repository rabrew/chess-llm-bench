# Spec: GGUF Export Check and Manual Re-export

**Closes:** the broken gemma4:e2b-board-ft export (2026-10-22)
**Created:** 2026-10-22

## Goal

gemma4:e2b trained correctly: held-out recall was 0.974 after epoch 1 and 0.990 after epoch 2. Loading
`checkpoint-2476` with its LoRA adapter in HuggingFace also lists the board correctly. But the GGUF that
Unsloth's `save_pretrained_gguf` exported behaves like the **untrained** model. In Ollama, and in
llama.cpp directly, it opens with `<|channel>thought` "Thinking Process…" and reads a different FEN. As a
result, 83.5% of its board-test answers are unparsed and its recall is 0.149.

The prompt and template were checked and are not the cause:

- Training fed plain Alpaca text with no BOS token (first ids `10354, 64017…` = `### Instruction`).
- Raw mode in Ollama also sends no BOS.

So the LoRA weights were lost in Unsloth's merge step for Gemma 4, and nothing caught it.

This spec adds two things:

1. **`verify_export.py`.** An automatic check, run after every import, that the Ollama model actually
   learned the task. A broken export now fails the batch instead of reaching the benchmark.
2. **`reexport_gguf.py`.** A way to rebuild a GGUF from a saved adapter without Unsloth's merge, so
   gemma4:e2b does not need retraining (about 4 hours).

## Inputs / Outputs

### verify_export.py <ollama_base> [--config] [--host] [--n 20]

- **Input:** the fine-tuned tag from config, and the first `n` positions of the held-out set
  (`split_held_out(seed=42, n=200)`, the same set as the F-9 board test).
- **Process:** each position is sent with the training-style prompt from `board_reading_test`, in raw
  mode at temperature 0.
- **Output:** a log line with mean recall and unparsed rate (and legal-move recall and precision for
  `board+legal`). Exit code 0 if mean recall ≥ `MIN_EXPORT_RECALL` (0.80), and for `board+legal`
  the legal-move check passes too; otherwise 1.
- 0.80 sits well below the weakest working fine-tuned model under this prompt (gemma3, 0.965 recall),
  and well above an untrained model (≈0.0 to 0.15).

### reexport_gguf.py <ollama_base> [--checkpoint DIR] [--config]

- **Inputs:**
  - The adapter is the last `checkpoint-*` in `<checkpoints_dir>/<ft_tag>/`, or `--checkpoint`.
  - The base is the full-precision version of the adapter's `base_model_name_or_path`:
    `-unsloth-bnb-4bit` or `-bnb-4bit` is stripped, so `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` becomes
    `unsloth/gemma-4-E2B-it`.
- **Work dir:** `paths.merge_work_dir` from config, on the shared drive because the main drive has about
  18 GB free. It holds the merged bf16 HF model and the bf16 GGUF, and is deleted at the end.
- **Output:** `<gguf_output_dir>/<ft_tag>_gguf/<name>.<QUANT>.gguf`, replacing the broken file. The
  mmproj file is left alone because the vision tower was not trained.
- **Next steps:** then run `import_finetuned_to_ollama.py` and `verify_export.py`.

## Steps / Logic

### reexport_gguf

1. Resolve the checkpoint and the full-precision base ID. Fail if the checkpoint has no
   `adapter_config.json`.
2. Load the base in bf16 on the CPU. The class comes from the base config's `architectures[0]`, so
   Gemma 4 gets its multimodal class and the module names match the adapter's.
3. Apply the adapter with PEFT, and **check it is really attached**:
   - the number of LoRA layers is greater than 0;
   - at least one `lora_B` weight is non-zero (B starts at zero, so all-zero means untrained).
4. Store one targeted base weight, run `merge_and_unload()`, and check that weight changed. If it did
   not, raise. This is the check Unsloth's export lacked.
5. Save the merged model and copy the tokenizer and processor files from the checkpoint.
6. Run llama.cpp `convert_hf_to_gguf.py --outtype bf16`, then `llama-quantize <QUANT>`. Each subprocess
   failure raises with its stderr.
7. Move the quantized file into the `_gguf` dir, then delete the work dir.

### verify_export

1. Load the config, find the entry, and take the first `n` held-out positions.
2. **Detect the task** from the dataset (`detect_task`): `board+legal` when the training targets
   carry a `Legal moves:` line, otherwise `board`. The prompt and `num_predict` follow the task, so a
   legal-moves model is asked for its legal moves.
3. Query Ollama for each position and score it with `verify_board_description`, and for
   `board+legal` also with `verify_legal_moves`.
4. `export_passes(recalls, threshold)` returns True or False, and raises `ValueError` on an empty list.
5. For `board+legal`, the export also needs `legal_passes(recalls, precisions, MIN_EXPORT_LEGAL)`:
   mean legal-move recall **and** precision ≥ 0.9.

#### Legal-move check (added 2026-10-29)

The legal-moves models passed this check on the board alone, and the check sent them the
pieces-only prompt. Their legal-move lists were only 65–81% complete and 52–60% correct, which was
found after a 12-hour benchmark (FINDINGS section F). `MIN_EXPORT_LEGAL` is 0.9, the same bar as
criterion 0 in `legal-moves-finetuning.md`, so a model that would fail criterion 0 now fails the batch
before the benchmark. Under this check all five 2026-10 legal-moves exports would fail.

### Batch pipeline

`fine-tuning/scripts/run_batch.sh` is the committed version of `/tmp/run_batch2.sh`. For each tag it
runs train, import, verify. A failed verify marks the model FAILED.

## Edge cases

| Case | Behaviour |
|---|---|
| No checkpoint folder, or no `checkpoint-*` | `FileNotFoundError` naming the folder |
| Adapter base already full precision | ID used unchanged |
| Adapter has 0 LoRA layers, all-zero `lora_B`, or merge leaves weights unchanged | `RuntimeError`, nothing written to the `_gguf` dir |
| Work dir already exists | `FileExistsError`, so an earlier half-done run is not reused by accident |
| `convert` or `quantize` fails | `RuntimeError` with stderr. The work dir is kept for inspection. |
| Ollama unreachable during verify | the `query_ollama` retries, then the error propagates and the exit is non-zero |
| `n` larger than the held-out set | capped at the held-out size |
| Empty dataset when detecting the task | `ValueError` |
| Model gives no `Legal moves:` line | recall and precision 0 for that position |

## Dependencies

- peft 0.21, transformers 5.17 (already in the venv).
- llama.cpp at `paths.llama_cpp_dir` (Unsloth's build at `~/.unsloth/llama.cpp`).
- Ollama for verify.

## Reproducibility

- Same seed-42 held-out split.
- Temperature 0, seed 42, raw prompts.
- The merge is deterministic from the adapter and base weights.
- The re-exported GGUF's Ollama digest is logged by the board test as usual.
