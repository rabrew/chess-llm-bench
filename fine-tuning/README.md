# Fine-tuning

Does teaching a model to read the board change how it plays? Five models (llama3.2:3b, gemma3:4b,
qwen2.5:7b, mistral:7b, deepseek-r1:7b) were trained with LoRA on 20,000 positions, none from the
benchmark, then run through the same benchmark as the base models. Results: FINDINGS, section F.

## The two experiments

| Version | Target the model learns | Spec | Result |
|---|---|---|---|
| Pieces only | List every piece and its square | [specs/chess-board-finetuning.md](specs/chess-board-finetuning.md) | Board reading rose from 1–19% to 89–99%; nothing else improved |
| Pieces + legal moves | The piece list, then every legal move | [specs/legal-moves-finetuning.md](specs/legal-moves-finetuning.md) | Failed its own pre-registered training check, so no transfer conclusion |

## Folders

| Path | What's there |
|---|---|
| `specs/` | The two experiment specs, plus the board-reading test, GGUF export check and Gemma fixes |
| `config/` | Training settings (`finetune_config.yaml`, `finetune_config_legal.yaml`) |
| `src/board_state.py` | Turns a position into the training target (piece list, legal moves) |
| `scripts/generate_board_dataset.py` | Builds the training data, excluding benchmark and held-out positions |
| `scripts/finetune_model.py` | Trains one model; writes a record to `../runs/` |
| `scripts/verify_export.py`, `reexport_gguf.py` | Checks the exported model before it's benchmarked |
| `scripts/import_finetuned_to_ollama.py` | Loads the fine-tuned model into Ollama with the base model's settings |
| `scripts/board_reading_test.py` | Measures board reading before and after training |
| `scripts/compare_finetuning_results.py` | Pairs base and fine-tuned answers on the same position and prompt |
| `scripts/run_batch.sh` | Runs train → export → check → import for a batch |
| `tests/` | Tests for all of the above, including that held-out positions never enter training |

Training records (config, git commit, seed, data hash, metrics) are in [`../runs/`](../runs/).
The results are in [`../results/finetuning_experiment/`](../results/finetuning_experiment/).
