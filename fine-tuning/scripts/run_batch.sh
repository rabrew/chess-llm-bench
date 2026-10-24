#!/bin/bash
# Train, import and verify each model in turn. See fine-tuning/specs/gguf-export-check.md.
# Usage: fine-tuning/scripts/run_batch.sh gemma4:e4b llama3.1:8b ...
#        FT_CONFIG=fine-tuning/config/finetune_config_legal.yaml fine-tuning/scripts/run_batch.sh qwen2.5:7b
cd "$(dirname "$0")/../.." || exit 1
FT_CONFIG="${FT_CONFIG:-fine-tuning/config/finetune_config.yaml}"
echo "Fine-tuning config: $FT_CONFIG"
source venv/bin/activate
failed=()
for tag in "$@"; do
    echo "=== Starting $tag ==="
    if python fine-tuning/scripts/finetune_model.py "$tag" --config "$FT_CONFIG" \
        && python fine-tuning/scripts/import_finetuned_to_ollama.py "$tag" --config "$FT_CONFIG" \
        && python fine-tuning/scripts/verify_export.py "$tag" --config "$FT_CONFIG"; then
        echo "=== $tag: DONE ==="
    else
        echo "=== $tag: FAILED ===" >&2
        failed+=("$tag")
    fi
done
if [ ${#failed[@]} -gt 0 ]; then
    echo "=== FAILED MODELS: ${failed[*]} ==="
    exit 1
else
    echo "=== All models complete ==="
fi
