#!/bin/bash
# Train, import and verify each model in turn. See fine-tuning/specs/gguf-export-check.md.
# Usage: fine-tuning/scripts/run_batch.sh gemma4:e4b llama3.1:8b ...
cd "$(dirname "$0")/../.." || exit 1
source venv/bin/activate
failed=()
for tag in "$@"; do
    echo "=== Starting $tag ==="
    if python fine-tuning/scripts/finetune_model.py "$tag" \
        && python fine-tuning/scripts/import_finetuned_to_ollama.py "$tag" \
        && python fine-tuning/scripts/verify_export.py "$tag"; then
        echo "=== $tag: DONE ==="
    else
        echo "=== $tag: FAILED ===" >&2
        failed+=("$tag")
    fi
done
if [ ${#failed[@]} -gt 0 ]; then
    echo "=== FAILED MODELS: ${failed[*]} ==="
else
    echo "=== All models complete ==="
fi
