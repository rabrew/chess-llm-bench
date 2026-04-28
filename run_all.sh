#!/bin/bash
set -eo pipefail
# Exit on error is intentional for setup steps (venv, dataset build).
# Worker runs use explicit retry loops and '|| true' to survive Ollama crashes.

cd /home/rabrew/Desktop/chess-llm-bench

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶${NC} ${BOLD}$1${NC}"
}

print_info() {
    echo -e "  ${YELLOW}→${NC} $1"
}

print_header "CHESS LLM BENCHMARK"
echo -e "  Started: $(date)"
echo -e "  Host:    $(hostname)"
echo -e "  CPUs:    $(nproc) cores"
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    print_step "[0/7] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

print_step "[1/7] Installing dependencies..."
pip install -r requirements.txt -q
print_info "Done"

print_header "STEP 2/7: BUILDING DATASET"
# Skip if dataset already exists with stockfish evals
if python3 -c "import json; d=json.load(open('data/easy.json')); exit(0 if 'stockfish_eval' in d[0] else 1)" 2>/dev/null; then
    print_info "Dataset with Stockfish evals already exists - SKIPPING rebuild"
    print_info "Delete data/*.json to force rebuild"
else
    print_info "Loading 5.8M puzzles from Lichess database..."
    print_info "This will validate positions using $(($(nproc))) CPU cores"
    echo ""
    python scripts/build_dataset.py
fi

print_header "STEP 3/7: LC0 GPU EVALUATION"
print_info "SKIPPED — lc0_eval unused by evaluator; stockfish_eval+stockfish_best_move already present in dataset"

print_header "STEP 4/7: PULLING OLLAMA MODELS"
CONFIGURED_MODELS=$(python3 -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print('\n'.join(c['models']))" 2>/dev/null)
AVAILABLE_MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import json,sys; print('\n'.join(m['name'] for m in json.load(sys.stdin).get('models',[])))" 2>/dev/null)
MISSING=$(comm -23 <(echo "$CONFIGURED_MODELS" | sort) <(echo "$AVAILABLE_MODELS" | sort))
if [ -z "$MISSING" ]; then
    print_info "All models already available - SKIPPING pull"
    print_info "Delete a model with 'ollama rm <model>' to force re-pull"
else
    print_info "Pulling missing models..."
    echo ""
    python scripts/pull_models.py
fi

print_header "STEPS 5+6/7: GENERATING JOBS AND RUNNING BENCHMARK (MODEL BY MODEL, ALL TIERS)"
print_info "Processing all tiers for one model at a time to maximise RAM usage..."
MODELS=$(python3 -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print('\n'.join(c['models']))" 2>/dev/null)
echo ""

# Restart Ollama with NUM_PARALLEL tuned to the model's VRAM footprint.
# Small models (≤14GB) fill in VRAM → high parallelism.
# Large/offloaded models → lower parallelism (CPU can't batch well).
CURRENT_OLLAMA_PARALLEL=""
OLLAMA_BG_PID=""

cleanup_ollama() {
    if [ -n "$OLLAMA_BG_PID" ] && kill -0 "$OLLAMA_BG_PID" 2>/dev/null; then
        kill "$OLLAMA_BG_PID" 2>/dev/null || true
    fi
}
trap cleanup_ollama EXIT

restart_ollama_for_model() {
    local model="$1"
    local num_parallel workers
    if echo "$model" | grep -qiE "72b"; then
        num_parallel=1; workers=1       # ~40GB+ — OOM at workers=2
    elif echo "$model" | grep -qiE "70b"; then
        num_parallel=1; workers=2       # ~40GB+ — barely fits offloaded
    elif echo "$model" | grep -qiE "mixtral"; then
        num_parallel=2; workers=3       # MoE, large active params
    elif echo "$model" | grep -qiE "32b|34b|35b|26b|31b"; then
        num_parallel=2; workers=4       # ~14-20GB, 2 fits in 16GB VRAM
    elif echo "$model" | grep -qiE "deepseek-r1:14b"; then
        num_parallel=2; workers=4       # 14B + long CoT + KV cache — tight on 16GB VRAM
    elif echo "$model" | grep -qiE "deepseek-r1"; then
        num_parallel=4; workers=8       # 7B + CoT fits easily; bumped from 2/4 to speed up
    elif echo "$model" | grep -qiE "e2b|e3b"; then
        num_parallel=6; workers=12      # ~2-3B, very small — push parallelism hard
    elif echo "$model" | grep -qiE "12b|13b|14b"; then
        num_parallel=2; workers=4       # ~7-8GB each, 2 fits safely in 16GB
    elif echo "$model" | grep -qiE "7b|8b|9b|10b|11b"; then
        num_parallel=3; workers=6       # ~4-5GB each, 3 fits in 16GB
    elif echo "$model" | grep -qiE "gemma4"; then
        num_parallel=2; workers=4       # gemma4 variants — default to medium settings
    else
        num_parallel=12; workers=16     # small (≤6B), fits many in VRAM
    fi

    echo "$workers" > /tmp/bench_workers
    if [ "$num_parallel" = "$CURRENT_OLLAMA_PARALLEL" ]; then
        return  # already correct, no restart needed
    fi

    print_info "Restarting Ollama: NUM_PARALLEL=$num_parallel for $model..."
    # Stop any running Ollama (our background process or a pre-existing one)
    if [ -n "$OLLAMA_BG_PID" ] && kill -0 "$OLLAMA_BG_PID" 2>/dev/null; then
        kill "$OLLAMA_BG_PID" 2>/dev/null || true
        wait "$OLLAMA_BG_PID" 2>/dev/null || true
    fi
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 3

    # Start Ollama in background, log to file
    OLLAMA_NUM_PARALLEL=$num_parallel \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_MODELS=/mnt/shared/ollama_models \
        ollama serve >> /tmp/ollama_serve.log 2>&1 &
    OLLAMA_BG_PID=$!

    until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 2; done
    CURRENT_OLLAMA_PARALLEL="$num_parallel"
    print_info "Ollama ready (NUM_PARALLEL=$num_parallel, workers=$workers, PID=$OLLAMA_BG_PID)"
}

RESULTS_FILE="results/evaluations.jsonl"
JOBS_PER_TIER=5900  # 6000 jobs per tier; skip if >= this many results already saved
SKIPPED_TIERS="jobs/db/skipped_tiers.txt"  # persistent record of tiers abandoned due to exhausted queue

tier_done() {
    local model="$1" tier="$2"
    # Treat as done if previously marked exhausted (queue ran dry, can't reach threshold)
    if grep -qxF "${model}::${tier}" "$SKIPPED_TIERS" 2>/dev/null; then
        return 0
    fi
    local count
    count=$(python3 - "$model" "$tier" <<'PYEOF' 2>/dev/null || echo 0
import json, sys
model, tier = sys.argv[1], sys.argv[2]
results_file = "results/evaluations.jsonl"
n = 0
try:
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("model") == model and r.get("difficulty") == tier:
                    n += 1
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    pass
print(n)
PYEOF
)
    [ "$count" -ge "$JOBS_PER_TIER" ]
}

while IFS= read -r model; do
    [ -z "$model" ] && continue
    print_header "MODEL: $model"

    # Tune Ollama parallelism for this model before running any tiers
    restart_ollama_for_model "$model"
    WORKERS=$(cat /tmp/bench_workers 2>/dev/null || echo '4')
    print_info "Workers: $WORKERS"

    for tier in easy medium hard extreme; do
        if tier_done "$model" "$tier"; then
            print_info "Model: $model | Tier: $tier — already complete, SKIPPING"
            echo ""
            continue
        fi

        print_info "Model: $model | Tier: $tier — generating jobs..."
        python scripts/generate_jobs.py --tier "$tier" --model "$model" || true

        print_info "Running workers for $model ($tier)..."
        until tier_done "$model" "$tier"; do
            # Reset stale in_progress jobs left by any previous crash
            python3 - <<'PYEOF' 2>/dev/null || true
import sys, yaml
sys.path.insert(0, '.')
from src.job_queue import JobQueue
config = yaml.safe_load(open('config/config.yaml'))
jq = JobQueue(config['paths']['jobs_db'])
n = jq.reset_stale_jobs(0)
if n:
    print(f"  → Reset {n} stale in_progress jobs to pending")
PYEOF
            # Wait for Ollama to be available before running workers
            until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
                echo "  → Waiting for Ollama to come back up..."
                sleep 15
            done
            python scripts/run_workers.py --workers "$WORKERS" --model "$model" || true

            # If no pending jobs remain but tier still not done, nothing left to process — move on
            PENDING=$(python3 -c "
import sqlite3, sys
try:
    conn = sqlite3.connect('jobs/db/jobs.db', timeout=3)
    cur  = conn.cursor()
    cur.execute(\"SELECT COUNT(*) FROM jobs WHERE model=? AND status IN ('pending','in_progress')\", ('$model',))
    print(cur.fetchone()[0])
    conn.close()
except Exception:
    print(0)
" 2>/dev/null || echo 0)
            if [ "$PENDING" -eq 0 ]; then
                print_info "No pending jobs remain for $model/$tier (failures exhausted queue) — skipping"
                mkdir -p "$(dirname "$SKIPPED_TIERS")"
                echo "${model}::${tier}" >> "$SKIPPED_TIERS"
                break
            fi
        done

        print_info "$model/$tier complete. Wiping job DB..."
        DB_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print(c['paths']['jobs_db'])" 2>/dev/null || echo "jobs/db/jobs.db")
        rm -f "$DB_PATH" "${DB_PATH}-shm" "${DB_PATH}-wal"
        echo ""
    done
done <<< "$MODELS"

print_header "STEP 7/7: RETRYING ILLEGAL MOVES"
print_info "Re-prompting models with legal move list for all illegal/missing moves..."
echo ""
python scripts/retry_illegal_moves.py
echo ""

print_header "STEP 8/9: ENRICHING CPL (Lc0 GPU)"
print_info "Computing centipawn loss for all legal moves..."
echo ""
python scripts/enrich_cpl.py
echo ""

print_header "STEP 9/9: GENERATING RESULTS"
print_info "Creating plots and metrics..."
echo ""
python scripts/generate_plots.py --save-metrics

print_header "COMPLETE!"
echo -e "  Finished: $(date)"
echo ""
echo -e "  ${GREEN}Results:${NC}       results/evaluations.jsonl"
echo -e "  ${GREEN}Retried moves:${NC} results/evaluations_retried.jsonl"
echo -e "  ${GREEN}Plots:${NC}         results/plots/"
echo -e "  ${GREEN}Metrics:${NC}       results/metrics/"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
