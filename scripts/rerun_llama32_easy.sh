#!/usr/bin/env bash
# Waits for mistral:7b medium to finish, pauses the main pipeline, re-runs
# llama3.2:3b easy exclusively, then resumes the main pipeline.
#
# Launched automatically in chess-bench:rerun-easy tmux window.

set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

RESULTS="results/evaluations.jsonl"
TEMP_CONFIG="/tmp/chess_bench_rerun_config.yaml"
TEMP_DB="/tmp/chess_bench_rerun_jobs.db"
PIPELINE_SESSION="chess_bench"
MODEL="llama3.2:3b"
TIER="easy"
WORKERS=6
MIN_MISTRAL_MEDIUM=5900   # treat as done when within 100 of 6000

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
RED="\033[0;31m"
NC="\033[0m"

pipeline_pgid=""

cleanup() {
    # If we exit early, make sure the pipeline is never left frozen
    if [ -n "${pipeline_pgid}" ]; then
        echo -e "\n${RED}  Interrupted — resuming pipeline (PGID ${pipeline_pgid})...${NC}"
        kill -CONT -"${pipeline_pgid}" 2>/dev/null || true
    fi
    rm -f "${TEMP_CONFIG}" "${TEMP_DB}" "${TEMP_DB}-shm" "${TEMP_DB}-wal"
}
trap cleanup EXIT INT TERM

# ── Get the process group of the main pipeline ──────────────────────────────
get_pipeline_pgid() {
    local pane_pid
    pane_pid=$(tmux list-panes -t "${PIPELINE_SESSION}" -F "#{pane_pid}" 2>/dev/null | head -1)
    [ -z "${pane_pid}" ] && return 1

    # Walk down to the deepest child (the actual python/bash worker)
    local pid="${pane_pid}"
    while true; do
        local child
        child=$(pgrep -P "${pid}" 2>/dev/null | head -1 || true)
        [ -z "${child}" ] && break
        pid="${child}"
    done

    # Return the process group ID of the root pane process
    ps -o pgid= -p "${pane_pid}" 2>/dev/null | tr -d ' '
}

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "  llama3.2:3b easy re-run watcher"
echo -e "  Polling for mistral:7b medium >= ${MIN_MISTRAL_MEDIUM} records..."
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# ── Wait for mistral:7b medium to finish ────────────────────────────────────
while true; do
    count=$(python3 -c "
import json
n = 0
try:
    with open('${RESULTS}') as f:
        for line in f:
            r = json.loads(line)
            if r.get('model') == 'mistral:7b' and r.get('difficulty') == 'medium':
                n += 1
except FileNotFoundError:
    pass
print(n)
" 2>/dev/null)

    echo -ne "\r  mistral:7b medium: ${count} / ${MIN_MISTRAL_MEDIUM}   "

    if [ "${count}" -ge "${MIN_MISTRAL_MEDIUM}" ]; then
        echo -e "\n\n${GREEN}  → mistral:7b medium done.${NC}"
        break
    fi
    sleep 30
done

# ── Pause the main pipeline ──────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}  [1/5] Pausing main pipeline (${PIPELINE_SESSION})...${NC}"

pipeline_pgid=$(get_pipeline_pgid) || {
    echo -e "${RED}  ERROR: could not find pipeline process. Aborting.${NC}"
    exit 1
}

echo -e "  Pipeline PGID: ${pipeline_pgid}"
kill -STOP -"${pipeline_pgid}"
echo -e "  ${GREEN}Pipeline paused.${NC}"
sleep 2  # let any in-flight writes settle

# ── Strip stale records ──────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}  [2/5] Stripping stale llama3.2:3b easy records...${NC}"

python3 - <<'PYEOF'
import json

results = "results/evaluations.jsonl"
tmpfile  = results + ".rerun_strip.tmp"

kept = removed = 0
with open(results) as fin, open(tmpfile, "w") as fout:
    for line in fin:
        r = json.loads(line)
        if r.get("model") == "llama3.2:3b" and r.get("difficulty") == "easy":
            removed += 1
        else:
            fout.write(line)
            kept += 1

import os
os.replace(tmpfile, results)
print(f"  Removed {removed} stale records. {kept} records remain.")
PYEOF

# ── Write isolated temp config ───────────────────────────────────────────────
echo ""
echo -e "${YELLOW}  [3/5] Writing isolated config (separate jobs DB)...${NC}"

python3 - <<PYEOF
import yaml

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["paths"]["jobs_db"] = "${TEMP_DB}"
cfg["benchmark"]["tiers"] = ["${TIER}"]

with open("${TEMP_CONFIG}", "w") as f:
    yaml.dump(cfg, f)

print("  Written: ${TEMP_CONFIG}")
PYEOF

# ── Generate jobs ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}  [4/5] Generating jobs for ${MODEL} / ${TIER}...${NC}"
python scripts/generate_jobs.py \
    --config "${TEMP_CONFIG}" \
    --tier "${TIER}" \
    --model "${MODEL}"

# ── Run workers ──────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}  [5/5] Running ${WORKERS} workers (exclusive)...${NC}"
python scripts/run_workers.py \
    --config "${TEMP_CONFIG}" \
    --workers "${WORKERS}" \
    --model "${MODEL}"

# ── Resume the pipeline ──────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}  Resuming main pipeline (PGID ${pipeline_pgid})...${NC}"
kill -CONT -"${pipeline_pgid}"
pipeline_pgid=""  # disarm the trap
echo -e "  ${GREEN}Pipeline resumed.${NC}"

# ── Cleanup ──────────────────────────────────────────────────────────────────
rm -f "${TEMP_CONFIG}" "${TEMP_DB}" "${TEMP_DB}-shm" "${TEMP_DB}-wal"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  llama3.2:3b easy re-run complete.${NC}"
NEW=$(python3 -c "
import json
n = sum(1 for line in open('${RESULTS}')
        if json.loads(line).get('model')=='llama3.2:3b' and json.loads(line).get('difficulty')=='easy')
print(n)
" 2>/dev/null)
echo -e "  New llama3.2:3b easy records: ${NEW}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
