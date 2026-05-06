#!/bin/bash
# Persistent wrapper — restarts run_all.sh automatically if it crashes.
# Only stops when run_all.sh exits 0 (fully complete).

cd /home/rabrew/Desktop/chess-llm-bench
source venv/bin/activate 2>/dev/null || true

LOG="results/logs/run_forever.log"
mkdir -p results/logs

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

log "=== run_forever.sh started (PID $$) ==="

ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    log "--- Attempt $ATTEMPT: launching run_all.sh ---"

    bash bin/run_all.sh
    EXIT_CODE=$?

    log "--- run_all.sh exited with code $EXIT_CODE ---"

    if [ $EXIT_CODE -eq 0 ]; then
        log "=== Benchmark complete! run_all.sh exited 0. ==="
        break
    fi

    log "run_all.sh crashed (exit $EXIT_CODE). Waiting 15s before retry..."
    sleep 15
done

log "=== run_forever.sh exiting ==="
