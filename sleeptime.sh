#!/bin/bash
# sleeptime.sh — pause the benchmark for a given duration, then resume automatically.
#
# Usage:
#   ./sleeptime.sh 2h      # pause for 2 hours
#   ./sleeptime.sh 90m     # pause for 90 minutes
#   ./sleeptime.sh 30      # pause for 30 minutes (bare number = minutes)
#
# How it works:
#   Kills ollama → workers fail and exit → run_all.sh waits for ollama to return
#   After the sleep, restarts ollama → run_all.sh resumes automatically.

DURATION=${1:-2h}

# Parse duration to seconds
case "$DURATION" in
    *h) SLEEP_SECS=$(( ${DURATION%h} * 3600 )) ;;
    *m) SLEEP_SECS=$(( ${DURATION%m} * 60 )) ;;
    *s) SLEEP_SECS=${DURATION%s} ;;
    *)  SLEEP_SECS=$(( DURATION * 60 )) ;;
esac

RESUME_AT=$(date -d "+${SLEEP_SECS} seconds" '+%H:%M')

echo "Stopping benchmark for $DURATION (resumes at $RESUME_AT)..."

# Stop ollama (runs as root via systemd)
sudo systemctl stop ollama
pkill -9 -f run_workers 2>/dev/null || true

# Confirm
curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && echo "WARNING: ollama still up" || echo "Ollama stopped."

echo "Benchmark paused. Sleeping until $RESUME_AT..."
sleep "$SLEEP_SECS"

echo "Waking up — restarting ollama..."

# Restart ollama via systemd — run_all.sh will re-tune NUM_PARALLEL itself
sudo systemctl start ollama

# Wait for ollama to be ready
until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
    sleep 2
done

echo "Ollama is back. run_all.sh will pick up automatically — benchmark resuming."
