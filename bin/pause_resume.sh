#!/bin/bash
# Pause the run_all.sh process group for 1 hour, then resume.
# Usage: ./pause_resume.sh <pgid> [seconds]

PGID=${1:-62295}
PAUSE_SECS=${2:-3600}

echo "[$(date)] Sending SIGSTOP to process group $PGID..."
kill -STOP -"$PGID"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to send SIGSTOP. Is the process group still running?"
    exit 1
fi

echo "[$(date)] Paused. Will resume in ${PAUSE_SECS}s ($(date -d "+${PAUSE_SECS} seconds"))..."
sleep "$PAUSE_SECS"

echo "[$(date)] Sending SIGCONT to process group $PGID..."
kill -CONT -"$PGID"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to send SIGCONT. Processes may have exited."
    exit 1
fi

echo "[$(date)] Resumed."
