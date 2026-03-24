#!/bin/bash
# Monitor script - constantly checks tmux session for errors

LOGFILE="/home/rabrew/Desktop/chess-llm-bench/crash_report.txt"
SESSION="chess"
LAST_PROGRESS=""

log() {
    echo "$1" | tee -a "$LOGFILE"
}

echo "=== Chess Benchmark Monitor ===" | tee "$LOGFILE"
log "Started: $(date)"
log "Checking every 10 seconds..."
log ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if tmux session exists
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        log ""
        log "========================================"
        log "[$TIMESTAMP] SESSION ENDED"
        log "========================================"

        # Check if it completed successfully or crashed
        if [ -f "/home/rabrew/Desktop/chess-llm-bench/results/evaluations.jsonl" ]; then
            log "STATUS: Completed successfully (results file exists)"
        else
            log "STATUS: May have crashed (no results file yet)"
        fi

        log ""
        log "Check tmux output with: tmux capture-pane -t chess -p -S -100"
        log ""
        log "Monitor stopping."
        exit 0
    fi

    # Capture recent tmux output
    RECENT_OUTPUT=$(tmux capture-pane -t "$SESSION" -p -S -30 2>/dev/null)

    # Check for error patterns (excluding the old KeyboardInterrupt)
    CURRENT_ERRORS=$(echo "$RECENT_OUTPUT" | grep -iE "(error|exception|failed|crash|killed|oom|memory)" | grep -v "KeyboardInterrupt" | tail -3)

    if [ -n "$CURRENT_ERRORS" ]; then
        log ""
        log "========================================"
        log "[$TIMESTAMP] ERROR DETECTED"
        log "========================================"
        log "$CURRENT_ERRORS"
        log ""
        log "Full recent output:"
        log "$RECENT_OUTPUT"
        log "========================================"
    fi

    # Get current progress line
    PROGRESS_LINE=$(echo "$RECENT_OUTPUT" | grep -E "(%\||Evaluating|Validating|STEP|Processing)" | tail -1)

    # Only log if progress changed
    if [ -n "$PROGRESS_LINE" ] && [ "$PROGRESS_LINE" != "$LAST_PROGRESS" ]; then
        log "[$TIMESTAMP] $PROGRESS_LINE"
        LAST_PROGRESS="$PROGRESS_LINE"
    fi

    # Sleep 10 seconds
    sleep 10
done
