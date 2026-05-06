#!/bin/bash
# Cleanly stop the benchmark pipeline and restart it.
# Safe to run at any time — completed tiers are preserved in results/evaluations.jsonl.

cd /home/rabrew/Desktop/chess-llm-bench

echo "Stopping benchmark pipeline..."

# Kill run_all.sh and any run_workers.py children
pkill -f "bash bin/run_all.sh" 2>/dev/null
pkill -f "run_workers.py" 2>/dev/null

# Wait for processes to exit
sleep 3

# Confirm they're gone
if pgrep -f "bin/run_all.sh\|run_workers.py" > /dev/null; then
    echo "Force killing remaining processes..."
    pkill -9 -f "bash bin/run_all.sh" 2>/dev/null
    pkill -9 -f "run_workers.py" 2>/dev/null
    sleep 2
fi

echo "Stopped. Restarting in the chess-bench tmux session..."
tmux send-keys -t chess-bench:0 "" Enter   # clear any partial input
tmux send-keys -t chess-bench:0 "bash bin/run_all.sh" Enter

echo "Done — benchmark restarted in chess-bench:0"
