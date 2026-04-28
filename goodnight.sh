#!/bin/bash
# goodnight.sh — shut down and wake at 6am to resume benchmark.

WAKE_TIME=$(date -d "tomorrow 06:00" +%s)
WAKE_DISPLAY=$(date -d "tomorrow 06:00" '+%H:%M on %a %d %b')

echo "Setting wake alarm for $WAKE_DISPLAY..."
echo "Shutting down now. PC will power on automatically at 6am."
sleep 2

sudo rtcwake -m off -t "$WAKE_TIME"
