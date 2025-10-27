#!/usr/bin/env bash
# alarm_when_done.sh — ring an alarm when a PID exits (macOS)
# Usage:
#   ./alarm_when_done.sh <PID> [message]
# Example:
#   ./alarm_when_done.sh 12345 "Training job finished"

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <PID> [message]" >&2
  exit 1
fi

pid="$1"
msg="${2:-Process $pid finished}"

# Poll until the PID disappears
while ps -p "$pid" >/dev/null 2>&1; do
  sleep 2
done

# 1) Terminal bell (instant, no deps)
printf '\a'

# 2) macOS user notification (with sound)
osascript -e 'display notification "'"$msg"'" with title "Process Done" sound name "Submarine"'

# 3) Play a system sound (fallback if you want it louder)
if command -v afplay >/dev/null 2>&1; then
  afplay /System/Library/Sounds/Submarine.aiff >/dev/null 2>&1 || true
fi

# 4) Optional spoken alert (uncomment to use)
say "DONE TRAINING"
