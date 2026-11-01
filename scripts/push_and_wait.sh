#!/usr/bin/env bash
# usage: scripts/push_and_wait.sh <kernel-name>   (kernels/<name>/ must exist)
set -uo pipefail
NAME="$1"; USER_ID="shamimahossain"; KID="$USER_ID/dna-grpo-$NAME"
cd "$(dirname "$0")/.."
kaggle kernels push -p "kernels/$NAME"
while true; do
  sleep 60
  ST=$(kaggle kernels status "$KID" 2>&1 | tr '[:upper:]' '[:lower:]')
  echo "[$(date +%H:%M:%S)] $ST"
  case "$ST" in
    *complete*|*error*|*cancel*) break ;;
  esac
done
mkdir -p "outputs/$NAME"
kaggle kernels output "$KID" -p "outputs/$NAME" -o >/dev/null 2>&1 || kaggle kernels output "$KID" -p "outputs/$NAME"
LOG=$(ls outputs/$NAME/*.log 2>/dev/null | head -1)
[ -n "$LOG" ] && tail -c 4000 "$LOG"
echo "FINAL: $ST"
