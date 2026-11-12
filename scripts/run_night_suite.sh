#!/usr/bin/env bash
# One unattended night: the pending speed measurements, one after another.
#
#   bash scripts/run_night_suite.sh [--max-wait-min M] [--points "d2tp d16tp"]
#                                   [--threads-list "6 4 2 1"]
#
# 1. run_scaling_suite.sh --points "$POINTS" at 4 threads (low-d with the throughput
#    batch law; about 10 min)
# 2. run_speed_suite.sh --profiles medium --threads N for each N in THREADS_LIST
#    (4 is re-run the same night so the thread comparison is within one night)
# Cheapest first, so a night cut short still leaves complete reports for what ran.
# Each step keeps the guards of the underlying suite (clean tree, idle wait,
# contention count). A failing step stops the night (set -e).
# Example (Windows Task Scheduler):
#   wsl.exe -d Ubuntu -- bash -lc "cd ~/Work_2404_001/e-means-validation && \
#     bash scripts/run_night_suite.sh > results/night_\$(date +%Y%m%d-%H%M).log 2>&1"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MAX_WAIT_MIN=60
POINTS="d2tp d16tp"
THREADS_LIST="6 4 2 1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-wait-min) MAX_WAIT_MIN="$2"; shift 2 ;;
    --points) POINTS="$2"; shift 2 ;;
    --threads-list) THREADS_LIST="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
for t in $THREADS_LIST; do
  [[ "$t" =~ ^[1-9][0-9]*$ ]] || { echo "bad thread count: $t" >&2; exit 2; }
done

echo "$(date -Is) night suite start (commit $(git rev-parse --short HEAD))"
if [[ -n "$POINTS" ]]; then
  bash scripts/run_scaling_suite.sh --threads 4 --max-wait-min "$MAX_WAIT_MIN" --points "$POINTS"
fi
for t in $THREADS_LIST; do
  bash scripts/run_speed_suite.sh --profiles medium --threads "$t" --max-wait-min "$MAX_WAIT_MIN"
done
echo "$(date -Is) night suite done"
