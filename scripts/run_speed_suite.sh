#!/usr/bin/env bash
# Unattended speed measurement (run when the machine is otherwise idle, e.g. at night).
#
#   bash scripts/run_speed_suite.sh [--threads N] [--max-wait-min M] [--profiles "quick medium"]
#
# - refuses to run on a dirty tree (results must map to a commit)
# - waits until the 1-minute load average is below LOAD_MAX (default 1.0)
# - runs the profiles one after another (never in parallel)
# - prints, per run, how many runs were flagged for CPU contention
# Raw results: results/<run>/, reports: reports/<run>/, log: results/<run>.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/_suite_common.sh
THREADS=4
MAX_WAIT_MIN=60
PROFILES="quick medium"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2 ;;
    --max-wait-min) MAX_WAIT_MIN="$2"; shift 2 ;;
    --profiles) PROFILES="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

require_clean_tree
wait_for_idle "$MAX_WAIT_MIN"

stamp="$(date +%Y%m%d-%H%M)"
for profile in $PROFILES; do
  extra=()
  [[ "$profile" == "quick" ]] && extra=(--repeats 3)
  run_benchmark "speed_${stamp}_${profile}_t${THREADS}" \
      --profile "$profile" --threads "$THREADS" "${extra[@]}"
done
