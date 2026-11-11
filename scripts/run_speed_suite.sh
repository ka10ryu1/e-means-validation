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
THREADS=4
MAX_WAIT_MIN=60
LOAD_MAX="${LOAD_MAX:-1.0}"
PROFILES="quick medium"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2 ;;
    --max-wait-min) MAX_WAIT_MIN="$2"; shift 2 ;;
    --profiles) PROFILES="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[[ -x "$UV" ]] || { echo "uv not found" >&2; exit 1; }

if [[ -n "$(git status --porcelain -- . ':!results' ':!reports')" ]]; then
  echo "Working tree has uncommitted changes; commit before measuring." >&2
  exit 1
fi

# Wait for an idle machine.
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
while :; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{exit !(l < m)}'; then break; fi
  if (( $(date +%s) >= deadline )); then
    echo "Load average stayed >= $LOAD_MAX for $MAX_WAIT_MIN min (now $load); not measuring." >&2
    exit 3
  fi
  echo "$(date -Is) load $load >= $LOAD_MAX, waiting..."
  sleep 60
done

stamp="$(date +%Y%m%d-%H%M)"
for profile in $PROFILES; do
  run="speed_${stamp}_${profile}_t${THREADS}"
  extra=()
  [[ "$profile" == "quick" ]] && extra=(--repeats 3)
  mkdir -p results
  echo "$(date -Is) start $run (load $(cut -d' ' -f1-3 /proc/loadavg))"
  "$UV" run python benchmark.py --profile "$profile" --threads "$THREADS" \
      --run-name "$run" "${extra[@]}" > "results/$run.log" 2>&1
  flagged=$(grep -o '[0-9]*/[0-9]* 回の実行で CPU の取り合いを検出' "reports/$run/report.md" || true)
  echo "$(date -Is) done  $run  contention: ${flagged:-none detected}"
done
