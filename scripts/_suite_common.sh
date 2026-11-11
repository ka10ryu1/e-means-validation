# Shared by run_speed_suite.sh and run_scaling_suite.sh (source it; not executable).
# Provides: UV, require_clean_tree, wait_for_idle MAX_WAIT_MIN, run_benchmark RUN ARGS...

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[[ -x "$UV" ]] || { echo "uv not found" >&2; exit 1; }
LOAD_MAX="${LOAD_MAX:-1.0}"

require_clean_tree() {
  if [[ -n "$(git status --porcelain -- . ':!results' ':!reports')" ]]; then
    echo "Working tree has uncommitted changes; commit before measuring." >&2
    exit 1
  fi
}

# Wait until the 1-minute load average is below LOAD_MAX, or give up after $1 minutes.
wait_for_idle() {
  local max_wait_min="$1" deadline load
  deadline=$(( $(date +%s) + max_wait_min * 60 ))
  while :; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{exit !(l < m)}'; then return 0; fi
    if (( $(date +%s) >= deadline )); then
      echo "Load average stayed >= $LOAD_MAX for $max_wait_min min (now $load); not measuring." >&2
      exit 3
    fi
    echo "$(date -Is) load $load >= $LOAD_MAX, waiting..."
    sleep 60
  done
}

# run_benchmark RUN_NAME BENCHMARK_ARGS... : one run, log to results/RUN.log, print contention.
run_benchmark() {
  local run="$1" flagged; shift
  mkdir -p results
  echo "$(date -Is) start $run (load $(cut -d' ' -f1-3 /proc/loadavg))"
  "$UV" run python benchmark.py --run-name "$run" "$@" > "results/$run.log" 2>&1
  flagged=$(grep -o '[0-9]*/[0-9]* 回の実行で CPU の取り合いを検出' "reports/$run/report.md" || true)
  echo "$(date -Is) done  $run  contention: ${flagged:-none detected}"
}
