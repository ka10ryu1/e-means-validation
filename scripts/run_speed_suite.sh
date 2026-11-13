#!/usr/bin/env bash
# Unattended speed measurement (run when the machine is otherwise idle, e.g. at night).
#
#   bash scripts/run_speed_suite.sh [--threads N] [--max-wait-min M] [--profiles "quick medium"]
#                                   [--real "sift1m deep1m"] [--real-points "base k256 ..."]
#
# --real runs the medium profile on data/<name>.npy (prepare_real_data.py) after the
# profiles; e.g. --profiles "" --real "sift1m deep1m" measures only the real data.
# --real-points (default "base") varies one factor around the medium point for each
# real dataset: base (n=200k, k=1024), k256, k4096, n800k. All points use the medium
# seeds (0-4). Points run in the given order for every dataset before the next point.
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
REAL=""
REAL_POINTS="base"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2 ;;
    --max-wait-min) MAX_WAIT_MIN="$2"; shift 2 ;;
    --profiles) PROFILES="$2"; shift 2 ;;
    --real) REAL="$2"; shift 2 ;;
    --real-points) REAL_POINTS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

real_point_args() {
  case "$1" in
    base)  echo "" ;;
    k256)  echo "--k 256" ;;
    k4096) echo "--k 4096" ;;
    n800k) echo "--ntrain 800000" ;;
    *) echo "unknown real point: $1" >&2; return 1 ;;
  esac
}
for point in $REAL_POINTS; do real_point_args "$point" > /dev/null; done  # fail before waiting

require_clean_tree
for name in $REAL; do
  [[ -f "data/$name.npy" ]] || { echo "data/$name.npy missing; run prepare_real_data.py" >&2; exit 1; }
done
wait_for_idle "$MAX_WAIT_MIN"

stamp="$(date +%Y%m%d-%H%M)"
for profile in $PROFILES; do
  extra=()
  [[ "$profile" == "quick" ]] && extra=(--repeats 3)
  run_benchmark "speed_${stamp}_${profile}_t${THREADS}" \
      --profile "$profile" --threads "$THREADS" "${extra[@]}"
done
for point in $REAL_POINTS; do
  for name in $REAL; do
    tag="medium"
    [[ "$point" != "base" ]] && tag="$point"
    # shellcheck disable=SC2046  # real_point_args returns separate words on purpose
    run_benchmark "speed_${stamp}_${name}_${tag}_t${THREADS}" \
        --profile medium --input "data/$name.npy" --threads "$THREADS" $(real_point_args "$point")
  done
done
