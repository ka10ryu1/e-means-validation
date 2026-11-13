#!/usr/bin/env bash
# Scaling sweep around the medium profile (n=200k, k=1024, d=64), one factor at a time.
# Run when the machine is otherwise idle; the whole default set takes about 3 hours
# at 4 threads on the development machine (Core Ultra 7 258V).
#
#   bash scripts/run_scaling_suite.sh [--threads N] [--max-wait-min M] [--points "k256 n1m ..."]
#
# The medium point itself is not re-run: use the latest speed_*_medium_t4 report.
# Each point uses seeds 0 1 2 and all three datasets. Same guards as run_speed_suite.sh.
# Extra points, not in the default set: d2tp / d16tp = d2 / d16 with
# --emeans-batch-law throughput (the official default picks the quality law at d <= 16),
# to separate the low-d slowness of e-means from its small batches.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/_suite_common.sh
THREADS=4
MAX_WAIT_MIN=60
# Cheapest first, so a run cut short still covers every axis it reached.
POINTS="d2 k256 d16 n100k d128 k4096 n1m d768"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2 ;;
    --max-wait-min) MAX_WAIT_MIN="$2"; shift 2 ;;
    --points) POINTS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

point_args() {
  case "$1" in
    k256)  echo "--k 256" ;;
    k4096) echo "--k 4096" ;;
    n100k) echo "--ntrain 100000" ;;
    n1m)   echo "--ntrain 1000000" ;;
    d2)    echo "--dim 2" ;;
    d16)   echo "--dim 16" ;;
    d128)  echo "--dim 128" ;;
    d768)  echo "--dim 768" ;;
    d2tp)  echo "--dim 2 --emeans-batch-law throughput" ;;
    d16tp) echo "--dim 16 --emeans-batch-law throughput" ;;
    *) echo "unknown point: $1" >&2; return 1 ;;
  esac
}
for point in $POINTS; do point_args "$point" > /dev/null; done  # fail before waiting

require_clean_tree
wait_for_idle "$MAX_WAIT_MIN"

stamp="$(date +%Y%m%d-%H%M)"
for point in $POINTS; do
  # shellcheck disable=SC2046  # point_args returns separate words on purpose
  run_benchmark "scale_${stamp}_${point}_t${THREADS}" \
      --profile medium --threads "$THREADS" --seeds 0 1 2 $(point_args "$point")
done
