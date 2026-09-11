#!/usr/bin/env bash
# Run the SMPL-refined NLF arm over every trial, chunked per participant.
#
# Same structure as the other sweeps: resumable, one summary row per trial, and
# chunked per participant to bound memory over a hundred-trial run. The refiner
# is compiled once per process and reused across trials within a participant --
# torch.compile costs ~18 s and recompiles per input shape, so it must not
# happen per trial.
#
#   bash scripts/bash/run_nlfsmpl_sweep.sh [dataset] [cameras] [beta-mode]
#
# e.g. bash scripts/bash/run_nlfsmpl_sweep.sh /root/workspace/COMFI "0 2 4 6" calibrated

set -u
DATASET="${1:-/root/workspace/COMFI}"
CAMS="${2:-0 2 4 6}"
BETA="${3:-calibrated}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASKS="Screwing Polishing SideOverhead RobotPolishing RobotWelding Lifting"
TAG="nlfsmpl_$(echo "$CAMS" | tr ' ' '-')"
LOGS="$REPO/results/logs"
SUMMARY="$REPO/results/${TAG}.csv"
mkdir -p "$LOGS" "$REPO/results"

echo "=== $(date '+%F %T')  stage $TAG (beta-mode $BETA) ==="
for p in $(ls "$DATASET/mmpose/output"); do
    timeout 7200 python3 "$REPO/scripts/python/paper/sweep.py" \
        --arm nlfsmpl --dataset "$DATASET" --cameras $CAMS \
        --beta-mode "$BETA" --summary "$SUMMARY" \
        --participants "$p" --tasks $TASKS \
        --tag "$TAG" >> "$LOGS/${TAG}.log" 2>&1
    rc=$?
    rows=$(( $(wc -l < "$SUMMARY" 2>/dev/null || echo 1) - 1 ))
    echo "  $(date '+%T') participant $p rc=$rc, $rows rows so far"
done
echo "=== $(date '+%F %T')  $TAG finished ==="
