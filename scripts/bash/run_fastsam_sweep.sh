#!/usr/bin/env bash
# Run the FastSAM-3D arm over every trial, chunked per participant.
#
# FastSAM is exported from camera 0 only, so there is a single configuration to
# run rather than the camera-count ladder the other arms need. Chunking per
# participant bounds memory over a hundred-trial run and matches how the other
# sweeps were produced; everything is resumable, since sweep.py skips trials
# already present in its summary CSV.
#
#   bash scripts/bash/run_fastsam_sweep.sh [dataset]

set -u
DATASET="${1:-/root/workspace/COMFI}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASKS="Screwing Polishing SideOverhead RobotPolishing RobotWelding Lifting"
LOGS="$REPO/results/logs"
SUMMARY="$REPO/results/fastsam_0.csv"
mkdir -p "$LOGS" "$REPO/results"

echo "=== $(date '+%F %T')  stage fastsam_0 ==="
for p in $(ls "$DATASET/fastsam"); do
    timeout 3600 python3 "$REPO/scripts/python/paper/sweep.py" \
        --arm fastsam --dataset "$DATASET" --cameras 0 \
        --summary "$SUMMARY" --participants "$p" --tasks $TASKS \
        --tag fastsam_0 >> "$LOGS/fastsam_0.log" 2>&1
    rc=$?
    rows=$(( $(wc -l < "$SUMMARY" 2>/dev/null || echo 1) - 1 ))
    echo "  $(date '+%T') participant $p rc=$rc, $rows rows so far"
done
echo "=== $(date '+%F %T')  fastsam_0 finished ==="
