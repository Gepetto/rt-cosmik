#!/usr/bin/env bash
# Run every configuration of the paper comparison, in priority order.
#
# Ordered so that if the machine dies partway, what exists is the most valuable
# subset: the two 4-camera arms first, since those are the headline comparison,
# then the 2-camera arms, then NLF's 1-camera row which has no baseline to
# compare against anyway.
#
# Everything is resumable. sweep.py skips trials already in its summary CSV, so
# re-running this script continues rather than restarting, and a trial that
# fails costs only itself. Work is chunked per participant to bound memory over
# a hundred-trial run; for NLF that costs nothing extra, because the estimator is
# rebuilt per participant regardless (the intrinsics are baked into it).
#
#   bash scripts/bash/run_paper_sweeps.sh <dataset> <2cam-pair, e.g. "0 2">

set -u
DATASET="${1:-/root/workspace/COMFI}"
PAIR="${2:-0 2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASKS="Screwing Polishing SideOverhead RobotPolishing RobotWelding Lifting"
LOGS="$REPO/results/logs"
mkdir -p "$LOGS" "$REPO/results"

PARTICIPANTS=$(ls "$DATASET/mmpose/output")

stage () {                      # stage <arm> <cameras...> -> one summary CSV
    local arm="$1"; shift
    local cams="$*"
    local tag="${arm}_$(echo "$cams" | tr ' ' '-')"
    local summary="$REPO/results/${tag}.csv"
    echo "=== $(date '+%F %T')  stage $tag  ==="
    for p in $PARTICIPANTS; do
        timeout 7200 python3 "$REPO/scripts/python/paper/sweep.py" \
            --arm "$arm" --dataset "$DATASET" --cameras $cams \
            --summary "$summary" --participants "$p" --tasks $TASKS \
            --tag "$tag" >> "$LOGS/${tag}.log" 2>&1
        local rc=$?
        local done_rows=$(( $(wc -l < "$summary" 2>/dev/null || echo 1) - 1 ))
        echo "  $(date '+%T') participant $p rc=$rc, $done_rows rows so far"
    done
    echo "=== $(date '+%F %T')  $tag finished ==="
}

# One NLF trial before committing hours to it. sweep.py survives per-trial
# failures, but a systematic one -- a missing engine for a camera count, a CUDA
# problem -- would burn the night writing error rows. Better to find out in two
# minutes and still deliver the mmpose arms.
echo "=== $(date '+%F %T')  NLF precheck ==="
timeout 1800 python3 "$REPO/scripts/python/paper/sweep.py" \
    --arm nlf --dataset "$DATASET" --cameras 0 2 4 6 \
    --summary "$LOGS/precheck.csv" --participants 1012 --tasks Lifting \
    --tag precheck_nlf > "$LOGS/precheck.log" 2>&1
if grep -q ",ok$" "$LOGS/precheck.csv" 2>/dev/null; then
    NLF_OK=1; echo "  NLF precheck passed"
else
    NLF_OK=0
    echo "  NLF PRECHECK FAILED -- running the mmpose arms only."
    echo "  last lines of $LOGS/precheck.log:"; tail -20 "$LOGS/precheck.log"
fi

stage mmpose 0 2 4 6
[ "$NLF_OK" = 1 ] && stage nlf 0 2 4 6
stage mmpose $PAIR
[ "$NLF_OK" = 1 ] && stage nlf $PAIR
[ "$NLF_OK" = 1 ] && stage nlf 0

echo "=== $(date '+%F %T')  all stages done, writing report ==="
python3 "$REPO/scripts/python/paper/report.py" \
    "$REPO"/results/mmpose_*.csv "$REPO"/results/nlf_*.csv --by-task \
    > "$REPO/results/REPORT.txt" 2>&1
cat "$REPO/results/REPORT.txt"
