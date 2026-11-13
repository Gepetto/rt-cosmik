#!/usr/bin/env bash
# FastSAM-3D out-of-fold pass: the four configurations the stored meshes allow,
# over every participant. CPU only, JOBS participants at a time; resumable, since
# sweep.py skips trials already in a participant's summary.
# Needs committed pipeline code (the sweep refuses otherwise) and
# landmark_map = "out_of_fold" in settings.py.
#
#   bash scripts/bash/run_fastsam_oof.sh [dataset] [runs dir] [summaries dir]

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${1:-/root/workspace/COMFI}"
RUNS="${2:-$REPO/output/oof}"
SUMMARIES="${3:-$REPO/results/oof}"
TASKS="Screwing Polishing SideOverhead RobotPolishing RobotWelding Lifting"
CONFIGURATIONS=( "2"  "4 6"  "2 4"  "2 4 6" )
JOBS=12
LOGS="$SUMMARIES/logs"
PARTS="$SUMMARIES/fastsam_parts"
OK=',ok[[:space:]]*$'       # the csv writer ends lines with CRLF
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export REPO DATASET RUNS TASKS LOGS PARTS OK
mkdir -p "$LOGS" "$PARTS"
echo "commit $(git -C "$REPO" rev-parse --short HEAD), start $(date '+%F %T')"

one () {    # one <participant> <cameras...>: a sweep with its own summary, merged below
    local p="$1"; shift
    local tag; tag="fastsam_$(echo "$*" | tr ' ' '-')"
    python3 "$REPO/scripts/python/paper/sweep.py" --arm fastsam --dataset "$DATASET" \
        --cameras "$@" --summary "$PARTS/$tag.$p.csv" --participants "$p" --tasks $TASKS \
        --tag "$tag" --output-dir "$RUNS" >> "$LOGS/$tag.$p.log" 2>&1
    echo "$(date '+%T') $tag $p rc=$? ok=$(grep -cE "$OK" "$PARTS/$tag.$p.csv" 2>/dev/null)"
}
export -f one

for cameras in "${CONFIGURATIONS[@]}"; do
    tag="fastsam_$(echo "$cameras" | tr ' ' '-')"
    ls "$DATASET/mmpose/output" | xargs -P "$JOBS" -I{} bash -c "one {} $cameras"
    parts=( "$PARTS/$tag".*.csv )
    { head -1 "${parts[0]}"; tail -q -n +2 "${parts[@]}"; } > "$SUMMARIES/$tag.csv"
    echo "$tag: $(grep -cE "$OK" "$SUMMARIES/$tag.csv") ok of $(( $(wc -l < "$SUMMARIES/$tag.csv") - 1 ))"
done
echo "done $(date '+%F %T')"
