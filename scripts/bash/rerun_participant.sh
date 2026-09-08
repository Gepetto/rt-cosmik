#!/usr/bin/env bash
# Re-run one participant across every finished configuration.
#
# For when a per-participant data problem is found after a sweep has already
# scored them -- COMFI's 3361 has its camera pairs labelled the other way round,
# which is a property of the dataset, so every configuration got it wrong the
# same way and every configuration has to be redone for that subject.
#
#   bash scripts/bash/rerun_participant.sh 3361 <dataset> "<2cam pair>"
set -u
PART="${1:?participant}"
DATASET="${2:-/root/workspace/COMFI}"
PAIR="${3:-0 2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASKS="Screwing Polishing SideOverhead RobotPolishing RobotWelding Lifting"

for summary in "$REPO"/results/mmpose_*.csv "$REPO"/results/nlf_*.csv; do
    [ -f "$summary" ] || continue
    name=$(basename "$summary" .csv)
    arm="${name%%_*}"; cams=$(echo "${name#*_}" | tr '-' ' ')
    before=$(grep -cE ",ok[[:space:]]*$" "$summary")
    # Drop this participant's rows, keeping the header and everyone else.
    awk -F, -v p="$PART" 'NR==1 || $2!=p' "$summary" > "$summary.tmp" && mv "$summary.tmp" "$summary"
    echo "=== $(date '+%T') $name: dropped $PART, re-running ($before rows before) ==="
    timeout 7200 python3 "$REPO/scripts/python/paper/sweep.py" \
        --arm "$arm" --dataset "$DATASET" --cameras $cams \
        --summary "$summary" --participants "$PART" --tasks $TASKS \
        --tag "$name" >> "$REPO/results/logs/${name}.log" 2>&1
    echo "    now $(grep -cE ",ok[[:space:]]*$" "$summary") ok rows"
done

python3 "$REPO/scripts/python/paper/report.py" \
    "$REPO"/results/mmpose_*.csv "$REPO"/results/nlf_*.csv --by-task \
    > "$REPO/results/REPORT.txt" 2>&1
echo "=== $(date '+%T') report rewritten ==="
