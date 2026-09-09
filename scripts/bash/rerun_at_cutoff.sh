#!/usr/bin/env bash
# Re-run every markerless configuration at a new IIR cutoff.
#
# The MoCap reference is deliberately unfiltered -- it is what noise is measured
# against -- so it is not regenerated here and the reference stays fixed across
# the change. The OCP does not depend on the filter either, so nothing is rebuilt.
#
#   bash scripts/bash/rerun_at_cutoff.sh 5 /root/workspace/COMFI "0 2"
set -u
CUT="${1:?cutoff Hz}"
DATASET="${2:-/root/workspace/COMFI}"
PAIR="${3:-0 2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASKS="Screwing Polishing SideOverhead RobotPolishing RobotWelding Lifting"
cd "$REPO"

OLD=$(python3 -c "import sys;sys.path.insert(0,'src');from rtcosmik.config_loader import settings;print(int(settings.cutoff_freq))")
echo "=== $(date '+%F %T') re-running at cutoff ${CUT} Hz (was ${OLD}) ==="

# Keep the previous summaries so the two cutoffs can be compared afterwards.
mkdir -p "results/cutoff${OLD}"
for f in results/mmpose_*.csv results/nlf_*.csv; do
    [ -f "$f" ] && mv "$f" "results/cutoff${OLD}/" 2>/dev/null
done
rm -rf "results/vs_mocap"

python3 - "$CUT" <<'PY'
import re, sys
cut = sys.argv[1]
p = "settings.py"; s = open(p).read()
s = re.sub(r"^    cutoff_freq: float = [\d.]+", f"    cutoff_freq: float = {cut}", s, flags=re.M)
open(p, "w").write(s)
print(f"settings.cutoff_freq set to {cut}")
PY

stage () {
    local arm="$1"; shift; local cams="$*"
    local tag="${arm}_$(echo "$cams" | tr ' ' '-')"
    echo "=== $(date '+%F %T') stage $tag ==="
    for p in $(ls "$DATASET/mmpose/output"); do
        timeout 7200 python3 scripts/python/paper/sweep.py --arm "$arm" \
            --dataset "$DATASET" --cameras $cams --summary "results/${tag}.csv" \
            --participants "$p" --tasks $TASKS --tag "$tag" \
            >> "results/logs/${tag}.log" 2>&1
    done
    echo "  $(date '+%T') $tag -> $(grep -cE ',ok[[:space:]]*$' "results/${tag}.csv" 2>/dev/null || echo 0) rows"
}

stage mmpose 0 2 4 6
stage mmpose $PAIR
stage nlf    0 2 4 6
stage nlf    $PAIR
stage nlf    0

echo "=== $(date '+%F %T') rescoring against the MoCap reference ==="
python3 scripts/python/paper/rescore_summaries.py
python3 scripts/python/paper/report.py results/vs_mocap/*.csv --by-task > results/REPORT.txt 2>&1
cat results/REPORT.txt
echo "=== $(date '+%F %T') CUTOFF RERUN COMPLETE ==="
