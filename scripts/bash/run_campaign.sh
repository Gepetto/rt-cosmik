#!/usr/bin/env bash
# Run the whole paper campaign -- every arm, then every metric -- into its own
# folders, so a campaign never mixes with runs made under other code:
#
#   runs      output/<campaign>/<participant>/<task>/<arm>
#   results   results/<campaign>/<arm>.csv, vs_mocap/, paper/, logs/
#
#   bash scripts/bash/run_campaign.sh validation 1012     # one participant first
#   bash scripts/bash/run_campaign.sh campaign            # everyone
#
# Arms run one after another, never in parallel: the sweep's throughput is part
# of the results, and two arms sharing the GPU would each report the other's
# load. Everything is resumable -- sweep.py skips trials already in its summary
# -- so re-running continues rather than restarting. Stages are chunked per
# participant to bound memory.

set -u
CAMPAIGN="${1:?campaign name, e.g. validation}"
shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="/root/workspace/COMFI"
TASKS="Screwing Polishing SideOverhead RobotPolishing RobotWelding Lifting"
RUNS="$REPO/output/$CAMPAIGN"
RESULTS="$REPO/results/$CAMPAIGN"
LOGS="$RESULTS/logs"
PY="$REPO/scripts/python/paper"
mkdir -p "$RUNS" "$LOGS"

PARTICIPANTS="${*:-$(ls "$DATASET/mmpose/output")}"

# tag = <arm>_<cameras joined by ->, e.g. nlf_0-2-4-6
ARMS=(
    "mocap 0"
    "mmpose 0 2 4 6"   "mmpose 0 2"
    "nlf2d 0 2 4 6"    "nlf2d 0 2"
    "nlf 0 2 4 6"      "nlf 0 2"      "nlf 0"
    "fastsam 0 2 4 6"  "fastsam 0 2"  "fastsam 0"
)
SCORED="mmpose_0-2-4-6 mmpose_0-2 nlf2d_0-2-4-6 nlf2d_0-2 nlf_0-2-4-6 nlf_0-2 nlf_0 fastsam_0-2-4-6 fastsam_0-2 fastsam_0"

echo "=== $(date '+%F %T')  campaign $CAMPAIGN: participants $PARTICIPANTS"
{   # what this campaign ran on, for the handoff
    echo "started: $(date '+%F %T')"; echo "participants: $PARTICIPANTS"
    echo "--- git"; git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --short
    echo "--- nvidia-smi"; nvidia-smi
    echo "--- cpu"; lscpu | grep -E "Model name|^CPU\(s\)"
    echo "--- governor"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
    echo "--- python"; python3 -c "import numpy, scipy, torch, pinocchio; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'pinocchio', pinocchio.__version__)"
} > "$RESULTS/environment.txt" 2>&1
for spec in "${ARMS[@]}"; do
    read -r arm cams <<< "$spec"
    tag="${arm}_$(echo "$cams" | tr ' ' '-')"
    [ "$arm" = mocap ] && tag=mocap_reference
    echo "=== $(date '+%F %T')  $tag"
    for p in $PARTICIPANTS; do
        # The proposed pipeline also saves its NLF output, which the design
        # studies (E3-E5) replay without the GPU.
        cache=(); [ "$tag" = nlf_0-2-4-6 ] && cache=(--views-cache "$RESULTS/views")
        timeout 7200 python3 "$PY/sweep.py" --arm "$arm" --dataset "$DATASET" --cameras $cams \
            --summary "$RESULTS/$tag.csv" --participants "$p" --tasks $TASKS \
            --tag "$tag" --output-dir "$RUNS" "${cache[@]}" >> "$LOGS/$tag.log" 2>&1
        rc=$?
        ok=$(grep -cE ",ok[[:space:]]*$" "$RESULTS/$tag.csv" 2>/dev/null)
        echo "  $(date '+%T') $p rc=$rc, $ok trials ok so far"
    done
done

echo "=== $(date '+%F %T')  design studies E3-E5 (replays of nlf_0-2-4-6)"
python3 "$PY/ik_filter_studies.py" --views "$RESULTS/views" --output-dir "$RUNS" \
    --out "$RESULTS/paper" > "$LOGS/studies.log" 2>&1
grep -E "trials x|baseline replay|study rows|Traceback" "$LOGS/studies.log"

echo "=== $(date '+%F %T')  metrics"
python3 "$PY/rescore_summaries.py" --results "$RESULTS" --output-dir "$RUNS" 2>&1 | tee "$LOGS/rescore.log"
common=(--output-dir "$RUNS" --results "$RESULTS" --out "$RESULTS/paper")
python3 "$PY/trial_metrics.py" --arms $SCORED "${common[@]}" 2>&1 | tee "$LOGS/trial_metrics.log"
python3 "$PY/reba_agreement.py" --arms $SCORED "${common[@]}" 2>&1 | tee "$LOGS/reba.log"
python3 "$PY/robot_distance.py" --arms $SCORED "${common[@]}" 2>&1 | tee "$LOGS/robot_distance.log"
python3 "$PY/fastsam_timing.py" --out "$RESULTS/paper" 2>&1 | tee "$LOGS/fastsam_timing.log"
python3 "$PY/aggregate_results.py" --arms $SCORED --root "$RESULTS/paper" --results "$RESULTS" \
    2>&1 | tee "$LOGS/aggregate.log"
python3 "$PY/build_handoff.py" --campaign "$CAMPAIGN" 2>&1 | tee "$LOGS/handoff.log"
echo "=== $(date '+%F %T')  campaign $CAMPAIGN done: copy $RESULTS/handoff to the paper workspace"
