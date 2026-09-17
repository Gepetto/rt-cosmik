#!/usr/bin/env bash
# Run the whole paper campaign -- every arm, then every metric -- into its own
# folders, so a campaign never mixes with runs made under other code:
#
#   runs      output/<campaign>/<participant>/<task>/<arm>
#   results   results/<campaign>/<arm>.csv, vs_mocap/, paper/, handoff/, logs/
#
#   bash scripts/bash/run_campaign.sh validation 1012     # one participant first
#   bash scripts/bash/run_campaign.sh campaign            # everyone
#
# Stages, in order:
#   1. direct arms: mocap reference, mmpose+LSTM, NLF-3D 4 cams (which also
#      records every camera's NLF output), FastSAM
#   2. direct timing runs of the replayed NLF configurations, on a few
#      participants only: replays have no throughput of their own
#   3. replayed NLF configurations (fewer cameras, 2D triangulation), CPU
#   4. design studies E3-E5 (IK type, horizon, filter), CPU replays
#   5. metrics, tables, handoff
#
# Direct arms run one after another, never in parallel: throughput is part of
# the results, and two arms sharing the GPU would each report the other's load.
# The CPU governor is set to performance for the run and restored on exit.
# Everything is resumable -- a stage skips what its summary already holds -- so
# re-running continues rather than restarting.

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
# Direct timing runs of the replayed configurations use the first three.
TIMING_PARTICIPANTS="$(echo $PARTICIPANTS | tr ' ' '\n' | head -3 | tr '\n' ' ')"

# tag = <arm>_<cameras joined by ->, e.g. nlf_0-2-4-6
DIRECT=(
    "mocap 0"
    "mmpose 0 2 4 6"   "mmpose 0 2"   "mmpose 0 4"
    "nlf 0 2 4 6"
    "fastsam 0 2 4 6"  "fastsam 0 2"  "fastsam 0 4"  "fastsam 0"
)
TIMED=( "nlf2d 0 2 4 6"  "nlf2d 0 2"  "nlf2d 0 4"  "nlf 0 2"  "nlf 0 4"  "nlf 0" )
SCORED="mmpose_0-2-4-6 mmpose_0-2 mmpose_0-4 nlf2d_0-2-4-6 nlf2d_0-2 nlf2d_0-4 nlf_0-2-4-6 nlf_0-2 nlf_0-4 nlf_0 fastsam_0-2-4-6 fastsam_0-2 fastsam_0-4 fastsam_0"

GOVERNORS=/sys/devices/system/cpu/cpufreq/policy*/scaling_governor
PREVIOUS_GOVERNOR="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
restore_governor () {
    [ -n "$PREVIOUS_GOVERNOR" ] && for g in $GOVERNORS; do echo "$PREVIOUS_GOVERNOR" > "$g"; done 2>/dev/null
}
trap restore_governor EXIT
trap 'exit 130' INT TERM
for g in $GOVERNORS; do echo performance > "$g"; done 2>/dev/null

tag_of () { local arm="$1"; shift; [ "$arm" = mocap ] && echo mocap_reference || echo "${arm}_$(echo "$*" | tr ' ' '-')"; }

sweep_stage () {    # sweep_stage <runs root> <summary dir> <participants> <arm> <cameras...>
    local runs="$1" summaries="$2" people="$3" arm="$4"; shift 4
    local tag; tag="$(tag_of "$arm" "$@")"
    echo "=== $(date '+%F %T')  $tag"
    mkdir -p "$summaries"
    for p in $people; do
        # The proposed pipeline records its NLF output for the replay stages.
        cache=(); [ "$tag" = nlf_0-2-4-6 ] && [ "$runs" = "$RUNS" ] && cache=(--views-cache "$RESULTS/views")
        timeout 7200 python3 "$PY/sweep.py" --arm "$arm" --dataset "$DATASET" --cameras "$@" \
            --summary "$summaries/$tag.csv" --participants "$p" --tasks $TASKS \
            --tag "$tag" --output-dir "$runs" "${cache[@]}" >> "$LOGS/$tag.log" 2>&1
        rc=$?
        ok=$(grep -cE ",ok[[:space:]]*$" "$summaries/$tag.csv" 2>/dev/null)
        echo "  $(date '+%T') $p rc=$rc, $ok trials ok so far"
    done
}

echo "=== $(date '+%F %T')  campaign $CAMPAIGN: participants $PARTICIPANTS"
{   # what this campaign ran on, for the handoff
    echo "started: $(date '+%F %T')"; echo "participants: $PARTICIPANTS"
    echo "timing participants for replayed configurations: $TIMING_PARTICIPANTS"
    echo "--- git"; git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --short
    echo "(uncommitted changes, if any, are saved in code.diff next to this file)"
    echo "--- nvidia-smi"; nvidia-smi
    echo "--- cpu"; lscpu | grep -E "Model name|^CPU\(s\)"
    echo "--- governor"; cat $GOVERNORS | sort | uniq -c
    echo "--- python"; python3 -c "import numpy, scipy, torch, pinocchio; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'pinocchio', pinocchio.__version__)"
} > "$RESULTS/environment.txt" 2>&1
git -C "$REPO" diff HEAD > "$RESULTS/code.diff"
git -C "$REPO" ls-files --others --exclude-standard | grep -E "^(src|scripts|settings)" | while read -r f; do
    git -C "$REPO" diff --no-index /dev/null "$f"; done >> "$RESULTS/code.diff"

# 1. direct arms
for spec in "${DIRECT[@]}"; do
    sweep_stage "$RUNS" "$RESULTS" "$PARTICIPANTS" $spec
done

# 2. direct timing runs of the configurations that are otherwise replayed
for spec in "${TIMED[@]}"; do
    sweep_stage "$REPO/output/${CAMPAIGN}_timing" "$RESULTS/timing_runs" "$TIMING_PARTICIPANTS" $spec
done

# 3. replayed NLF configurations
echo "=== $(date '+%F %T')  replayed NLF configurations"
python3 "$PY/replay_nlf_arms.py" --views "$RESULTS/views" --output-dir "$RUNS" \
    --results "$RESULTS" > "$LOGS/replay_arms.log" 2>&1
grep -E "replays to run|replays done|Traceback" "$LOGS/replay_arms.log"

# 4. design studies
echo "=== $(date '+%F %T')  design studies E3-E5"
python3 "$PY/ik_filter_studies.py" --views "$RESULTS/views" --output-dir "$RUNS" \
    --out "$RESULTS/paper" > "$LOGS/studies.log" 2>&1
grep -E "trials x|OCP|baseline replay|study rows|Traceback" "$LOGS/studies.log"

# 5. metrics
echo "=== $(date '+%F %T')  metrics"
python3 "$PY/rescore_summaries.py" --results "$RESULTS" --output-dir "$RUNS" 2>&1 | tee "$LOGS/rescore.log"
common=(--output-dir "$RUNS" --results "$RESULTS" --out "$RESULTS/paper")
python3 "$PY/trial_metrics.py" --arms $SCORED "${common[@]}" 2>&1 | tee "$LOGS/trial_metrics.log"
python3 "$PY/reba_agreement.py" --arms $SCORED "${common[@]}" 2>&1 | tee "$LOGS/reba.log"
python3 "$PY/robot_distance.py" --arms $SCORED "${common[@]}" 2>&1 | tee "$LOGS/robot_distance.log"
python3 "$PY/fastsam_timing.py" --out "$RESULTS/paper" 2>&1 | tee "$LOGS/fastsam_timing.log"
python3 "$PY/aggregate_results.py" --arms $SCORED --root "$RESULTS/paper" --results "$RESULTS" \
    2>&1 | tee "$LOGS/aggregate.log"
cp "$RESULTS/code.diff" "$LOGS/code.diff" 2>/dev/null
python3 "$PY/build_handoff.py" --campaign "$CAMPAIGN" 2>&1 | tee "$LOGS/handoff.log"
echo "=== $(date '+%F %T')  campaign $CAMPAIGN done: copy $RESULTS/handoff to the paper workspace"
