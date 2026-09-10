#!/bin/bash

subjects="all"
tasks="all"

base_dir="/root/workspace/RT-COSMIK-paper-offline/"
output_dir="$base_dir/output"
log_dir="$output_dir/logs"
mkdir -p "$log_dir"

# Resolve "all" -> actual subject directory names under output_dir
if [ "$subjects" = "all" ]; then
    subjects=$(find "$output_dir" -mindepth 1 -maxdepth 1 -type d ! -name "logs" -printf "%f\n" | sort)
fi

for p in $subjects; do

    # Resolve "all" -> actual task directory names under output_dir/$p
    if [ "$tasks" = "all" ]; then
        task_list=$(find "$output_dir/$p" -mindepth 1 -maxdepth 1 -type d ! -name "eval_*" -printf "%f\n" | sort)
    else
        task_list="$tasks"
    fi

    for t in $task_list; do
        summary_file="$log_dir/${p}_${t}.log"

        python3 ../python/eval/compare_to_mocap.py \
            --reference $base_dir/data/mocap_old/aligned/$p/$t/ \
            1cam=$output_dir/$p/$t/1cam_mhe_acados \
            2cam=$output_dir/$p/$t/2cam_mhe_acados/ \
            4cam=$output_dir/$p/$t/4cam_mhe_acados \
            --plots $output_dir/$p/eval_$t \
            2>&1 | tee "$summary_file"

        status=${PIPESTATUS[0]}
        if [ $status -ne 0 ]; then
            echo "FAILED $p/$t" | tee -a "$summary_file"
        fi
    done
done