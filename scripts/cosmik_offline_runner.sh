#!/bin/bash

source /deps_ws/devel/setup.bash

echo "Running cosmik offline..."

echo "Enter infos in this order: idx_cam1 idx_cam2 subject trial weight height start end"

read idx_cam1 idx_cam2 subject trial weight height start end

if [ -z "$idx_cam1" ] || [ -z "$idx_cam2" ] || [ -z "$subject" ] || [ -z "$trial" ] || [ -z "$weight" ] || [ -z "$height" ] || [ -z "$start" ] || [ -z "$end" ] ; then
    echo "Error: All four parameters are required (idx_cam1, idx_cam2, subject, trial, weight, height)."
    exit 1
fi

echo "Running pipeline for Subject: $subject | Trial: $trial | Weight: $weight | Height: $height | start: $start | end: $end"

python scripts/run_posetracker_on_video.py $idx_cam1 $subject $trial $start $end
python scripts/run_posetracker_on_video.py $idx_cam2 $subject $trial $start $((end+1))
python scripts/frames_number_correcter.py $idx_cam1 $idx_cam2 $subject $trial
python scripts/run_triangulation.py $idx_cam1 $idx_cam2 $subject $trial
python scripts/run_marker_augmenter.py $subject $trial $weight $height
python scripts/run_ik.py $subject $trial


