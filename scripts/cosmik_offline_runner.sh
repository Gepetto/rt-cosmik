#!/bin/bash

echo "Running cosmik offline..."

echo "Enter infos in this order: subject trial weight height"

read subject trial weight height

if [ -z "$subject" ] || [ -z "$trial" ] || [ -z "$weight" ] || [ -z "$height" ]; then
    echo "Error: All four parameters are required (subject, trial, weight, height)."
    exit 1
fi

echo "Running pipeline for Subject: $subject | Trial: $trial | Weight: $weight | Height: $height"

python scripts/run_posetracker_on_video.py 2 $subject $trial
python scripts/run_posetracker_on_video.py 4 $subject $trial
python scripts/run_triangulation.py $subject $trial
python scripts/run_marker_augmenter.py $subject $trial $weight $height
python scripts/run_ik.py $subject $trial


