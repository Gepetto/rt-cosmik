#!/bin/bash
echo "Running verification..."
echo "Please enter subject name and trial..."

read trial

python scripts/frames_counter.py ./output/Anastasia/$trial/camera_0.mp4
python scripts/frames_counter.py ./output/Anastasia/$trial/camera_2.mp4
python scripts/frames_counter.py ./output/Anastasia/$trial/camera_4.mp4
python scripts/frames_counter.py ./output/Anastasia/$trial/camera_6.mp4
python scripts/csv_counter.py ./output/Anastasia/$trial/mks_data.csv

python scripts/get_real_fps.py ./output/Anastasia/$trial/camera_0_timestamps.csv
python scripts/get_real_fps.py ./output/Anastasia/$trial/camera_2_timestamps.csv
python scripts/get_real_fps.py ./output/Anastasia/$trial/camera_4_timestamps.csv
python scripts/get_real_fps.py ./output/Anastasia/$trial/camera_6_timestamps.csv