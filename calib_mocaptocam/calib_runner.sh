#!/bin/bash

echo "Before running the calibration, calibrate Vicon Mocap system. When it's done type P"

while true; do
    read -n 1 -s key
    if [ "$key" = "p" ]; then
        break
    fi
done

echo "Running calibration..."

echo "Running cam1 intrinsics..."
python3 ../cams_calibration/scripts/rgb/calibrate_camera_indiv.py 1

echo "Running cam2 intrinsics..."
python3 ../cams_calibration/scripts/rgb/calibrate_camera_indiv.py 2

echo "Setting world frame..."
python3 ../cams_calibration/scripts/rgb/set_world_frame.py

echo "Running cam1 extrinsics..."
python3 calib_mocaptocam/calib_mocap_to_cam_indiv.py 1

echo "Running cam2 extrinsics..."
python3 calib_mocaptocam/calib_mocap_to_cam_indiv.py 2

echo "Calib cam2cam..."
python3 calib_mocaptocam/cam2cam_calib.py

echo "Calibration done!"