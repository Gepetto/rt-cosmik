#!/bin/bash

echo "Running calibration..."

echo "Take the chessboard and put it in front of the cameras to take pictures of it"

echo "Running cam1 intrinsics..."
python3 ../cams_calibration/scripts/rgb/calibrate_camera_indiv.py 1

echo "Running cam2 intrinsics..."
python3 ../cams_calibration/scripts/rgb/calibrate_camera_indiv.py 2

echo "Setting world frame..."
python3 ../cams_calibration/scripts/rgb/set_world_frame.py

echo "Before continuing the calibration, calibrate Vicon Mocap system. When it's done, stream Vicon data and the press P"

while true; do
    read -n 1 -s key
    if [ "$key" = "p" ]; then
        break
    fi
done

echo "Take the QR code and put it in front of the cameras to take pictures of it"

echo "Be sure that the QR code is visible by the cameras and that you are streaming data from Vicon or there could be a bug"

echo "Running cam1 extrinsics..."
python3 calib_mocaptocam/calib_mocap_to_cam_indiv.py 1

echo "Running cam2 extrinsics..."
python3 calib_mocaptocam/calib_mocap_to_cam_indiv.py 2

echo "Calib cam2cam..."
python3 calib_mocaptocam/cam2cam_calib.py

echo "Calibration done!"