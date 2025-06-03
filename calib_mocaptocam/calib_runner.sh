#!/bin/bash

echo "Running calibration..."

# Utiliser v4l2-ctl pour lister les dispositifs vidéo et capturer les index
mapfile -t cameras < <(v4l2-ctl --list-devices | grep -oP 'video\d+' | sort -u)

echo "Cameras detected: ${cameras[*]}"

echo "Take the chessboard and put it in front of the cameras to take pictures of it"

# Boucle pour calibrer les intrinsèques de chaque caméra détectée
for cam in "${cameras[@]}"; do
    # Extraire le numéro de la caméra de "videoX"
    cam_num=${cam#video}
    echo "Running cam$cam_num intrinsics..."
    python3 ../cams_calibration/scripts/rgb/calibrate_camera_indiv.py $cam_num
done

echo "Before continuing the calibration, calibrate Vicon Mocap system. When it's done, stream Vicon data and the press P"

while true; do
    read -n 1 -s key
    if [ "$key" = "p" ]; then
        break
    fi
done

echo "Take the QR code and put it in front of the cameras to take pictures of it"

echo "Be sure that the QR code is visible by the cameras and that you are streaming data from Vicon or there could be a bug"

# Boucle pour calibrer les extrinsèques de chaque caméra
for cam in "${cameras[@]}"; do
    # Extraire le numéro de la caméra de "videoX"
    cam_num=${cam#video}
    echo "Running cam$cam_num extrinsics..."
    python3 calib_mocaptocam/calib_mocap_to_cam_indiv.py $cam_num
done

echo "Calib cam2cam..."
python3 calib_mocaptocam/all_cam2cam_calib.py

echo "Setting world frame..."
python3 ../cams_calibration/scripts/rgb/set_world_frame_all.py

echo "Calibration done!"