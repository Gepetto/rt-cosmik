#!/bin/bash

echo "Running calibration..."

# Utiliser v4l2-ctl pour lister les dispositifs vidéo et capturer les index
mapfile -t cameras < <(v4l2-ctl --list-devices | grep -oP 'video\d+' | sort -u)

echo "Cameras detected: ${cameras[*]}"


echo "Before continuing the calibration, calibrate Vicon Mocap system. When it's done, stream Vicon data and then press P"

while true; do
    read -n 1 -s key
    if [ "$key" = "p" ]; then
        break
    fi
done

echo "Take the QR code and put it in front of the cameras to take pictures of it"

echo "Be sure that the QR code is visible by the cameras and that you are streaming data from Vicon or there could be a bug"

echo "When the QR code is in front of the first camera, press P another time"

# Boucle pour calibrer les extrinsèques de chaque caméra
for cam in "${cameras[@]}"; do
    while true; do
        # Extraire le numéro de la caméra de "videoX"
        cam_num=${cam#video}
        echo "Running cam$cam_num extrinsics..."
        python3 calib_mocaptocam/calib_mocap_to_cam_indiv.py $cam_num

        read -p "Do you want to retry this step? (y/n) " answer
        if [ "$answer" != "y" ]; then
            break
        else
            echo "Removing files generated in this step..."
            # Ajoutez ici la commande pour supprimer les fichiers générés lors de cette étape
            rm -r /home/ngouget/Codes/rt-cosmik/config/images_calib_cam${cam_num}
        fi
    done
done

echo "Calib cam2cam..."
python3 calib_mocaptocam/all_cam2cam_calib.py

while true; do
    echo "Setting world frame..."
    python3 ../cams_calibration/scripts/rgb/set_world_frame_all.py

    read -p "Do you want to retry this step? (y/n) " answer
    if [ "$answer" != "y" ]; then
        break
    else
        echo "Removing files generated in this step..."
        # Ajoutez ici la commande pour supprimer les fichiers générés lors de cette étape
        rm -r /home/ngouget/Codes/rt-cosmik/config/cam_params/pose_cam${cam_num}.yaml
        rm -r /home/ngouget/Codes/rt-cosmik/config/images_world_${cam_num}
    fi
done

echo "Calibration done!"
