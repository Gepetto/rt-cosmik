#!/bin/bash

read -p "Enter subject name: " subject
read -p "Enter trial name: " trial
read -p "Use mocap data (T/F)? " use_mocap
read -p "Add noise (T/F)? " add_noise
read -p "Use weights (T/F)? " use_weights
read -p "Rotation probability (0-1)? " rot_prob
read -p "Rotation max degree (0-180)? " rot_max_deg
read -p "Rotation scheme (off/prob/det)? " rotation_scheme
read -p "Rotation number ? " rotation_number

python convert_tf2onnx.py --body-part upper --use-mocap $use_mocap --add-noise $add_noise --fine-tune F --add-layer T --use-weights F \
 --rot-prob $rot_prob --rot-max-deg $rot_max_deg --rotation-scheme $rotation_scheme --n-rotations $rotation_number 
python convert_tf2onnx.py --body-part lower --use-mocap $use_mocap --add-noise $add_noise --fine-tune F --add-layer T --use-weights $use_weights \
 --rot-prob $rot_prob --rot-max-deg $rot_max_deg --rotation-scheme $rotation_scheme --n-rotations $rotation_number
python run_marker_augmenter.py --subject $subject --trial $trial --use-mocap $use_mocap --add-noise $add_noise --fine-tune F --add-layer T \
 --use-weights $use_weights --rot-prob $rot_prob --rot-max-deg $rot_max_deg --rotation-scheme $rotation_scheme --n-rotations $rotation_number
python ../process_data_manip/z_check_multiple_mks_gp_v.py --subject $subject --trial $trial --use-mocap $use_mocap --add-noise $add_noise --fine-tune F --add-layer T \
 --use-weights $use_weights --rot-prob $rot_prob --rot-max-deg $rot_max_deg --rotation-scheme $rotation_scheme --n-rotations $rotation_number