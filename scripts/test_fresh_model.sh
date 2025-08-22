#!/bin/bash

read -p "Enter subject name: " subject
read -p "Enter trial name: " trial

python convert_tf2onnx.py --body-part upper --use-mocap T --add-noise F --fine-tune F --add-layer T --use-weights F
python convert_tf2onnx.py --body-part lower --use-mocap T --add-noise F --fine-tune F --add-layer T --use-weights F
python run_marker_augmenter.py $subject $trial
python ../process_data_manip/z_check_multiple_mks_gp_v.py $subject $trial