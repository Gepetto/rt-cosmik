#!/bin/bash

read -p "test over mocap (T/F)? " test_over_mocap
read -p "upper id ? " upper_id
read -p "lower id ? " lower_id
read -p "test subjects ? " test_subjects
# read -p "Exclude trials ? " excluded_trials
read -p "Visualize (T/F)? " visualize


python convert_tf2onnx_only_HPE.py --body-part upper --model-id $upper_id
python convert_tf2onnx_only_HPE.py --body-part lower --model-id $lower_id

python run_marker_augmenter_only_HPE.py --test-over-mocap $test_over_mocap --upper-model-id $upper_id --lower-model-id $lower_id --test-subjects $test_subjects #--excluded-trials $excluded_trials

python ../process_data_manip/z_check_multiple_mks_gp_v_only_HPE.py --test-over-mocap $test_over_mocap --upper-model-id $upper_id --lower-model-id $lower_id \
 --test-subjects $test_subjects --visualize $visualize #--excluded-trials $excluded_trials