
import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

import pandas as pd
from collections import deque
import numpy as np
from scipy import signal
from augmenter.marker_augmenter import augmentTRC, loadModel

subject_mass = 73.0
subject_height = 1.81
augmenter_path = '/root/workspace/ros_ws/src/rt-cosmik/augmentation_model'
path_to_3d_kpt = '/root/workspace/ros_ws/src/rt-cosmik/output/keypoints_3d_test.csv'

keypoints_buffer = deque(maxlen=30)

def main():
    first_frame = True
    #load lstm model
    warmed_models= loadModel(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3')

    #load 3d keypoints
    data = pd.read_csv(path_to_3d_kpt).values
    num_columns = data.shape[1]

    if num_columns % 3 != 0:
        raise ValueError(f"Unexpected number of columns: {num_columns}. It should be divisible by 3.") #60/3 = 20

    num_keypoints = num_columns // 3
    coordinates_per_keypoint = 3
    
    #apply the lstm on the data
    for i in range(len(data)):
        #reshape each frame into a (num_keypoints, 3) array and add to the buffer
        frame_data = data[i].reshape(num_keypoints, coordinates_per_keypoint)

        if first_frame:
            for _ in range(30):
                keypoints_buffer.append(np.array(frame_data))
            first_frame = False 
        else:
            keypoints_buffer.append(np.array(frame_data))

        if len(keypoints_buffer) == 30:

            keypoints_buffer_array = np.array(keypoints_buffer)
            augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=subject_mass, subject_height=subject_height, models = warmed_models,
                                augmenterDir=augmenter_path, augmenter_model='v0.3')
            print(augmented_markers)

if __name__ == "__main__":
    main()
