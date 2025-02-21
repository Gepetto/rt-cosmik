
import os
import sys
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
rt_cosmik_path = os.path.dirname(script_directory)
# Append it to sys.path
sys.path.append(str(rt_cosmik_path))
meshes_folder_path = os.path.join(rt_cosmik_path, 'meshes')

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

augmenter_path = os.path.join(parent_directory, 'augmentation_model')

import pandas as pd
from collections import deque
import numpy as np
from scipy import signal
from utils.lstm_v2 import augmentTRC, loadModel
from utils.settings import Settings
settings = Settings()

first_frame = True

warmed_models= loadModel(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3')

def butterworth_filter(data, cutoff_frequency, order=5, sampling_frequency=60):
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data

# file_path = '/home/kahina/mmdeploy-1.0.0-linux-x86_64-cxx11abi-cuda11.3/example/python/a9fd6740-1c9d-40df-beca-15e6eecf08d7.csv'
# df = pd.read_csv(file_path, skiprows=2)
# data = df.values[:,2:]
# print(f"Data shape: {data[1]}")


data = pd.read_csv('/root/workspace/ros_ws/src/rt-cosmik/output/frontal_plan/keypoints_3d.csv')
data= data.values

#data = butterworth_filter(data, 5.0)

num_columns = data.shape[1]
if num_columns % 3 != 0:
    raise ValueError(f"Unexpected number of columns: {num_columns}. It should be divisible by 3.") #60/3 = 20

num_keypoints = num_columns // 3
coordinates_per_keypoint = 3

keypoints_buffer = deque(maxlen=30)

#reshape each frame into a (num_keypoints, 3) array and add to the buffer
for i in range(len(data)):
    frame_data = data[i].reshape(num_keypoints, coordinates_per_keypoint)

    if first_frame:
        for _ in range(30):
            keypoints_buffer.append(np.array(frame_data))
        first_frame = False 
    else:
        keypoints_buffer.append(np.array(frame_data))

    if len(keypoints_buffer) == 30:

        keypoints_buffer_array = np.array(keypoints_buffer)
        # print(keypoints_buffer_array)
        augmented_markers = augmentTRC(keypoints_buffer_array, subject_mass=settings.human_mass, subject_height=settings.human_height, models = warmed_models,
                               augmenterDir=augmenter_path, augmenter_model='v0.3', offset=True)
