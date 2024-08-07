from utils.model_utils import Robot, model_scaling
from utils.read_write_utils import formatting_keypoints_data, remove_nans_from_list_of_dicts, set_zero_data
from utils.viz_utils import Rquat, place
from utils.calib_utils import load_cam_pose
from utils.ik_utils import IK_Quadprog, RT_Quadprog
import pandas as pd
import sys
import pinocchio as pin 
from pinocchio.visualize import GepettoVisualizer
import numpy as np 
import time 

# Loading human urdf
human = Robot('models/human_urdf/urdf/human.urdf','models') 
human_model = human.model
human_visual_model = human.visual_model
human_collision_model = human.collision_model

# Loading joint center positions
data = pd.read_csv('output/keypoints_3D_pos_2RGB_2.csv')

# Convert X, Y, Z columns to a numpy array
keypoints = data[['X', 'Y', 'Z']].values  # Shape: (3, N) where N is the number of keypoints

# Loading camera pose 
R1_global, T1_global = load_cam_pose('cams_calibration/cam_params/camera1_pose_test_test.yml')
R1_global = R1_global@pin.utils.rotate('z', np.pi) # aligns measurements to human model definition

# Subtract the translation vector (shifting the origin)
keypoints_shifted = keypoints - T1_global.T

# Apply the rotation matrix to align the points
keypoints_transformed = np.dot(keypoints_shifted,R1_global)

# Update the DataFrame with the transformed points, replacing the old X, Y, Z values
data['X'], data['Y'], data['Z'] = keypoints_transformed[:, 0], keypoints_transformed[:, 1], keypoints_transformed[:, 2]
set_zero_data(data)

keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]

orange = [1,165/255,0,1]
blue = [0,0,1,1]
green = [0,1,0,1]

color_values=[blue,blue,blue,blue,blue,green,orange,green,orange,green,orange,green,orange,green,orange,green,orange,green,orange,green,orange,green,orange]
color = dict(zip(keypoint_names,color_values))

# Scaling segments lengths 
human_model,human_data=model_scaling(human_model, data[(data['Frame']==1)])

# Visualisation

viz = GepettoVisualizer(human_model,human_collision_model,human_visual_model)
try:
    viz.initViewer()
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install gepetto-viewer"
    )
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print(
        "Error while loading the viewer model. It seems you should start gepetto-viewer"
    )
    print(err)
    sys.exit(0)

for keypoint in keypoint_names:
    viz.viewer.gui.addSphere('world/'+keypoint,0.01,color[keypoint])

# IK calculations 
q0 = pin.neutral(human_model)
dt = 1/30
keys_to_track_list = ["Right Knee","Right Hip","Right Shoulder","Right Elbow","Right Wrist"]
dict_dof_to_keypoints = dict(zip(keys_to_track_list,['knee_Z', 'lumbar_Z', 'shoulder_Z', 'elbow_Z', 'hand_fixed']))

meas_list = formatting_keypoints_data(data)
meas_list = remove_nans_from_list_of_dicts(meas_list)

# ik_problem = IK_Quadprog(human_model, meas_list, q0, keys_to_track_list,dt,dict_dof_to_keypoints,False)
ik_problem = RT_Quadprog(human_model, meas_list[0], q0, keys_to_track_list,dt,dict_dof_to_keypoints,False)

q = ik_problem.solve_ik_sample()
q=np.array(q)

for ii in range(1,data['Frame'].iloc[-1]+1): 
    ik_problem = RT_Quadprog(human_model, meas_list[ii-1], q, keys_to_track_list,dt,dict_dof_to_keypoints,False)
    t1= time.time()
    q = ik_problem.solve_ik_sample()
    t2=time.time()
    print('Time spent = ',t2-t1)
    q=np.array(q)
    q_ii = q
    for name in keypoint_names:
        # Filter the DataFrame for the specific frame and keypoint
        keypoint_data = data[(data['Frame'] == ii) & (data['Keypoint'] == name)]
        if not keypoint_data.empty:
            M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([keypoint_data.iloc[0]['X'],keypoint_data.iloc[0]['Y'],keypoint_data.iloc[0]['Z']]).T))
            place(viz,'world/'+keypoint_data.iloc[0]['Keypoint'],M)
            viz.display(q_ii)
    input()