import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

rt_cosmik_path = os.path.dirname(script_directory)
from src.rtcosmik.human_model.urdf_model import * 
from pinocchio.visualize import GepettoVisualizer
import pinocchio as pin 
import numpy as np 
import sys
from src.rtcosmik.utils.read_write_utils import read_mks_data,udp_csv_to_dataframe,marker_data_to_dataframe,read_subject_info
import pandas as pd
from src.rtcosmik.human_model.urdf_model import * 
from src.rtcosmik.viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from src.rtcosmik.config_loader import settings
from src.rtcosmik.human_model.model_utils import get_segment_length
from scipy import signal

def calculate_first_second_order_differentiation(model,q_data,dt):
    """_This function calculates the derivatives (velocities and accelerations here) by central difference for given angular configurations accounting that the robot has a freeflyer or not (which is indicated in the params_settings)._

    Args:
        model (_model_): _Pinocchio model_
        q (_array_): _the angular configurations whose derivatives need to be calculated_
        param (_dict_): _a dictionnary containing the settings_
        dt (_list_, optional): _ a list containing the different timesteps between the samples (set to None by default, which means that the timestep is constant and to be found in param['ts'])_. Defaults to None.

    Returns:
        _array_: _angular configurations (whose size match the samples removed by central differences)_
        _array_: _angular velocities_
        _array_: _angular accelerations_
    """

    q = q_data.copy()
    dq = np.zeros([q.shape[0]-1, q.shape[1]-1])
    ddq = np.zeros([q.shape[0]-1, q.shape[1]-1])
    
    for ii in range(q.shape[0]-1):
        dq[ii,:] = pin.difference(model,q[ii,:],q[ii+1,:])/dt[ii]

    for jj in range(model.nq-1):
        ddq[:,jj] = np.gradient(dq[:,jj], edge_order=1) / dt

    q = np.delete(q, len(q)-1, 0)
    q = np.delete(q, len(q)-1, 0)

    dq = np.delete(dq, len(dq)-1, 0)
    ddq = np.delete(ddq, len(ddq)-1,0)

    return q, dq, ddq

def low_pass_filter_data(data,dt,cutoff=10,nbutter=5):
    """_This function filters and elaborates data used in the identification process. The filter used is a zero phase lag butterworth filter_

    Args:
        data (_array_): _The data to filter_
        dt (_float_): _The time step_
        cutoff (_float_): _The cutoff frequency_
        nbutter (_int_): _Order of the butterworth filter_ Defaults to 5

    Returns:
        _array_: _The filtered data_
    """

    b, a = signal.butter(nbutter, dt*cutoff / 2, "low")
   
    #data = signal.medfilt(data, 3)
    data= signal.filtfilt(
            b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1) )
    
    
    # suppress end segments of samples due to the border effect
    nbord = 5 * nbutter
    data = np.delete(data, np.s_[0:nbord], axis=0)
    data = np.delete(data, np.s_[(data.shape[0] - nbord): data.shape[0]], axis=0)
     
    return data

info_path = f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/info.txt"
subject_height,subject_mass, gender = read_subject_info(info_path) 

rt_cosmik_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_csv = f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mocap/squat/mks_data_gapfilled.csv"

mks_to_skip = ['LForearm','LUArm', 'RUArm', 'RHJC_study','LHJC_study','r_pelvis','l_pelvis','LHL2','LHM5','RHL2','RHM5',
                'LHand', 'RForearm','RHand', 'L_sh1_study', 'L_thigh1_study','r_sh1_study', 'r_thigh1_study']

mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
            'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
            'BHD','RHD','LHD','FHD',
            'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
            'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
            'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
            'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
            'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
            'r_pelvis', 'l_pelvis']

# Load UDP CSV (wide format)
df_wide = udp_csv_to_dataframe(path_to_csv, mks_names)
# df_wide = pd.read_csv(path_to_csv)
# df_wide.columns = [col.replace(f"{no_trial}:", "") for col in df_wide.columns]
# frames = df_wide["Frame"] if "Frame" in df_wide.columns else range(len(df_wide))
# mks_names = sorted(set(col.rsplit("_", 1)[0] for col in df_wide.columns if "_x" in col))
result_markers, start_sample_dict = read_mks_data(df_wide, start_sample=0, converter = 1.0)

# Load URDF
human = Robot('/home/msabbah/pinocchio-3x/src/rt-cosmik/urdf/human.urdf', rt_cosmik_path, isFext=True)
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

human_model = scale_human_model(human_model, start_sample_dict, with_hand=True, gender=gender, subject_height=subject_height)
human_model = mks_registration(human_model, start_sample_dict, with_hand=True)
human_data = pin.Data(human_model)

#########################################################LOCK JOINTS 
all_joint_ids = set(range(1, human_model.njoints))
joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", "left_wrist_X", "left_wrist_Z", "right_wrist_X","right_wrist_Z"]
joint_ids_to_lock = []
for jn in joints_to_lock:
    if human_model.existJointName(jn):
        joint_ids_to_lock.append(human_model.getJointId(jn))
    else:
        print('Warning: joint ' + str(jn) + ' does not belong to the model!')

q0 = pin.neutral(human_model)
# Build reduced model
human_model, human_visual_model = pin.buildReducedModel(
    human_model, human_visual_model, joint_ids_to_lock, q0)

print(human_model.nq)
human_data = pin.Data(human_model)
#######################################################################################

# Load human kinematics and force

# Load human kinematics and force
csv_path = "/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mouv/squat/camera_0_timestamps.csv"

# Read with header; don't force dtypes
df = pd.read_csv(csv_path)

# Pick the timestamp column (named 'timestamp' in your file)
if "timestamp" in df.columns:
    ts_series = df["timestamp"].astype(str)
else:
    # Fallback: take the last column as timestamps
    ts_series = df.iloc[:, -1].astype(str)

# Parse timestamps (strict first, then flexible as fallback)
try:
    ts = pd.to_datetime(ts_series.str.strip(), format="%Y-%m-%d %H:%M:%S.%f", errors="raise")
except ValueError:
    ts = pd.to_datetime(ts_series.str.strip(), errors="coerce")
    if ts.isna().any():
        bad = ts_series[ts.isna()].head(5).tolist()
        raise ValueError(f"Some timestamps could not be parsed. Examples: {bad}")

# Inter-frame gaps in seconds as float
dt = ts.diff().dt.total_seconds().to_numpy()[1:]
dt_filt = dt[25:-25]
dt_mean = float(np.mean(dt))

q_data = pd.read_csv(f"/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mocap/squat/q_mocap.csv").to_numpy()

# Filter q data
for ii in range(human_model.nq):
    if ii == 0:
        q_filt= low_pass_filter_data(q_data[:,ii], dt_mean)
    else:
        q_filt = np.column_stack((q_filt,low_pass_filter_data(q_data[:,ii], dt_mean)))

q,dq,ddq = calculate_first_second_order_differentiation(human_model,q_filt,dt_filt)


# Path to the synced file you generated
force_csv_path = "/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/mocap/squat/force_resampled_40Hz.csv"  # change if needed

# Load (parse timestamp to pandas datetime)
df = pd.read_csv(force_csv_path, parse_dates=["timestamp"])

# Helper to pick exactly Fx, Fy, Fz for a given sensor ID
def get_force_cols(df, sensor_id: int):
    wanted = [f"Sensix_{sensor_id}_Fx", f"Sensix_{sensor_id}_Fy", f"Sensix_{sensor_id}_Fz"]
    # Be tolerant to column order; ensure they exist
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for Sensix_{sensor_id}: {missing}")
    return wanted

s2_cols = get_force_cols(df, 2)
s3_cols = get_force_cols(df, 3)

# Keep just camera info + the two 3D force triplets
df_forces = df[["camera_frame", "timestamp"] + s2_cols + s3_cols].copy()

# Optionally: convert to NumPy arrays (N, 3) for each sensor
F2 = df[s2_cols].to_numpy(dtype=float)  # Sensix 2 forces [Fx, Fy, Fz]
F3 = df[s3_cols].to_numpy(dtype=float)  # Sensix 3 forces [Fx, Fy, Fz]

# reexpress as the reaction force instead
F2[:, 0] = -F2[:, 0]
F3[:, 0] = -F3[:, 0]

F2[:, 2] = -F2[:, 2]
F3[:, 2] = -F3[:, 2]

# remove mean of forces X and Y and filter

F2[:, :2] -=  F2[:, :2].mean(axis=0)
F3[:, :2] -=  F3[:, :2].mean(axis=0)

for ii in range(3):
    if ii == 0:
        F2_filt = low_pass_filter_data(F2[:,ii], dt_mean)
        F3_filt = low_pass_filter_data(F3[:,ii], dt_mean)
    else:
        F2_filt = np.column_stack((F2_filt, low_pass_filter_data(F2[:,ii], dt_mean)))
        F3_filt = np.column_stack((F3_filt, low_pass_filter_data(F3[:,ii], dt_mean)))

F2_filt = F2_filt[:-2,:]
F3_filt = F3_filt[:-2,:]

for ii in range(q.shape[0]):
    pin.framesForwardKinematics(human_model, human_data, q_data[ii,:])

    fs_ext = [pin.Force(np.zeros(6)) for _ in range(len(human_model.joints))]
    
    M_left_ankle_mocap = human_data.oMi[human_model.getJointId('left_ankle_X')]
    fs_ext[human_model.getJointId("left_ankle_X")]=pin.Force(np.hstack((F2_filt[ii,:],np.zeros(3)))).se3ActionInverse(M_left_ankle_mocap)

    M_right_ankle_mocap = human_data.oMi[human_model.getJointId('right_ankle_X')]
    fs_ext[human_model.getJointId("right_ankle_X")]=pin.Force(np.hstack((F3_filt[ii,:],np.zeros(3)))).se3ActionInverse(M_right_ankle_mocap)

    tau = pin.rnea(human_model, human_data, q[ii,:], dq[ii,:], ddq[ii,:], fs_ext)
