from dataclasses import dataclass, field
import numpy as np

@dataclass
class Settings:
    # CAM PARAMS
    fs: int = 40
    dt: float = field(init=False)  # Mark `dt` as excluded from the constructor
    width: int = 1280 # image resolution
    height: int = 720 # image resolution

    # CAMS CALIB 
    # checkerboard params
    checkerboard_rows: int = 6 # number of rows on the checkerboard -1 
    checkerboard_columns: int = 7 # number of columns on the checkerboard -1 
    checkerboard_scaling: float = 0.108 # size of squares in meters
    # wand params
    wand_end_effector_local_pos: np.ndarray = field(
        default_factory=lambda: np.array([[-0.00004], [0.262865], [-0.000009]]) # local pose of wand's end effector for pointing calibration
    )
    wand_marker_size: float = 0.176 # Marker size in meters (17.6 cm)

    # FILTER PARAMS
    order: int = 4
    system_freq: int = 30 # For now the system update time is at around 0.034 ms so around 30 Hz 
    cutoff_freq: float = 10
    filter_type: str = "lowpass"

    # IK AND DATA HANDLING
    # For planar case 
    side_to_track: str =  "right" # bilateral (if we want to track both side, i.e., lifting), right or left
    
    # HUMAN ANTHROPOMETRY
    human_height: float = 1.81
    human_mass: float = 73.0

    #MMPOSE MODEL (here body 26)
    keypoints_names: list = field(default_factory=lambda: [
        "Nose", "LEye", "REye", "LEar", "REar", 
        "LShoulder", "RShoulder", "LElbow", "RElbow", 
        "LWrist", "RWrist", "LHip", "RHip", 
        "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
        "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
    ])
    
    # OPENCAP MARKER SET 
    marker_names: list = field(default_factory=lambda: [
           'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
           'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
           'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
           'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
           'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
           'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
           'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
           'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
           'L_lwrist_study','L_mwrist_study'])
    
    def __post_init__(self):
        self.dt = 1 / self.fs  # Compute `dt` after initialization


