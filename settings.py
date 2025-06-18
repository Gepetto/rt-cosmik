from dataclasses import dataclass, field
import os 
from pathlib import Path

@dataclass
class Settings:
    cosmik_path: str = field(init=False)
    
    # SAVE 
    subject = "Test_2"
    motion = "test"
    SAVE_VID: bool = True
    SAVE_CSV: bool = True
    SAVE_DIR: str = f"C:\\Users\\krauszm\\COSMIK\\rt-cosmik\\output\\{subject}\\{motion}" # abs path to the save folder

    # HUMAN ANTHROPOMETRY
    human_height: float = 1.85
    human_mass: float = 75.0

    # CAM PARAMS
    fs: int = 30
    dt: float = field(init=False)  # Mark `dt` as excluded from the constructor
    width: int = 1280 # image resolution
    height: int = 800 # image resolution
    fourcc: str = "I420" # video codec

    # VIEWER PARAMS
    viewer: str = "gv" # viewer type: gv or ros
    urdf_path: str = field(init=False) # relative path to the robot urdf
    meshes_path: str = field(init=False) # relative path to the robot meshes

    # CALIB
    cam_calib_path: str = field(init=False) # relative path to the camera calibration file
    human_calib_path: str = field(init=False)  # relative path to the human calibration file
    robot_calib_path: str = field(init=False)  # relative path to the robot calibration file

    # FILTER PARAMS
    order: int = 4
    system_freq: int = 40 # For now the system update time is at around 0.034 ms so around 30 Hz 
    cutoff_freq: float = 7
    filter_type: str = "lowpass"

    # IK AND DATA HANDLING
    # For whole body model :
    joint_angles_names = ['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx','FF_quaty',
                          'FF_quatz', 'FF_quatw', 'Lumbar_flex_ext', 'Lumbar_int_ext_rot',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot',
                          'Relbow_flex_ext', 'Relbow_pron_supi', 'Lshoulder_flex_ext',
                          'Lshoulder_abd_add', 'Lshoulder_int_ext_rot', 'Lelbow_flex_ext',
                          'Lelbow_pron_supi','Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext','Lhip_flex_ext', 'Lhip_abd_add', 
                          'Lhip_int_ext_rot', 'Lknee_flex_ext', 'Lankle_flex_ext']

    # Ik type
    ik_type: str ="mhe" # either "mhe" for SWIKA or "sbs" for sample by sample qp
    
    # if ik_type = "mhe"
    ik_code: str = "python" # either "python" or "c" 
    cost_weights: list = field(default_factory=lambda: [1, 1e-3, 1e-5])
    N: int = 10 # number of time steps
    
    # For planar case 
    side_to_track: str =  "right" # bilateral (if we want to track both side, i.e., lifting), right or left

    #MMPOSE MODELS (here body 26)
    det_model_path: str = "/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano" # absolute path
    pose_model_path: str = "/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m" # absolute path 

    keypoints_names: list = field(default_factory=lambda: [
        "Nose", "LEye", "REye", "LEar", "REar", 
        "LShoulder", "RShoulder", "LElbow", "RElbow", 
        "LWrist", "RWrist", "LHip", "RHip", 
        "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
        "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
    ])
    
    # OPENCAP 
    # AUGMENTER MODEL 
    augmenter_model: str = field(init=False)


    # MARKER SET 
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
    
    marker_mocap_names: list = field(default_factory=lambda: ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study',
             'TV8','TV12','SJN','STRN','C7_study','r_shoulder_study','L_shoulder_study',
             'BHD','RHD','LHD','FHD',
             'L_lelbow_study','L_melbow_study','LUArm','L_lwrist_study','L_mwrist_study','LForearm','LHand','LHL2','LHM5',
             'r_lelbow_study','r_melbow_study','RUArm','r_lwrist_study','r_mwrist_study','RForearm','RHand','RHL2','RHM5',
             'L_thigh1_study','L_knee_study','L_mknee_study','L_sh1_study','L_ankle_study','L_mankle_study','L_calc_study','L_5meta_study','L_toe_study',
             'r_thigh1_study','r_knee_study','r_mknee_study','r_sh1_study',
             'r_ankle_study','r_mankle_study','r_calc_study','r_5meta_study','r_toe_study',
             'r_pelvis', 'l_pelvis'])


    
    # Add this to the class definition
    keys_to_track_list: list = field(default_factory=lambda: [
        'Head', 'Nose', 'REar', 'LEar', 'REye', 'LEye',
        'C7_study', 
        'r.ASIS_study', 'L.ASIS_study', 
        'r.PSIS_study', 'L.PSIS_study', 
        'r_shoulder_study',
        'r_lelbow_study', 'r_melbow_study',
        'r_lwrist_study', 'r_mwrist_study',
        'r_ankle_study', 'r_mankle_study',
        'r_toe_study','r_5meta_study', 'r_calc_study',
        'r_knee_study', 'r_mknee_study',
        'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
        'r_sh1_study', 'r_sh2_study', 'r_sh3_study',
        'L_shoulder_study', 
        'L_lelbow_study', 'L_melbow_study',
        'L_lwrist_study','L_mwrist_study',
        'L_ankle_study', 'L_mankle_study', 
        'L_toe_study','L_5meta_study', 'L_calc_study',
        'L_knee_study', 'L_mknee_study',
        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
        'L_sh1_study', 'L_sh2_study', 'L_sh3_study'
    ])

    keys_to_track_list_mocap: list = field(default_factory=lambda: [
        'LBHD','RBHD','LFHD','RFHD',
        'C7_study', 
        'r.ASIS_study', 'L.ASIS_study', 
        'r.PSIS_study', 'L.PSIS_study', 
        'r_shoulder_study',
        'r_lelbow_study', 'r_melbow_study',
        'r_lwrist_study', 'r_mwrist_study',
        'r_ankle_study', 'r_mankle_study',
        'r_toe_study','r_5meta_study', 'r_calc_study',
        'r_knee_study', 'r_mknee_study',
        'L_shoulder_study', 
        'L_lelbow_study', 'L_melbow_study',
        'L_lwrist_study','L_mwrist_study',
        'L_ankle_study', 'L_mankle_study', 
        'L_toe_study','L_5meta_study', 'L_calc_study',
        'L_knee_study', 'L_mknee_study'
    ])


    def __post_init__(self):
        # Use Pathlib for better path handling
        self.cosmik_path = str(Path(__file__).parent.resolve())
        self.cam_calib_path = str(Path(self.cosmik_path) / "config/cam_params")
        self.human_calib_path = str(Path(self.cosmik_path) / "config/human_params")
        self.robot_calib_path = str(Path(self.cosmik_path) / "config/robot_params")
        self.augmenter_path = str(Path(self.cosmik_path) / "src/rtcosmik/augmenter/augmentation_model")
        self.urdf_path = str(Path(self.cosmik_path) / "urdf/human.urdf")
        self.meshes_path = str(Path(self.cosmik_path) / "meshes")
        self.dt = 1 / self.fs
