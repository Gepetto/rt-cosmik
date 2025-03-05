from dataclasses import dataclass, field
@dataclass
class Settings:
    # SAVE 
    SAVE_VID: bool = False
    SAVE_CSV: bool = False
    SAVE_DIR: str = "/root/workspace/ros_ws/src/rt-cosmik/output" # abs path to the save folder

    # CAM PARAMS
    fps: int = 40
    dt: float = field(init=False)  # Mark `dt` as excluded from the constructor
    width: int = 1280 # image resolution
    height: int = 720 # image resolution
    fourcc: str = "MJPG" # video codec

    # VIEWER PARAMS
    viewer: str = "gv" # viewer type: gv or ros
    
    # CALIB
    cam_calib_path: str = "" # relative path to the camera calibration file
    human_calib_path: str = "" # relative path to the human calibration file
    robot_calib_path: str = "" # relative path to the robot calibration file

    # FILTER PARAMS
    order: int = 4
    system_freq: int = 40 # For now the system update time is at around 0.034 ms so around 30 Hz 
    cutoff_freq: float = 10
    filter_type: str = "lowpass"

    # IK AND DATA HANDLING
    # For planar case 
    side_to_track: str =  "right" # bilateral (if we want to track both side, i.e., lifting), right or left

    #MMPOSE MODELS (here body 26)
    det_model_path: str = "/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano" # absolute path
    pose_model_path: str = "/root/workspace/mmdeploy/rtmpose-trt/rtmpose-s" # absolute path 

    keypoints_names: list = field(default_factory=lambda: [
        "Nose", "LEye", "REye", "LEar", "REar", 
        "LShoulder", "RShoulder", "LElbow", "RElbow", 
        "LWrist", "RWrist", "LHip", "RHip", 
        "LKnee", "RKnee", "LAnkle", "RAnkle", "Head",
        "Neck", "midHip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
    ])
    
    # OPENCAP 
    # HUMAN ANTHROPOMETRY
    human_height: float = 1.81
    human_mass: float = 74.0   

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
    
    # Add this to the class definition
    keys_to_track_list: list = field(default_factory=lambda: [
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


    def __post_init__(self):
        self.dt = 1 / self.fps  # Compute `dt` after initialization


