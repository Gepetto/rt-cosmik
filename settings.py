from dataclasses import dataclass, field
import os 
from pathlib import Path

@dataclass
class Settings:
    cosmik_path: str = field(init=False)
    
    # SAVE 
    no_trial = "test"
    SAVE_VID: bool = True
    SAVE_CSV: bool = True
    SAVE_DIR: str = f"/root/workspace/ros_ws/src/RT-COSMIK/output/{no_trial}" # abs path to the save folder

    # CAM PARAMS
    fs: int = 40
    dt: float = field(init=False)  # Mark `dt` as excluded from the constructor
    width: int = 1280 # image resolution
    height: int = 720 # image resolution
    fourcc: str = "MJPG" # video codec

    # HUMAN ANTHROPOMETRY
    human_height: float = 1.81
    human_mass: float = 74.0   

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
    system_freq: int = 40 
    cutoff_freq: float = 7
    filter_type: str = "lowpass"

    # NLF
    device = "cuda:0"

    cano_path: str = "/root/workspace/ros_ws/RT-COSMIK/weights/canonical_verts/smplx.npy"
    nlf_path: str = "/root/workspace/ros_ws/RT-COSMIK/weights/nlf/nlf_s_multi_0.2.2.torchscript"
    
    nlf_indices = [
        5484, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457, 4721, #c7, rshoulder,lshoulder,r_lelbow,l_lelbow,r_melbow,l_melbow,r_lwrist, l_lwrist,r_mwrist, l_mwrist
        8421, 5727, 8371, 5677, #r_asis,l_asis,r_psis,l_psis 
        6401, 3640, 6407, 3646, 8576,5882,8680,8892, #r_knee,l_knee,r_mknee,l_mknee,r_ankle,l_ankle,r_mankle,l_mankle,
        8596,5902,8589,5895,8482,5788,8846,8634, #r_5meta, l_5meta, r_toe, l_toe, r_big_toe, l_big_toe, l_calc, r_calc,
        7978,4807,8004,5268,7483,4747,7664,4928,7776,5040,7887,5151,7420,4684,8078,5342, #7978,4807, r_tpinky, l_tpinky, r_bindex, l_bindex, r_tindex, l_tindex, r_tmiddle, l_tmiddle, r_tring,l_tring, r_bthumb, l_bthumb,r_tthumb, l_tthumb
        9008,9002,1253,399,10049,9503 ,  #nose, head,right_ear,left_ear, right_eye, left_eye
        5941,5489,5500 #L2, T11, T6
    ]

    # Yolo detector
    yolo_path: str = "/root/workspace/ros_ws/RT-COSMIK/weights/yolo/yolov10n.engine"

    yolo_conf = 0.2
    yolo_imgsz = 640

    # IK AND DATA HANDLING
    # For whole body model :

    joint_angles_names = ['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx','FF_quaty',
                            'FF_quatz', 'FF_quatw', 'Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                            'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                            'Thoracic_flex_ext','Thoracic_lateral_flex','Thoracic_rot_int_ext',
                            'Lcalvicule_x',
                            'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi','Lwrist_flex_ext','Lwrist_x',
                            'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                            'rcalvicule_x',
                            'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi', 'Rwrist_flex_ext','Rwrist_x',
                            'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                            'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']

    # Ik type
    ik_type: str ="mhe" # either "mhe" for SWIKA or "sbs" for sample by sample qp
    
    # if ik_type = "mhe"
    ik_code: str = "python" # either "python" or "c" 
    cost_weights: list = field(default_factory=lambda: [1, 1e-3, 1e-5])
    N: int = 10 # number of time steps
    
    # For planar case 
    side_to_track: str =  "right" # bilateral (if we want to track both side, i.e., lifting), right or left


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
        'Nose', 'Head', 'REye', 'LEye',
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
        'L_knee_study', 'L_mknee_study',
    ])


    def __post_init__(self):
        # Use Pathlib for better path handling
        self.cosmik_path = str(Path(__file__).parent.resolve())
        self.cam_calib_path = str(Path(self.cosmik_path) / "config/cam_params")
        self.human_calib_path = str(Path(self.cosmik_path) / "config/human_params")
        self.robot_calib_path = str(Path(self.cosmik_path) / "config/robot_params")
        self.urdf_path = str(Path(self.cosmik_path) / "urdf/human.urdf")
        self.meshes_path = str(Path(self.cosmik_path) / "meshes")
        self.dt = 1 / self.fs
