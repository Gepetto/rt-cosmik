from dataclasses import dataclass, field
import os 
from pathlib import Path
from typing import Dict
import torch

@dataclass
class Settings:
    cosmik_path: str = field(init=False)
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    
    # SAVE 
    no_trial = "test"
    SAVE_VID: bool = False
    SAVE_CSV: bool = False
    SAVE_DIR: str = f"/root/workspace/RT-COSMIK/output/{no_trial}" # abs path to the save folder

    # CAM PARAMS
    fs: int = 40
    dt: float = field(init=False)  # Mark `dt` as excluded from the constructor
    width: int = 1280 # image resolution
    height: int = 720 # image resolution
    fourcc: str = "MJPG" # video codec

    # HUMAN ANTHROPOMETRY
    human_height: float = 1.81
    human_weight: float = 74.0 
    human_gender: str = 'm'

    # CALIB
    cam_calib_path: str = field(init=False) # relative path to the camera calibration file
    human_calib_path: str = field(init=False)  # relative path to the human calibration file
    robot_calib_path: str = field(init=False)  # relative path to the robot calibration file

    # FILTER PARAMS
    order: int = 4
    system_freq: int = 40 
    cutoff_freq: float = 5
    filter_type: str = "lowpass"

    # NLF
    cano_path: str = "/root/workspace/RT-COSMIK/weights/canonical_verts/smplx.npy"
    nlf_path: str = "/root/workspace/RT-COSMIK/weights/nlf/nlf_s_multi_0.2.2.torchscript"
    nlf_indices = [             # For SMPLX model
        8421, 5727, 8371, 5677, # pelvis: RASI, LASI, RPSI, LPSI 
        5484, 5489, 5500, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457, 4721, # upper: C7, T11, T6,  RSHO, LSHO, RELB, LELB, RMELB, LMELB, RWRI, LWRI, RMWRI, LMWRI
        8079, 5361, 7794, 5058, 8022, 5286,  # hands:  RTHU, LTHU, RMID, LMID, RPIN, LPIN
        6401, 3640, 6407, 3646, 8576, 5882, 8680, 8892, # legs: RKNE, LKNE, RMKNE, LMKNE, RANK, LANK, RMANK, LMANK,
        8474,5780,8463,5770,8635,8846, # feet: R5MHD, L5MHD, RTOE, LTOE, RHEE, LHEE,
        9120,9002,616,6,9929,9448,  # face: Nose, Head, REar, LEar, REye, LEye
    ]

    # Yolo detector
    yolo_path: str = "/root/workspace/RT-COSMIK/weights/yolo/yolov10n.engine"
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
    ik_type: str ="sbs" # either "mhe" for SWIKA or "sbs" for sample by sample qp

    # if ik_type = "mhe"
    mhe_backend: str = "fatrop" # solver backend: "fatrop" (validated reference) or "acados"
    ik_code: str = "python" # fatrop only: either "python" or "c"
    cost_weights: list = field(default_factory=lambda: [1, 1e-3, 1e-5])
    N: int = 10 # number of time steps
    # acados only: where generated C code/.so/.json go (default: <repo>/output/acados),
    # and the acados install dir (default: read from the ACADOS_SOURCE_DIR env var).
    acados_export_dir: str = None
    acados_source_dir: str = None

    # MARKER SET 
    marker_names: list = field(default_factory=lambda: [
           "RASI", "LASI", "RPSI", "LPSI",
           "C7", "T11", "T6", "RSHO", "LSHO", "RELB", "LELB", "RMELB", "LMELB", "RWRI", "LWRI", "RMWRI", "LMWRI",
           "RTHU", "LTHU", "RMID", "LMID", "RPIN", "LPIN",
           "RKNE", "LKNE", "RMKNE", "LMKNE", "RANK", "LANK", "RMANK", "LMANK",
           "R5MHD", "L5MHD", "RTOE", "LTOE", "RHEE", "LHEE", 
           "Nose", "Head", "REar", "LEar", "REye", "LEye",
           ])
    
    keys_to_track_list: list = field(default_factory=lambda: [
           "RASI", "LASI", "RPSI", "LPSI",
           "C7", "T11", "T6", "RSHO", "LSHO", "RELB", "LELB", "RMELB", "LMELB", "RWRI", "LWRI", "RMWRI", "LMWRI",
           "RTHU", "LTHU", "RMID", "LMID", "RPIN", "LPIN",
           "RKNE", "LKNE", "RMKNE", "LMKNE", "RANK", "LANK", "RMANK", "LMANK",
           "R5MHD", "L5MHD", "RTOE", "LTOE", "RHEE", "LHEE", 
           "Nose", "Head", "REar", "LEar", "REye", "LEye",
    ])

    def __post_init__(self):
        # Use Pathlib for better path handling
        self.cosmik_path = str(Path(__file__).parent.resolve())
        self.cam_calib_path = str(Path(self.cosmik_path) / "config/cam_params")
        self.human_calib_path = str(Path(self.cosmik_path) / "config/human_params")
        self.robot_calib_path = str(Path(self.cosmik_path) / "config/robot_params")
        self.dt = 1 / self.fs
