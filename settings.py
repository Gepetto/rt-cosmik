from dataclasses import dataclass, field
import os 
from pathlib import Path
from typing import Dict
import torch

@dataclass
class Settings:
    cosmik_path: str = field(init=False)
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    
    ### WHERE RESULTS ARE SAVED ###
    #
    # Everything RT-COSMIK writes goes under one folder: <repo>/output
    #
    #   ONLINE runs (--online, recording from live cameras)
    #       -> output/<no_trial>/
    #       Set `no_trial` below to name the run, e.g. no_trial = "pilot_03"
    #       gives output/pilot_03/. Change it between recordings so runs do not
    #       overwrite each other.
    #
    #   OFFLINE runs (processing recorded video)
    #       -> output/<participant>/<task>/      e.g. output/1012/Lifting/
    #       Named automatically from the trial being processed, so results
    #       mirror the dataset layout. Override per run with --out DIR.
    #
    # Each run directory receives joint_angles.csv and markers.csv (and the
    # recorded videos when SAVE_VID is on).

    # Name of the current ONLINE recording. Ignored offline.
    no_trial = "test"

    # Save recorded videos / CSV files during ONLINE runs.
    # (Offline runs always write their CSV files unless --no-save is passed.)
    SAVE_VID: bool = True
    SAVE_CSV: bool = True
    # Recording controls (online runs). The hotkeys toggle live:
    # 's' starts, 'q' stops. record_on_start begins immediately, which
    # is what a headless or scripted run wants.
    record_hotkeys: bool = False
    record_on_start: bool = True

    # Computed automatically in __post_init__ - no need to edit these.
    output_dir: str = field(init=False)  # <repo>/output
    SAVE_DIR: str = field(init=False)    # <repo>/output/<no_trial>, online runs only

    ### CAM PARAMS ###
    fs: int = 40
    dt: float = field(init=False)  # Mark `dt` as excluded from the constructor
    width: int = 1280 # image resolution
    height: int = 720 # image resolution
    fourcc: str = "MJPG" # video codec
    # Which cameras to use, in order. The FIRST one is the reference frame that
    # triangulated 3D points are expressed in before the world transform.
    # Each id must exist in the calibration folder, and a matching detector
    # engine must have been built for this many cameras (see fetch_models.sh).
    # Override per run with: --cameras 0 2
    cameras: tuple = (0, 2, 4, 6)

    ### HUMAN ANTHROPOMETRY ###
    # Used for ONLINE runs. Offline runs read these per participant from the
    # dataset's metadata file instead (--subject), and only fall back to these
    # values when no subject file is given.
    human_height: float = 1.81
    human_weight: float = 74.0 
    human_gender: str = 'm'

    ### CALIB ###
    cam_calib_path: str = field(init=False) # relative path to the camera calibration file
    human_calib_path: str = field(init=False)  # relative path to the human calibration file
    robot_calib_path: str = field(init=False)  # relative path to the robot calibration file

    ### FILTER PARAMS ###
    order: int = 4
    system_freq: int = 40 
    cutoff_freq: float = 5
    filter_type: str = "lowpass"

    ### NLF ###
    cano_path: str = field(init=False)
    nlf_path: str = field(init=False)
    nlf_indices = [             # For SMPLX model
        8421, 5727, 8371, 5677, # pelvis: RASI, LASI, RPSI, LPSI 
        5484, 5489, 5500, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457, 4721, # upper: C7, T11, T6,  RSHO, LSHO, RELB, LELB, RMELB, LMELB, RWRI, LWRI, RMWRI, LMWRI
        8079, 5361, 7794, 5058, 8022, 5286,  # hands:  RTHU, LTHU, RMID, LMID, RPIN, LPIN
        6401, 3640, 6407, 3646, 8576, 5882, 8680, 8892, # legs: RKNE, LKNE, RMKNE, LMKNE, RANK, LANK, RMANK, LMANK,
        8474,5780,8463,5770,8635,8846, # feet: R5MHD, L5MHD, RTOE, LTOE, RHEE, LHEE,
        9120,9002,616,6,9929,9448,  # face: Nose, Head, REar, LEar, REye, LEye
    ]

    ### YOLO detector ###
    # Which detector to run. fetch_models.sh downloads all of these and exports
    # a TensorRT engine for whichever is named here.
    #
    # Benchmark the ENGINE, not the checkpoint: the ranking inverts between them.
    # Measured on real 4-camera frames from the dataset (batch 4, imgsz 640):
    #
    #     as .pt          yolo11n  12.5 ms    yolov10n  14.9 ms   (11n faster)
    #     as .engine      yolo11n  15.7 ms    yolov10n  12.7 ms   (10n faster)
    #
    # The pipeline loads engines, so yolov10n is the right default here even
    # though yolo11n wins as a checkpoint.
    yolo_model: str = "yolov10n"
    yolo_path: str = field(init=False)
    yolo_conf = 0.2
    yolo_imgsz = 640

    #### IK AND DATA HANDLING ###
    # For whole body model :

    # Joint angle column names. These are the RT-COSMIK standard and match the
    # COMFI dataset exactly, so estimates and reference mocap are directly
    # comparable without renaming anything.
    joint_angles_names = [
                            'Freeflyer_X[m]', 'Freeflyer_Y[m]', 'Freeflyer_Z[m]', 'Freeflyer_quaternion_X', 'Freeflyer_quaternion_Y', 'Freeflyer_quaternion_Z', 'Freeflyer_quaternion_W',
                            'Left_Hip_Flexion_Extension[rad]', 'Left_Hip_Abduction_Adduction[rad]', 'Left_Hip_Internal_External_Rotation[rad]', 'Left_Knee_Flexion_Extension[rad]', 'Left_Ankle_Plantarflexion_Dorsiflexion[rad]', 'Left_Ankle_Inversion_Eversion[rad]',
                            'Lumbar_Flexion_Extension[rad]', 'Lumbar_Lateral_Bending[rad]',
                            'Thoracic_Flexion_Extension[rad]', 'Thoracic_Lateral_Bending[rad]', 'Thoracic_Internal_External_Rotation[rad]',
                            'Left_Clavicle_Elevation_Depression[rad]',
                            'Left_Shoulder_Flexion_Extension[rad]', 'Left_Shoulder_Abduction_Adduction[rad]', 'Left_Shoulder_Internal_External_Rotation[rad]', 'Left_Elbow_Flexion_Extension[rad]', 'Left_Elbow_Pronation_Supination[rad]', 'Left_Wrist_Flexion_Extension[rad]', 'Left_Wrist_Radial_Ulnar_Deviation[rad]',
                            'Cervical_Flexion_Extension[rad]', 'Cervical_Lateral_Bending[rad]', 'Cervical_Internal_External_Rotation[rad]',
                            'Right_Clavicle_Elevation_Depression[rad]',
                            'Right_Shoulder_Flexion_Extension[rad]', 'Right_Shoulder_Abduction_Adduction[rad]', 'Right_Shoulder_Internal_External_Rotation[rad]', 'Right_Elbow_Flexion_Extension[rad]', 'Right_Elbow_Pronation_Supination[rad]', 'Right_Wrist_Flexion_Extension[rad]', 'Right_Wrist_Radial_Ulnar_Deviation[rad]',
                            'Right_Hip_Flexion_Extension[rad]', 'Right_Hip_Abduction_Adduction[rad]', 'Right_Hip_Internal_External_Rotation[rad]', 'Right_Knee_Flexion_Extension[rad]', 'Right_Ankle_Plantarflexion_Dorsiflexion[rad]', 'Right_Ankle_Inversion_Eversion[rad]']

    # Ik type
    ik_type: str ="mhe" # either "mhe" for SWIKA or "sbs" for sample by sample qp
    # fatrop only: "python" runs the CasADi function directly, "c" loads the
    # pre-compiled OCP from <repo>/ocp/fatrop (see run_ocp_codegen.py).
    ik_code: str = "c"

    # if ik_type = "mhe"
    mhe_backend: str = "acados" # solver backend: "fatrop" (validated reference) or "acados"
    # Speed/accuracy trade. "realtime" bounds the per-frame cost (acados uses
    # one real-time iteration per frame); "accurate" converges fully. Each is a
    # separate generated artefact -- run run_ocp_codegen.py after changing this.
    # Measured numbers are in rtcosmik.ik.ocp_model.SOLVER_PROFILES.
    mhe_profile: str = "realtime" # or "accurate"
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
        self.output_dir = str(Path(self.cosmik_path) / "output")
        self.SAVE_DIR = str(Path(self.output_dir) / self.no_trial)
        self.cano_path = str(Path(self.cosmik_path) / "weights/canonical_verts/smplx.npy")
        self.nlf_path = str(Path(self.cosmik_path) / "weights/nlf/nlf_s_multi_0.2.2.torchscript")
        self.yolo_path = str(Path(self.cosmik_path) / "weights" / "yolo"
                             / f"{self.yolo_model}.engine")
        self.dt = 1 / self.fs
