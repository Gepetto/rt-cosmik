from src.triangulation.triangulation import triangulate_points
from src.augmenter.marker_augmenter import augmentTRC, loadModel
from src.filtering.iir import IIR
from src.ik.ik import RT_IK, RT_SWIKA
from src.utils.calib_utils import load_camera_parameters,load_world_transformation

from settings import Settings
from collections import dequeu
from multiprocessing import Process, Array, Lock, Value, Event
from typing import List

class PipelineProcess(Process):
    def __init__(self, 
                 settings: Settings,
                 camera_buffers: List[Array],
                 camera_timestamp_buffers: List[Array], # Character array for timestamp
                 camera_locks: List[Lock],
                 camera_frame_counters: List[Value],
                 stop_event: Event,
                 frame_shape: tuple = (720, 1280, 3),
                 num_cameras: int
                 ):
        super().__init__()
        self.DET_MODEL_PATH = settings.det_model_path
        self.POSE_MODEL_PATH = settings.pose_model_path
        self.AUGMENTER_PATH = settings.augmenter_path
        self.CAM_CONFIG_PATH = settings.cam_calib_path
        self.fs = settings.fs
        self.subject_mass=settings.human_mass
        self.subject_height=settings.human_height
        self.keypoints_names=settings.keypoints_names
        self.marker_names=settings.marker_names
        self.dt = settings.dt
        self.keys_to_track_list = settings.keys_to_track_list
        self.ik_type = settings.ik_type

        # MP
        self.camera_buffers = camera_buffers
        self.camera_timestamp_buffers = camera_timestamp_buffers
        self.camera_locks = camera_locks
        self.camera_frame_counters = camera_frame_counters
        self.stop_event = stop_event

        self.frame_shape = frame_shape
        self.num_cameras = num_cameras

        self.keypoints_buffer = dequeu(maxlen=30)
        self.warmed_augmenter_model = loadModel(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3')

        #load camera param and config
        self.mtxs, self.dists, self.projections, self.rotations, self.translations = load_camera_parameters(self.CAM_CONFIG_PATH)
        self.world_R1_cam, self.world_T1_cam = load_world_transformation(self.CAM_CONFIG_PATH)

        def run(self):
            self.tracker = 