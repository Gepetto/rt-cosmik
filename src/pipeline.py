from src.triangulation.triangulation import triangulate_points
from src.augmenter.marker_augmenter import augmentTRC, loadModel
from src.filtering.iir import IIR
from src.ik.ik import RT_IK, RT_SWIKA
from src.utils.calib_utils import load_camera_parameters,load_world_transformation

from settings import Settings
from collections import dequeu
from multiprocessing import Process, Array, Lock, Value, Event, Queue
from typing import List

class PipelineProcess(Process):
    def __init__(self, 
                 settings: Settings,
                 camera_buffers: List[Array],
                 camera_timestamp_buffers: List[Array], # Character array for timestamp
                 camera_locks: List[Lock],
                 camera_frame_counters: List[Value],
                 results_queues: List[Queue],
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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # MP
        self.camera_buffers = camera_buffers
        self.camera_timestamp_buffers = camera_timestamp_buffers
        self.camera_locks = camera_locks
        self.camera_frame_counters = camera_frame_counters
        self.stop_event = stop_event
        self.results_queues = results_queues

        self.frame_shape = frame_shape
        self.num_cameras = num_cameras

        self.buffer_max_len = 30
        self.keypoints_buffer = dequeu(maxlen=self.buffer_max_len)
        self.warmed_augmenter_model = loadModel(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3')

        # Pinocchio related 
        self.human_model = build_dummy_model()

        #load camera param and config
        self.mtxs, self.dists, self.projections, self.rotations, self.translations = load_camera_parameters(self.CAM_CONFIG_PATH)
        self.world_R1_cam, self.world_T1_cam = load_world_transformation(self.CAM_CONFIG_PATH)

        ### Set up real time filter 
        # Constant
        num_channel = 3*len(self.keypoints_names)

        # Creating IIR instance
        self.iir_filter = IIR(
            num_channel=num_channel,
            sampling_frequency=self.fs
        )

        self.iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq, filter_type=settings.filter_type)

        self.first_sample = True

        def run(self):
            self.tracker = BatchPoseTrackerEstimator(self.num_cameras,self.DET_MODEL_PATH, self.POSE_MODEL_PATH, device=self.device)
            # Warmup
            _ = self.tracker.estimate([np.zeros(self.frame_shape, dtype=np.uint8) for _ in range(self.num_cameras)])

            try:
            while not self.stop_event.is_set():
                    frames = []
                    timestamps = []
                    keypoints_list = []
                    for lock, buffer, cam_ts in zip(self.camera_locks, self.camera_buffers, self.camera_timestamps):
                        with lock:
                            # Read and copy shared data atomically
                            arr = np.frombuffer(buffer, dtype=np.uint8)
                            frame = arr.reshape(self.frame_shape).copy()
                            # Get current timestamp
                            timestamp = bytes(cam_ts[:]).decode().strip('\x00')
                            frames.append(frame)
                            timestamps.append(timestamp)
                    
                    results = self.tracker.estimate(frames)

                    if results is None:
                        pass
                    else :
                        for res in results: 
                            keypoints_list.append(res[..., :2].astype(float))

                        keypoints_in_cam = triangulate_points(keypoints_list, self.mtxs, self.dists, self.projections)
                        keypoints_in_world = np.array([np.dot(self.world_R1_cam,point) + self.world_T1_cam for point in keypoints_in_cam])

                        if self.first_sample:
                            for k in range(self.buffer_max_len):
                                self.keypoints_buffer.append(keypoints_in_world)  #add the 1st frame 30 times
                        else:
                            self.keypoints_buffer.append(keypoints_in_world) #add the keypoints to the buffer normally 
                        
                        if len(self.keypoints_buffer) == self.buffer_max_len:
                            keypoints_buffer_array = np.array(self.keypoints_buffer)

                            # Filter keypoints in world to remove noisy artefacts 
                            filtered_keypoints_buffer = self.iir_filter.filter(np.reshape(keypoints_buffer_array,(self.buffer_max_len, 3*len(self.keypoints_names))))
                            filtered_keypoints_buffer = np.reshape(filtered_keypoints_buffer,(self.buffer_max_len, len(self.keypoints_names), 3))

                            augmented_markers = augmentTRC(filtered_keypoints_buffer, subject_mass=self.subject_mass, subject_height=self.subject_height, models = self.warmed_augmenter_model,
                                        augmenterDir=self.AUGMENTER_PATH, augmenter_model='v0.3')
                            
                            if len(augmented_markers) % 3 != 0:
                                raise ValueError("The length of the list must be divisible by 3.")

                            augmented_markers = np.array(augmented_markers).reshape(-1, 3)

                            if self.first_sample:
                                kp_dict = dict(zip(self.keypoints_names,filtered_keypoints_buffer[-1]))
                                mks_dict = dict(zip(self.marker_names, augmented_markers))
                                
                                self.human_model = rescale_human_model(self.human_model, kp_dict, mks_dict)
                                
                                if self.ik_type == 'qp':
                                    q = pin.neutral(self.human_model)
                                    ik_class = RT_IK(self.human_model, mks_dict, q, self.keys_to_track_list, self.dt)

                                    q = ik_class.solve_ik_sample_casadi()
                                    ik_class._q0 = q

                                elif self.ik_type == 'mhe':

                                else : 
                                    raise ValueError("Invalid ik type, should be qp or mhe")

                                self.first_sample = False
                            
                            else:
                                if self.ik_type == 'qp':
                                    kp_dict = dict(zip(keypoints_names,filtered_keypoints_buffer[-1]))
                                    mks_dict = dict(zip(marker_names, augmented_markers))
                                    
                                    ### IK calculations
                                    ik_class._dict_m= mks_dict
                                    q = ik_class.solve_ik_sample_quadprog() 
                                    ik_class._q0 = q
                                    
                                elif self.ik_type == 'mhe':

                                else : 
                                    raise ValueError("Invalid ik type, should be qp or mhe")