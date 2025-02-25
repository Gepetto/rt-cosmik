# What we call piepeline is the process that takes as input the camera streams and process up to the inverse kinematics output

import os
import sys
# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# from pose_estimator.pose_estimator import PoseEstimator
from triangulation.triangulation import triangulate_points
from augmenter.marker_augmenter import augmentTRC, loadModel
from filtering.iir import IIR
from ik.ik import RT_IK

from human_model.pin_model import * 
from human_model.model_utils import construct_segments_frames, get_segments_mks_dict
from viewer.gv_viewer import place, gv_init, Rquat, add_marker, add_frames
from collections import deque
from utils.calib_utils import load_camera_parameters,load_world_transformation
from utils.settings import Settings

settings = Settings()
# rtmpose model paths
DET_MODEL_PATH = "/root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano"
POSE_MODEL_PATH = "/root/workspace/mmdeploy/rtmpose-trt/rtmpose-m"

base_path = "/root/workspace/ros_ws/src/rt-cosmik"
meshes_folder_path = os.path.join(base_path, "meshes/old_meshes")
augmenter_path = os.path.join(base_path, "augmentation_model")
config_path = os.path.join(base_path, "config/cam_params")


width = settings.width
heighyt = settings.height 
fps = settings.fs 
subject_mass = settings.human_mass
subject_height = settings.human_height
keypoints_names = settings.keypoints_names
marker_names = settings.marker_names
dt = settings.dt
keys_to_track_list = settings.keys_to_track_list

keypoints_buffer = deque(maxlen=30)
warmed_models= loadModel(augmenterDir=augmenter_path, augmenterModelName="LSTM",augmenter_model='v0.3')

def main():

    #load camera param and config
    mtxs, dists, projections, rotations, translations = load_camera_parameters(config_path)
    world_R1_cam, world_T1_cam = load_world_transformation(config_path)
    
    # Initialize Pose Tracker
    tracker = PoseTrackerEstimator(DET_MODEL_PATH, POSE_MODEL_PATH)

    # Initialize Multi-Camera System
    multi_cam = MultiCameraSystem(width=width, height=height, fps=fps)
    multi_cam.start_processes()

    ### Set up real time filter 
    # Constant
    num_channel = 3*len(keypoints_names)

    # Creating IIR instance
    iir_filter = IIR(
        num_channel=num_channel,
        sampling_frequency=settings.system_freq
    )

    iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq, filter_type=settings.filter_type)
    

    first_sample = True
    try:
        while True:
            # Get frames from all cameras
            frames = multi_cam.get_frames()

            keypoints_list = []
            
            if frames is None or len(frames) < 2:
                continue  # Skip if no valid frames

            # Concatenate frames horizontally
            stacked_frame = hconcat_frames(frames)

            # Run pose estimation
            results = tracker.estimate(stacked_frame)

            # Reproject results to original frames
            first_result, second_result = reproject(results, width, axis="horizontal")
            keypoints_list.append(first_result)
            keypoints_list.append(second_result)

            # Visualize results on each frame
            # for idx, (frame, result) in enumerate(zip(frames, [first_result, second_result])):
            #     if result is not None and not tracker.visualize(frame, result, idx=idx):
            #         return  # Exit if 'q' is pressed

            if len(keypoints_list)!=2: #HPE has been applied to both frames
                pass
            else: 
                p3d_frame = triangulate_points(keypoints_list, mtxs, dists, projections)
                keypoints_in_cam = p3d_frame

                # Apply the rotation matrix to align the points
                keypoints_in_world = np.array([np.dot(world_R1_cam,point) + world_T1_cam for point in keypoints_in_cam])
                
                if first_sample:
                    for k in range(30):
                        keypoints_buffer.append(keypoints_in_world)  #add the 1st frame 30 times
                
                else:
                    keypoints_buffer.append(keypoints_in_world) #add the keypoints to the buffer normally 


                if len(keypoints_buffer) == 30:
                    keypoints_buffer_array = np.array(keypoints_buffer)

                    # Filter keypoints in world to remove noisy artefacts 
                    filtered_keypoints_buffer = iir_filter.filter(np.reshape(keypoints_buffer_array,(30, 3*len(keypoints_names))))
                    filtered_keypoints_buffer = np.reshape(filtered_keypoints_buffer,(30, len(keypoints_names), 3))

                    augmented_markers = augmentTRC(filtered_keypoints_buffer, subject_mass=subject_mass, subject_height=subject_height, models = warmed_models,
                                augmenterDir=augmenter_path, augmenter_model='v0.3')

                    if len(augmented_markers) % 3 != 0:
                        raise ValueError("The length of the list must be divisible by 3.")

                    augmented_markers = np.array(augmented_markers).reshape(-1, 3)

                    if first_sample:
                        mks_dict = dict(zip(marker_names, augmented_markers))
                        ### Generate human model
                        human_model, human_geom_model, visuals_dict = build_model(mks_dict, meshes_folder_path)

                        ### IK init 
                        q = pin.neutral(human_model) # init pos

                        ### IK calculations
                        ik_class = RT_IK(human_model, mks_dict, q, keys_to_track_list, dt)
                        q = ik_class.solve_ik_sample_casadi()
                        ik_class._q0=q

                        first_sample = False  #put the flag to false 
                    
                    else:
                        mks_dict = dict(zip(marker_names, augmented_markers))
                        ### IK calculations
                        ik_class._dict_m= mks_dict
                        q = ik_class.solve_ik_sample_quadprog() 
                        # q = ik_class.solve_ik_sample_casadi()

                        ik_class._q0 = q

    except KeyboardInterrupt:
        print("Exiting gracefully...")
    finally:
        multi_cam.stop_processes()

if __name__ == "__main__":
    main()