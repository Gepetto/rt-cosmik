#python3 -m unittests.test_multiprocess_posetracker cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m

import os
import time
import cv2
import multiprocessing as mp
from mmdeploy_runtime import PoseTracker
from utils.viz_utils import visualize, VISUALIZATION_CFG
from utils.settings import Settings
from utils.process_utils import parse_args, capture_frames_buffer, initialize_cameras
import numpy as np

def main():
    args = parse_args()
    settings = Settings()

    # Initialize cameras
    camera_ids, shape, buffers, locks, capture_times, barrier = initialize_cameras(settings)
    if camera_ids is None:
        return

    # Start camera capture processes
    processes = [
        mp.Process(
            target=capture_frames_buffer, args=(cam_id, buffers[cam_id], locks[cam_id], shape, settings, barrier, capture_times)
        )
        for cam_id in camera_ids
    ]

    for p in processes:
        p.start()

    # Initialize Pose Tracker
    tracker = PoseTracker(det_model=args.det_model, pose_model=args.pose_model, device_name=args.device_name)
    sigmas = VISUALIZATION_CFG[args.skeleton]["sigmas"]
    state1 = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas)
    states = [state1] * len(camera_ids)

    frame_idx = 0
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    try:
        while True:
            frames = []

            fetch_start_time = time.time()
            for cam_id in camera_ids:
                with locks[cam_id]:  # Prevent race conditions
                    frame = np.frombuffer(buffers[cam_id].get_obj(), dtype=np.uint8).reshape(shape).copy()
                frames.append(frame)

            fetch_time = time.time() - fetch_start_time  # Measure fetching time

            if len(frames) < len(camera_ids):
                continue  # Skip if not all frames are captured


            t0 = time.time()
            results = tracker.batch(states, frames, detects=[-1] * len(camera_ids))  # Pose inference
            k,b = results
            print(k[0])
            inference_time = time.time() - t0

            total_time =fetch_time + inference_time
            print(f"Total Processing Time: {total_time}")

            for i, frame in enumerate(frames):
                if not visualize(frame, results[i], args.output_dir, i, frame_idx + i, skeleton_type=args.skeleton):
                    break

            frame_idx += len(camera_ids)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit")
                break

    finally:
        for p in processes:
            p.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Ensure compatibility on some platforms
    main()
