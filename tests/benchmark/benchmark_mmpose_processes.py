import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Event, Barrier, set_start_method
from rtcosmik.pose_estimator.pose_estimator import PoseTrackerEstimator
from settings import Settings
import os 

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
settings = Settings()

def worker_fn(worker_id, video_path, det_model, pose_model, barrier, result_queue, stop_event):
    """Worker process handling video processing"""
    print(f"Beginning for worker {worker_id}")
    cap = cv2.VideoCapture(video_path)
    tracker = PoseTrackerEstimator(det_model, pose_model)
    
    # Warmup
    _ = tracker.estimate(np.zeros((720, 1280, 3), dtype=np.uint8))
    
    while not stop_event.is_set():
        barrier.wait()  # Sync to start processing
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.perf_counter()
        results, _ = tracker.estimate(frame)
        # if not tracker.visualize(frame,results,worker_id):
        #     break
        end_time = time.perf_counter()
        
        result_queue.put((worker_id, start_time, end_time, results))
        
        barrier.wait()  # Sync after processing
    
    cap.release()

class VideoBenchmarker:
    def __init__(self, video_paths, det_model, pose_model, barrier, result_queue, stop_event):
        self.video_paths = video_paths
        self.barrier = barrier
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.workers = [
            Process(
                target=worker_fn,
                args=(i, video_path, det_model, pose_model, barrier, result_queue, stop_event)
            ) for i, video_path in enumerate(video_paths)
        ]

    def run_benchmark(self):
        """Main benchmark execution loop"""
        print("Main starts")
        for w in self.workers:
            w.start()

        try:
            while True:
                self.barrier.wait()
                time_init = time.time()
                
                # Sync for next frame
                self.barrier.wait()
                time_final = time.time()
                print("Processing time in main : ", time_final-time_init)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.stop_event.set()
            for w in self.workers:
                w.join()

def main():
    # Configuration
    VIDEO_PATHS = [
        os.path.join(script_directory, 'videos/camera_0.mp4'),
        os.path.join(script_directory, 'videos/camera_2.mp4'),
        os.path.join(script_directory, 'videos/camera_4.mp4'),
        # os.path.join(script_directory, 'videos/camera_6.mp4')
    ]
    DET_MODEL = settings.det_model_path
    POSE_MODEL = settings.pose_model_path

    # Using 'fork' start method by default on Linux
    barrier = Barrier(len(VIDEO_PATHS) + 1)  # +1 for main process
    result_queue = Queue()
    stop_event = Event()

    benchmarker = VideoBenchmarker(VIDEO_PATHS, DET_MODEL, POSE_MODEL, barrier, result_queue, stop_event)
    benchmarker.run_benchmark()

if __name__ == '__main__':
    set_start_method('spawn')
    main()
    