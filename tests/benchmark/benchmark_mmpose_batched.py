import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Event, Barrier, set_start_method
from rtcosmik.pose_estimator.pose_estimator import BatchPoseTrackerEstimator
from settings import Settings
import os 

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
settings = Settings()

def worker_fn(video_paths, det_model, pose_model, barrier, result_queue, stop_event):
    """Worker process handling video processing"""
    print(f"Beginning for worker")
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    tracker = BatchPoseTrackerEstimator(len(video_paths),det_model, pose_model)
    
    # Warmup
    _ = tracker.estimate([np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(len(video_paths))])
    
    while not stop_event.is_set():
        barrier.wait()  # Sync to start processing
        frames = [cap.read()[1] for cap in caps]

        start_time = time.perf_counter()
        results = tracker.estimate(frames)
        # if not tracker.visualize(frames,results):
        #     break
        end_time = time.perf_counter()
        
        barrier.wait()  # Sync after processing
    
    for cap in caps:
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
                args=(video_paths, det_model, pose_model, barrier, result_queue, stop_event)
            )
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
    barrier = Barrier(2)  # +1 for main process
    result_queue = Queue()
    stop_event = Event()

    benchmarker = VideoBenchmarker(VIDEO_PATHS, DET_MODEL, POSE_MODEL, barrier, result_queue, stop_event)
    benchmarker.run_benchmark()

if __name__ == '__main__':
    set_start_method('spawn')
    main()
    