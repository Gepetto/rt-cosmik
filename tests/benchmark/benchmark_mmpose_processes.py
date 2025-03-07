import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Event, Barrier
from src.pose_estimator.pose_estimator import PoseTrackerEstimator
from settings import Settings
settings=Settings()
import os 
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class VideoBenchmarker:
    def __init__(self, video_paths, det_model, pose_model):
        self.video_paths = video_paths
        self.det_model = det_model
        self.pose_model = pose_model
        self.num_videos = len(video_paths)
        self.benchmark_data = []
        
        # Initialize multiprocessing components
        self.barrier = Barrier(self.num_videos + 1)  # +1 for main process
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.stop_event = Event()
        
        # Create worker processes
        self.workers = [
            Process(
                target=self._worker_fn,
                args=(i, video_path)
            ) for i, video_path in enumerate(video_paths)
        ]

    def _worker_fn(self, worker_id, video_path):
        """Worker process handling video processing"""
        cap = cv2.VideoCapture(video_path)
        tracker = PoseTrackerEstimator(self.det_model, self.pose_model)
        
        # Warmup
        _ = tracker.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
        
        while not self.stop_event.is_set():
            # Wait for synchronization
            self.barrier.wait()
            
            # Get current frame
            ret, frame = cap.read()
            if not ret:
                self.frame_queue.put((worker_id, None))
                break
            
            # Process frame and time execution
            start_time = time.perf_counter()
            results, _ = tracker.estimate(frame)
            end_time = time.perf_counter()
            
            # Send results back
            self.result_queue.put((
                worker_id,
                start_time,
                end_time,
                results
            ))
            
            # Wait for visualization sync
            self.barrier.wait()
        
        cap.release()

    def _display_frame(self, frame, worker_id):
        """Helper function to display frames"""
        cv2.imshow(f'Video {worker_id}', frame)
        cv2.displayStatusBar(f'Video {worker_id}', 'Press "s" to benchmark next frame')

    def run_benchmark(self):
        """Main benchmark execution loop"""
        # Start worker processes
        for w in self.workers:
            w.start()

        try:
            while True:
                # Get current frames from all workers
                frames = []
                for _ in range(self.num_videos):
                    worker_id, frame = self.frame_queue.get()
                    if frame is None:
                        return  # End of video
                    frames.append((worker_id, frame))
                
                # Display frames
                for worker_id, frame in frames:
                    self._display_frame(frame, worker_id)
                
                # Wait for user input
                key = cv2.waitKey(0)
                
                if key == ord('s'):
                    # Signal workers to start processing
                    self.barrier.wait()
                    
                    # Collect results
                    start_times = []
                    end_times = []
                    for _ in range(self.num_videos):
                        worker_id, st, et, res = self.result_queue.get()
                        start_times.append(st)
                        end_times.append(et)
                    
                    # Calculate timing
                    global_start = min(start_times)
                    global_end = max(end_times)
                    frame_time = global_end - global_start
                    self.benchmark_data.append(frame_time)
                    
                    print(f"Frame processed in {frame_time:.4f}s")
                    
                    # Sync for next frame
                    self.barrier.wait()
                
                elif key == ord('q'):
                    break
        
        finally:
            self.stop_event.set()
            for w in self.workers:
                w.join()
            cv2.destroyAllWindows()

    def print_results(self):
        """Print final benchmark statistics"""
        if not self.benchmark_data:
            print("No benchmark data collected!")
            return

        print("\nBenchmark Results:")
        print(f"Total Frames Processed: {len(self.benchmark_data)}")
        print(f"Average Time per Frame: {np.mean(self.benchmark_data):.4f}s")
        print(f"Maximum Time: {np.max(self.benchmark_data):.4f}s")
        print(f"Minimum Time: {np.min(self.benchmark_data):.4f}s")
        print(f"Standard Deviation: {np.std(self.benchmark_data):.4f}s")

if __name__ == '__main__':
    # Configuration
    VIDEO_PATHS = [os.path.join(script_directory,'videos/camera_0.mp4'), os.path.join(script_directory,'videos/camera_2.mp4'), os.path.join(script_directory,'videos/camera_4.mp4'), os.path.join(script_directory,'videos/camera_6.mp4')]
    DET_MODEL = settings.det_model_path
    POSE_MODEL = settings.pose_model_path

    # Run benchmark
    benchmarker = VideoBenchmarker(VIDEO_PATHS, DET_MODEL, POSE_MODEL)
    benchmarker.run_benchmark()
    benchmarker.print_results()