import cv2
import os
import subprocess
import numpy as np
import time
import multiprocessing as mp
from datetime import datetime
from camera.camera import SingleCamera

class MultiCameraSystem:
    """Class to handle multiple cameras using multiprocessing."""
    def __init__(self, width=1280, height=720, fps=40):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_ids = self.list_cameras()
        self.shape = (height, width, 3)
        self.buffers = {cam_id: mp.Array("B", width * height * 3) for cam_id in self.camera_ids}
        self.locks = {cam_id: mp.Lock() for cam_id in self.camera_ids}
        self.barrier = mp.Barrier(len(self.camera_ids))

    @staticmethod
    def list_cameras():
        """List all available cameras using v4l2-ctl."""
        cameras = {}
        try:
            output = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
            devices = output.strip().split("\n\n")
            for device in devices:
                lines = device.split("\n")
                if len(lines) > 1:
                    video_path = lines[1].strip()
                    if "/dev/video" in video_path:
                        index = int(video_path.split("video")[-1])
                        cameras[index] = lines[0].strip()
        except subprocess.SubprocessError as e:
            print(f"Error using v4l2-ctl: {e}")
        return list(cameras.keys())

    def capture_frames_buffer(self, cam_id, buffer, lock, barrier):
        """Capture frames from a camera and store them in a shared memory buffer."""
        barrier.wait() #wait for all process before launching cameras
        camera = SingleCamera(camera_id=cam_id, width=self.width, height=self.height, fps=self.fps)
        camera.open()

        while True:
            try:
                frame = camera.read_frame()
                with lock:
                    np_buffer = np.frombuffer(buffer.get_obj(), dtype=np.uint8).reshape(self.shape)
                    np_buffer[:] = frame  # Copy frame to shared buffer
            except Exception as e:
                print(f"Camera {cam_id} error: {e}")
                break

        camera.release()

    def start_processes(self):
        """Start multiprocessing for all cameras."""
        if len(self.camera_ids) < 2:
            print("Error: At least two cameras are required!")
            return
        
        self.processes = [
            mp.Process(target=self.capture_frames_buffer, args=(cam_id, self.buffers[cam_id], self.locks[cam_id], self.barrier))
            for cam_id in self.camera_ids
        ]
        
        for p in self.processes:
            p.start()

    def get_frames(self):
        """Retrieve frames from all cameras(buffers)."""
        frames = []
        for cam_id in self.camera_ids:
            with self.locks[cam_id]:
                frame = np.frombuffer(self.buffers[cam_id].get_obj(), dtype=np.uint8).reshape(self.shape).copy()
            frames.append(frame)
        return frames

    def stop_processes(self):
        """Terminate all processes and clean up."""
        for p in self.processes:
            p.terminate()
            p.join()
        cv2.destroyAllWindows()