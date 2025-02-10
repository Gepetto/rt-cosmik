import cv2
import os
from datetime import datetime

class Camera:
    def __init__(self, camera_id=0, width=640, height=480, fps=30, fourcc='MJPG'):
        """
        Initialize the camera module with default settings.
        :param camera_id: Index or device name of the camera.
        :param width: Desired frame width.
        :param height: Desired frame height.
        :param fps: Frames per second.
        :param fourcc: FourCC code for the video codec (e.g., 'MJPG').
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.cap = None         # This will hold the VideoCapture object.
        self.writer = None      # This will hold the VideoWriter (if saving video).
    
    def open_camera(self):
        """Opens the camera and applies the settings."""
        # Create a VideoCapture object using the V4L2 backend for Linux (adjust if needed)
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise Exception(f"Camera {self.camera_id} could not be opened.")
        
        self.set_camera_settings()
        print(f"Camera {self.camera_id} opened with resolution {self.width}x{self.height} at {self.fps} FPS.")
    
    def set_camera_settings(self):
        """Sets the desired camera settings."""
        if self.cap is None:
            raise Exception("Camera is not opened. Call open_camera() first.")
        
        # Set frame width, height, fps, and FOURCC codec.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc)) # The FOURCC code tells the camera which format to use (MJPG often gives better performance).
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
    
    def read_frame(self):
        """
        Reads a single frame from the camera.
        :return: The captured frame.
        """
        if self.cap is None:
            raise Exception("Camera is not opened. Call open_camera() first.")
        
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture frame.")
        return frame
    
    def start_capture(self, display=True, save_video=False, output_dir='output'):
        """
        Starts capturing video frames.
        :param display: Whether to display the video in a window.
        :param save_video: Whether to save the video to a file.
        :param output_dir: Directory where the video file will be saved.
        """
        # Open the camera if not already opened.
        if self.cap is None:
            self.open_camera()
        
        # Setup video writer if saving video.
        if save_video:
            os.makedirs(output_dir, exist_ok=True)
            video_filename = os.path.join(
                output_dir,
                f"camera_{self.camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            )
            # Create a VideoWriter using the same codec and settings.
            fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
            self.writer = cv2.VideoWriter(video_filename, fourcc, self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                raise Exception("Video writer could not be opened.")
            print(f"Saving video to {video_filename}")
        
        print("Starting capture loop. Press 'q' to quit.")
        while True:
            try:
                frame = self.read_frame()
            except Exception as e:
                print(f"Error reading frame: {e}")
                break
            
            if display:
                cv2.imshow(f"Camera {self.camera_id}", frame)
            
            if save_video and self.writer is not None:
                self.writer.write(frame)
            
            # Check for quit key.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.release()
    
    def release(self):
        """Releases the camera and writer resources and closes any windows."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        cv2.destroyAllWindows()
        print(f"Camera {self.camera_id} and all resources released.")

# Example usage:
# if __name__ == "__main__":
#     # Create a camera module instance with desired settings.
#     cam = Camera(camera_id=0, width=1280, height=720, fps=30, fourcc='MJPG')
#     try:
#         # Optionally, open the camera explicitly.
#         cam.open_camera()
#         # Start capturing with both display and video saving enabled.
#         cam.start_capture(display=True, save_video=False)
#     except Exception as e:
#         print("An error occurred:", e)
#         cam.release()
