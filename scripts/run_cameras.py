from .settings import Settings
settings = Settings()

from src.camera import start_camera_process
from src.camera.camera_manager import CameraManager, CameraBufferReader
from src.process_manager import ProcessManager
import signal
import cv2

def main():
    # Initialize managers
    process_manager = ProcessManager()
    camera_manager = CameraManager(settings.width, 
                            settings.height, 
                            settings.fps, 
                            settings.fourcc)

    # Share resources with consumers
    shared_res = camera_manager.get_shared_resources()
    reader = CameraBufferReader(shared_res)

    # Add camera processes
    for cam in camera_manager._cameras:
        process_manager.add_process(
            target=start_camera_process,
            args=(
                cam['id'],
                settings.width,
                settings.height,
                settings.fps,
                settings.fourcc,
                cam['frame_buffer'],
                cam['timestamp_buffer'],
                cam['lock'],
                camera_manager._barrier
            )
        )

    # Signal handling
    def handle_interrupt(sig, frame):
        print("\nTermination requested")
        process_manager.stop_all()
        cv2.destroyAllWindows()
        
    signal.signal(signal.SIGINT, handle_interrupt)

    # Start all processes
    process_manager.start_all()

    # For better performance, add this before the display loop
    cv2.startWindowThread()
    
    try:
        while not process_manager.shared_events['stopping_event'].is_set():
            frames = reader.read_all()
            
            # Display frames using OpenCV
            for frame_data in frames:
                # Convert RGB to BGR for OpenCV display
                bgr_frame = cv2.cvtColor(frame_data['frame'], cv2.COLOR_RGB2BGR)
                window_name = f"Camera {frame_data['camera_id']}"
                cv2.imshow(window_name, bgr_frame)
                print("timestamp: ", frame_data['timestamp'])
                
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        process_manager.stop_all()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()