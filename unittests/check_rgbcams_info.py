import cv2

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            # Display the camera feed for a few seconds
            print(f"Camera index {index} opened. Press 'q' to quit preview.")
            while True: 
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(f'Camera {index}', frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            cap.release()  # Release the camera after checking
            cv2.destroyWindow(f'Camera {index}')  # Close the preview window
    return available_cameras

if __name__ == "__main__":
    cameras = list_available_cameras()
    if cameras:
        print("Available cameras:", cameras)
    else:
        print("No cameras found.")
