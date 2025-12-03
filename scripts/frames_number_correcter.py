import cv2
import sys
import csv

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Total number of frames: {total_frames}")
    return total_frames

if __name__=="__main__":

    idx_cam1 = sys.argv[1]
    idx_cam2 = sys.argv[2]
    subject = sys.argv[3]
    trial = sys.argv[4]

    video1_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{subject}/{trial}/camera_{idx_cam1}.avi"
    video2_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{subject}/{trial}/camera_{idx_cam2}.avi"
    csv1_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{subject}/{trial}/keypoints_cam{idx_cam1}.csv"
    csv2_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/{subject}/{trial}/keypoints_cam{idx_cam2}.csv"

    if count_frames(video1_path) > count_frames(video2_path):
        # Lire tout sauf la dernière ligne
        with open(csv1_path, newline='') as f:
            reader = list(csv.reader(f))
            lignes_sans_derniere = reader[:-1]
        
        # Réécrire le fichier sans la dernière ligne
        with open(csv1_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(lignes_sans_derniere)
    elif count_frames(video2_path) > count_frames(video1_path):
        # Lire tout sauf la dernière ligne
        with open(csv2_path, newline='') as f:
            reader = list(csv.reader(f))
            lignes_sans_derniere = reader[:-1]
        
        # Réécrire le fichier sans la dernière ligne
        with open(csv2_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(lignes_sans_derniere)
