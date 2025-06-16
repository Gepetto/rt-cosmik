import cv2
import numpy as np
import csv
from rtmlib import BodyWithFeet, PoseTracker, draw_skeleton
import sys

# Configuration
num_cam = sys.argv[1]
device = 'cpu'  # 'cpu', 'cuda', 'mps'
backend = 'onnxruntime'  # 'opencv', 'onnxruntime', 'openvino'
video_path = f"/root/workspace/ros_ws/src/rt-cosmik/output/Test_end2end/static/camera_{num_cam}.mp4"  # Remplace par le chemin vers ta vidéo
csv_output = f"/root/workspace/ros_ws/src/rt-cosmik/output/Test_end2end/static/keypoints_cam{num_cam}.csv"
openpose_skeleton = False  # True pour style OpenPose, False pour style MMPose

# Initialisation du modèle
wholebody = PoseTracker(BodyWithFeet,
                        det_frequency=7,
                        to_openpose=openpose_skeleton,
                        mode='performance',
                        backend=backend,
                        device=device)

# Ouverture de la vidéo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Ouverture du fichier CSV en mode écriture (sans header)
with open(csv_output, mode='w', newline='') as f:
    writer = csv.writer(f)

    # Boucle de traitement frame par frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo

        keypoints, scores = wholebody(frame)

        # Affichage du squelette
        frame_with_skeleton = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.5)
        # cv2.imshow('Skeleton Video', frame_with_skeleton)

        # Sauvegarde dans le CSV
        if keypoints is not None and scores is not None:
            keypoints_flat = keypoints.flatten().tolist()  # x1, y1, x2, y2, ...
            scores_flat = scores.flatten().tolist()        # s1, s2, ...
            scores_mean = [np.mean(scores_flat)]
            writer.writerow(scores_mean + keypoints_flat)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Nettoyage
cap.release()
cv2.destroyAllWindows()

