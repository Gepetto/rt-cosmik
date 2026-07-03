import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

def process_video(video_path, yolo_model_path, nlf_model_path, output_path="output_video.mp4"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Charger les modèles
    yolo = YOLO(yolo_model_path)
    nlf = torch.jit.load(nlf_model_path).eval().to(device)

    # 2. Préparer les poids une seule fois (pour tous les sommets ici)
    cano_verts_full = np.load("/root/workspace/RT-COSMIK/weights/canonical_verts/smplx.npy")

    indices = [5621, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457,
               4721, 8421, 5727, 8371, 5677, 6401, 3640, 6407, 3646, 8576,5882,8680,8892,8596,5902,8589,5895
               ,8482,5788,8846,8634]

    indices = [5621, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457,
               4721, 8421, 5727, 8371, 5677, 6401, 3640, 6407, 3646, 8576,5882,8680,8892,8596,5902,8589,5895
               ,8482,5788,8846,8634,7978,4807,8004,5268,7483,4747,7664,4928,7776,5040,7887,5151,7420,4684,8078,5342,5484,5941,5489,5500]

    indices = [5484, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457, 4721,#c7, rshoulder,lshoulder,r_lelbow,l_lelbow,r_melbow,l_melbow,r_lwrist, l_lwrist,r_mwrist, l_mwrist
               8421, 5727, 8371, 5677, #r_asis,l_asis,r_psis,l_psis 
               6401, 3640, 6407, 3646, 8576,5882,8680,8892, #r_knee,l_knee,r_mknee,l_mknee,r_ankle,l_ankle,r_mankle,l_mankle,
               8596,5902,8589,5895,8482,5788,8846,8634,#r_5meta, l_5meta, r_toe, l_toe, r_big_toe, l_big_toe, l_calc, r_calc,
               7978,4807,8004,5268,7483,4747,7664,4928,7776,5040,7887,5151,7420,4684,8078,5342, #7978,4807, r_tpinky, l_tpinky, r_bindex, l_bindex, r_tindex, l_tindex, r_tmiddle, l_tmiddle, r_tring,l_tring, r_bthumb, l_bthumb,r_tthumb, l_tthumb
               9008,9002,1253,399,10049,9503, ##nose, head,right_ear,left_ear, right_eye, left_eye
              5941,5489,5500] #L2, T11, T6

    selected_points = torch.from_numpy(cano_verts_full[indices]).float().to(device)

    all_points = torch.from_numpy(cano_verts_full).float().to(device)
    weights = nlf.get_weights_for_canonical_points(selected_points)

    # 3. Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Infos vidéo
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurer le VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Paramètres Caméra (fixe pour la vidéo)
    focal_length = width / (2 * np.tan(np.deg2rad(55) / 2))
    intrinsics = torch.tensor([[[focal_length, 0, width/2], 
                                [0, focal_length, height/2], 
                                [0, 0, 1]]]).float().to(device)
    extrinsics = torch.eye(4).unsqueeze(0).to(device)

    print(f"Début du traitement : {total_frames} frames à traiter...")

    frame_count = 0
    yolo_time_list=[]
    nlf_time_list=[]
    total_time_list=[]
    fps_list=[]

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()

            # --- Etape A: YOLO ---
            start_yolo = time.time()
            # verbose=False pour éviter de polluer la console
            yolo_results = yolo.predict(frame_bgr, classes=0, conf=0.25, device=device, verbose=False)[0]
            yolo_time = (time.time() - start_yolo) * 1000  # Conversion en ms

            nlf_boxes = []
            if len(yolo_results.boxes) > 0:
                boxes = yolo_results.boxes.xyxy
                scores = yolo_results.boxes.conf.unsqueeze(1)
                wh = boxes[:, 2:] - boxes[:, :2]
                boxes_formatted = torch.cat([boxes[:, :2], wh, scores], dim=1)
                nlf_boxes.append(boxes_formatted)
            else:
                nlf_boxes.append(torch.zeros((0, 5)).to(device))

            # --- Etape B: NLF ---
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().divide(255).unsqueeze(0).to(device)

            start_nlf = time.time()
            with torch.inference_mode():
                outputs = nlf.estimate_poses_batched(
                    img_tensor, nlf_boxes,
                    intrinsic_matrix=intrinsics,
                    extrinsic_matrix=extrinsics,
                    world_up_vector=torch.tensor([0.0, -1.0, 0.0]).to(device),
                    weights=weights,
                    num_aug=1
                )
            nlf_time = (time.time() - start_nlf) * 1000

            # --- Etape C: Dessin ---
            poses_3d = outputs['poses3d'][0]
            processed_frame = draw_projections(frame_bgr, poses_3d, intrinsics)

            # Calcul temps total frame
            total_time = (time.time() - frame_start_time) * 1000
            
            # Afficher les infos sur la frame (optionnel)
            cv2.putText(processed_frame, f"NLF: {nlf_time:.1f}ms | Total: {total_time:.1f}ms", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Ecrire la frame
            out.write(processed_frame)
            
            fps=1000/total_time

            frame_count += 1
            if frame_count>10:
                print(f"Frame {frame_count}/{total_frames}\t yolo=\t{yolo_time:.1f}\t nlf=\t{nlf_time:.1f}\t total=\t{total_time:.1f} \tfps={fps:.1f}")
                yolo_time_list.append(yolo_time)
                nlf_time_list.append(nlf_time)
                total_time_list.append(total_time)
                fps_list.append(fps)




    finally:
        cap.release()
        out.release()
        print(f"\nTraitement terminé. Vidéo sauvegardée sous : {output_path}\n")
        print(f"Statistiques: frame total: {total_frames}")
        print(f"yolo:\tmoyenne {np.mean(yolo_time_list):.1f}\t mediane {np.median(yolo_time_list):.1f}\t min {np.min(yolo_time_list):.1f}\t max {np.max(yolo_time_list):.1f}")
        print(f"nlf:\tmoyenne {np.mean(nlf_time_list):.1f}\t mediane {np.median(nlf_time_list):.1f}\t min {np.min(nlf_time_list):.1f}\t max {np.max(nlf_time_list):.1f}")
        print(f"total:\tmoyenne {np.mean(total_time_list):.1f}\t mediane {np.median(total_time_list):.1f}\t min {np.min(total_time_list):.1f}\t max {np.max(total_time_list):.1f}")
        print(f"fps:\tmoyenne {1000.0 / float(np.mean(total_time_list)):.1f}\t mediane {1000.0 / float(np.median(total_time_list)):.1f}\t min {1000.0 / float(np.max(total_time_list)):.1f}\t max {1000.0 / float(np.min(total_time_list)):.1f}")

def draw_projections(img_bgr, poses_3d, intrinsics):
    img_viz = img_bgr.copy()
    poses_3d_np = poses_3d.cpu().numpy()
    K = intrinsics[0].cpu().numpy()

    for person_id in range(poses_3d_np.shape[0]):
        points_3d = poses_3d_np[person_id]
        # On ajuste la taille des points selon le nombre (6890 vs 20)
        radius = 1 if len(points_3d) > 100 else 4
        color = (0, 255, 255) if len(points_3d) > 100 else (0, 0, 255)

        for pt_3d in points_3d:
            x_3d, y_3d, z_3d = pt_3d
            if z_3d != 0:
                u = (K[0, 0] * x_3d / z_3d) + K[0, 2]
                v = (K[1, 1] * y_3d / z_3d) + K[1, 2]
                cv2.circle(img_viz, (int(u), int(v)), radius, color, -1)
    return img_viz

# --- LANCEMENT ---
if __name__ == "__main__":
    video_in = "/root/workspace/RT-COSMIK/data/robot_sanding/camera_0.mp4"
    yolo_p = "/root/workspace/RT-COSMIK/weights/yolo/yolov10n.pt"
    nlf_p = "/root/workspace/RT-COSMIK/weights/nlf/nlf_s_multi_0.2.2.torchscript"
    
    process_video(video_in, yolo_p, nlf_p, "resultat_final.mp4")
