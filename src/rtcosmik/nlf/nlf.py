import numpy as np
import torch
from ultralytics import YOLO
import logging
import time
import cv2
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from multiprocessing import Process, Array, Value, Lock, Barrier, Event, Queue
from src.rtcosmik.triangulation.triangulation import triangulate_points

LOGGER = logging.getLogger(__name__)

class NLFEstimator:
    """Roll YOLO detection (batched) + per-camera NLF sequential estimation for multiple images."""

    def __init__(
        self,
        yolo_path,
        nlf_path,
        cano_path,
        image_size,
        cam_Ks,
        indices,
        all_in_one=False, # performs detection + nlf all in one or not 
        conf=0.75,
        imgsz=640,
        device="cuda:0",
        logger=None,
        warmup=True,
        warmup_iters=10,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.device = device
        self.logger = logger or LOGGER
        self.conf = conf
        self.imgsz = imgsz
        self.width, self.height = image_size
        self.indices = indices

        self.logger.info(f"[INFO] Loading YOLO detector model at {yolo_path}")
        self.yolo = YOLO(yolo_path, task="detect")

        self.logger.info(f"[INFO] Loading NLF model at {nlf_path}")
        self.nlf0 = self.load_nlf(nlf_path)

        self.logger.info(f"[INFO] Loading canonical vertices at {cano_path}")
        cano = np.load(cano_path)
        pts = torch.from_numpy(cano[self.indices]).float().to(self.device)
        with torch.inference_mode():
            self.weights = self.nlf0.get_weights_for_canonical_points(pts)

        self.geom_dtype = torch.float32
        K_stack = np.stack(cam_Ks, axis=0) 
        self.Kt = torch.from_numpy(K_stack).to(self.device, dtype=self.geom_dtype)  # (C,3,3)

        if warmup:
            self.logger.info("[INFO] Starting to warm up all the models")
            if all_in_one:
                self._warmup_all_in_one(iters=warmup_iters)
            else:
                self._warmup(iters=warmup_iters)
            self.logger.info("[INFO] Models warmed up")
    
    def load_nlf(self, path: str):
        model = torch.jit.load(path).eval().to(self.device)

        def _nop(*args, **kwargs):
            return None

        try:
            model.forward = _nop
        except Exception:
            pass
        
        try:
            model = torch.jit.optimize_for_inference(
                model,
                other_methods=["detect_poses_batched", "estimate_poses_batched", "get_weights_for_canonical_points"],
            )
        except RuntimeError as exc:
            self.logger.warning("[WARN] NLF optimize_for_inference skipped: %s", exc)

        return model

    def _warmup(self, iters: int = 10):
        C = len(self.Kt)
        H, W = self.height, self.width

        frames = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(C)]
        dummy_xywh = torch.tensor([[0.0, 0.0, float(W - 1), float(H - 1)]],
                                  device=self.device, dtype=self.geom_dtype)
        boxes_list = [dummy_xywh.clone() for _ in range(C)]

        with torch.inference_mode():
            for _ in range(iters):
                _ = self.yolo.predict(
                    frames, imgsz=self.imgsz, classes=0, conf=self.conf,
                    device=self.device, verbose=False, half=True,
                )

        with torch.inference_mode():
            imgs = self.preprocess_batch(frames)  # (C,3,H,W)

            for _ in range(iters):
                # One single batched call
                _ = self.nlf0.estimate_poses_batched(
                    imgs,
                    boxes_list,
                    intrinsic_matrix=self.Kt,   # see note below if shape mismatch
                    weights=self.weights,
                    num_aug=1,
                )
    
    def _warmup_all_in_one(self, iters: int = 10):
        C = len(self.Kt)
        H, W = self.height, self.width

        frames = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(C)]

        with torch.inference_mode():
            imgs = self.preprocess_batch(frames)  # (C,3,H,W)

            for _ in range(iters):
                # One single batched call
                _ = self.nlf0.detect_poses_batched(
                    imgs,
                    intrinsic_matrix=self.Kt,   # see note below if shape mismatch
                    weights=self.weights,
                    num_aug=1,
                )

    def top1_box_xywh(self, res):
        """Return top-1 bbox as (1,4) xywh on self.device, or (0,4) if none."""
        if len(res.boxes) == 0:
            return torch.zeros((0, 4), device=self.device, dtype=self.geom_dtype)

        j = int(torch.argmax(res.boxes.conf).item())
        boxes_xyxy = res.boxes.xyxy[j:j + 1].to(self.device)  # (1,4) xyxy
        wh = boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]
        xywh = torch.cat([boxes_xyxy[:, :2], wh], dim=1)       # (1,4) xywh
        return xywh.contiguous().to(self.geom_dtype)

    def preprocess_batch(self, frames_bgr):
        arr = np.ascontiguousarray(np.stack(frames_bgr, axis=0))   # (C,H,W,3) uint8
        x = torch.from_numpy(arr).to(self.device)                  # single transfer
        x = x[..., [2, 1, 0]]                                       # BGR->RGB
        x = x.permute(0, 3, 1, 2).contiguous()                      # (C,3,H,W)
        x = x.to(torch.float16).mul_(1.0 / 255.0)                   # try fp16 for speed
        return x

    @torch.inference_mode()
    def estimate_from_frames(self, frames_bgr):
        C = len(frames_bgr)
        assert C == len(self.Kt)

        # --- CPU preprocess timing (stacking etc.) ---
        t_cpu0 = time.perf_counter()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # YOLO
        yres = self.yolo.predict(
            frames_bgr, imgsz=self.imgsz, classes=0, conf=self.conf,
            device=self.device, verbose=False, half=True
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        boxes = [self.top1_box_xywh(y) for y in yres]

        # preprocess_batch includes H2D; count it separately
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        imgs = self.preprocess_batch(frames_bgr)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # NLF
        out = self.nlf0.estimate_poses_batched(
            imgs, boxes, intrinsic_matrix=self.Kt, weights=self.weights, num_aug=1
        )
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        timings = {
            "yolo_ms": (t1 - t0) * 1000.0,
            "h2d+pre_ms": (t3 - t2) * 1000.0,
            "nlf_ms": (t4 - t3) * 1000.0,
            "total_ms": (t4 - t0) * 1000.0,
            "cpu_overhead_ms": (time.perf_counter() - t_cpu0) * 1000.0,  # small sanity
        }
        return out, timings, yres, boxes

    @torch.inference_mode()
    def detect_and_estimate_from_frames(self, frames_bgr):
        C = len(frames_bgr)
        assert C == len(self.Kt)

        # --- CPU preprocess timing (stacking etc.) ---
        t_cpu0 = time.perf_counter()

        # preprocess_batch includes H2D; count it separately
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        imgs = self.preprocess_batch(frames_bgr)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # NLF
        out = self.nlf0.detect_poses_batched(
            imgs, intrinsic_matrix=self.Kt, weights=self.weights, num_aug=1
        )
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        timings = {
            "h2d+pre_ms": (t3 - t2) * 1000.0,
            "nlf_ms": (t4 - t3) * 1000.0,
            "total_ms": (t4 - t_cpu0) * 1000.0,
            "cpu_overhead_ms": (time.perf_counter() - t_cpu0) * 1000.0,  # small sanity
        }
        return out, timings


    @staticmethod
    def draw_points(frame_bgr, poses_2d, color=(0, 255, 255)):
        """Draws projected 3D points (no skeleton) on a BGR image."""
        if poses_2d is None:
            return frame_bgr

        img = frame_bgr.copy()
        pts = poses_2d.detach().float().cpu().numpy()
        h, w = img.shape[:2]

        # pts: (P,J,2)
        for person in pts:
            radius = 1 if len(person) > 100 else 3
            for x, y in person:
                ui, vi = int(round(x)), int(round(y))
                if 0 <= ui < w and 0 <= vi < h:
                    cv2.circle(img, (ui, vi), radius, color, -1, lineType=cv2.LINE_AA)
        return img

    @staticmethod
    def draw_bbox_xywh(frame_bgr, b, color=(0, 255, 0), thickness=2):
        """Optional helper to draw the top-1 bbox used for NLF."""
        img = frame_bgr.copy()
        x, y, w, h = b.detach().float().cpu().numpy()[0]
        p0 = (int(round(x)), int(round(y)))
        p1 = (int(round(x + w)), int(round(y + h)))
        cv2.rectangle(img, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        return img

    def visualize_frames(
        self,
        frames_bgr,
        nlf_outputs,
        boxes=None,
        draw_boxes=False,
        draw=True,
        put_text=False,
        text_prefix="cam",
    ):
        """Return a list of frames with projected NLF keypoints drawn.

        - frames_bgr: list of BGR images
        - nlf_outputs: list aligned with frames (the `out` from estimate_from_frames)
        - boxes: optional list of bbox tensors aligned with frames
        - draw_boxes: if True, draws the bbox used for NLF
        - draw: if False, returns copies of frames without drawing
        """
        if len(frames_bgr) != len(nlf_outputs["poses2d"]):
            raise ValueError(f"frames_bgr and nlf_outputs length mismatch: {len(frames_bgr)} vs {len(nlf_outputs)}")
        if boxes is not None and len(boxes) != len(frames_bgr):
            raise ValueError(f"boxes length mismatch: {len(boxes)} vs {len(frames_bgr)}")
        if len(frames_bgr) != self.Kt.shape[0]:
            raise ValueError(f"Need {self.Kt.shape[0]} frames (one per camera), got {len(frames_bgr)}")

        out_frames = []
        for i, (frm, nlf_out) in enumerate(zip(frames_bgr, nlf_outputs["poses2d"])):
            img = frm.copy()

            if draw:
                img = self.draw_points(img, nlf_out)
                if draw_boxes and boxes is not None:
                    img = self.draw_bbox_xywh(img, boxes[i])

            if put_text:
                cv2.putText(
                    img,
                    f"{text_prefix}{i}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

            out_frames.append(img)
        return out_frames

class DisplayConsumerNLF(Process):
    def __init__(self,
                 settings,
                 frame_counters,
                 camera_buffers, 
                 camera_locks, 
                 timestamp_buffers, 
                 stop_event, 
                 mtxs,
                 frame_shape: tuple = (720, 1280, 3),
                 num_cameras: int = 2,
                 with_triangul=False,
                 world_R1_cam=None,
                 world_T1_cam=None,
                 dists=None,
                 projections=None,
                 logger=None,
                 ):
        super().__init__()
        self.camera_buffers = camera_buffers
        self.camera_locks = camera_locks
        self.timestamp_buffers = timestamp_buffers
        self.frame_shape = frame_shape  # (height, width, channels)
        self.num_cameras = num_cameras
        self.stop_event = stop_event

        self.last_frame_counters = [0] * self.num_cameras
        self.frame_counters = frame_counters

        self.yolo_path=settings.yolo_path
        self.nlf_path=settings.nlf_path
        self.cano_path=settings.cano_path
        self.marker_names=settings.marker_names

        self.mtxs=mtxs
        self.dists=dists
        self.projections=projections

        self.nlf_indices=settings.nlf_indices
        self.yolo_conf=settings.yolo_conf
        self.yolo_imgsz=settings.yolo_imgsz
        self.device=settings.device 

        self.with_triangul=with_triangul
        self.world_R1_cam=world_R1_cam
        self.world_T1_cam=world_T1_cam
        self.logger = logger or LOGGER


    def run(self):

        est = NLFEstimator(
            yolo_path=self.yolo_path,
            nlf_path=self.nlf_path,
            cano_path=self.cano_path,
            image_size=(self.frame_shape[1], self.frame_shape[0]),
            cam_Ks=self.mtxs,
            indices=self.nlf_indices,
            conf=self.yolo_conf,
            imgsz=self.yolo_imgsz,
            device=self.device,
        )

        if self.with_triangul:
            
            # --- 1. INITIALISATION MESHCAT ---
            vis = meshcat.Visualizer()
            self.logger.info(f"[INFO] Meshcat visualizer available here: {vis.url()}")

            vis_markers = vis["markers"]

            if self.world_R1_cam is None or self.world_T1_cam is None:
                raise TypeError("with_triangul=True requires world_R1_cam and world_T1_cam")

            world_M_cam = np.eye(4, dtype=np.float64)
            world_M_cam[:3, :3] = self.world_R1_cam
            world_M_cam[:3, 3] = self.world_T1_cam
            vis_markers.set_transform(world_M_cam)

            if self.dists is None or self.projections is None:
                raise TypeError("For triangulation, please provide dists and projections")

            # Names aligned 1-to-1 with nlf_indices order
            J = len(self.marker_names)

            right_joint_ids = [i for i, n in enumerate(self.marker_names) if n.startswith("R") or n.startswith("right_")]
            left_joint_ids  = [i for i, n in enumerate(self.marker_names) if n.startswith("L") or n.startswith("left_")]

            # Arm medial points only
            right_arm_medial_ids = [self.marker_names.index("RMELB"), self.marker_names.index("RMWRI")]
            left_arm_medial_ids  = [self.marker_names.index("LMELB"), self.marker_names.index("LMWRI")]

            try: 
                while not self.stop_event.is_set():
                    frames = []
                    new_counters = []
                    for i, (lock, buffer, cam_ts, frame_counter) in enumerate(zip(self.camera_locks, self.camera_buffers, self.timestamp_buffers, self.frame_counters)):
                        with lock:
                            #  Only accept data if this camera has produced a new frame
                            if frame_counter.value > self.last_frame_counters[i]:
                                # Read and copy shared data atomically
                                arr = np.frombuffer(buffer, dtype=np.uint8)
                                frame = arr.reshape(self.frame_shape).copy()
                                # Get current timestamp
                                timestamp = bytes(cam_ts[:]).decode().strip('\x00')

                                if timestamp == '': # empty data
                                    continue
                                else:
                                    frames.append(frame)
                                new_counters.append(frame_counter.value)
                    
                    if len(frames)!=self.num_cameras:
                        continue

                    self.last_frame_counters = new_counters.copy()

                    nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)

                    nlf_out_2d = nlf_out["poses2d"]

                    if nlf_out_2d is None or len(nlf_out_2d) < self.num_cameras:
                        continue

                    keypoints_list = [None] * self.num_cameras
                    valid_cam_ids = []

                    for ii in range(self.num_cameras):
                        poses2d = nlf_out_2d[ii]
                        
                        if poses2d is None or len(poses2d) == 0 or poses2d[0] is None:
                            continue

                        keypoints_list[ii] = poses2d[0].detach().float().cpu().numpy()
                        valid_cam_ids.append(ii)

                    if len(valid_cam_ids) < 2:
                        continue
                    
                    p3d = triangulate_points(
                        keypoints_list=keypoints_list,
                        mtxs=self.mtxs,
                        dists=self.dists,
                        projections=self.projections,
                    )

                    poses_triangul = torch.from_numpy(p3d).to(dtype=torch.float32)

                    if poses_triangul.shape[0] > 0:
                        points_all = poses_triangul.view(-1, 3).cpu().numpy().T  # (3, N)
                        N = points_all.shape[1]

                        # Joint id per point (works if points are flattened as [p0 joints..., p1 joints..., ...])
                        joint_ids = np.arange(N) % J

                        # Default: midline/other = blue
                        colors = np.tile(np.array([[0.0], [0.0], [1.0]], dtype=np.float32), (1, N))

                        # Right body = green
                        mask_right = np.isin(joint_ids, right_joint_ids)
                        colors[:, mask_right] = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)

                        # Left body = red
                        mask_left = np.isin(joint_ids, left_joint_ids)
                        colors[:, mask_left] = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)

                        # Tiny modification: arm medial points get a distinct color to separate medial vs lateral
                        mask_r_med = np.isin(joint_ids, right_arm_medial_ids)
                        mask_l_med = np.isin(joint_ids, left_arm_medial_ids)

                        colors[:, mask_r_med] = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)  # right medial = black
                        colors[:, mask_l_med] = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)  # left medial = black

                        vis_markers.set_object(
                            g.PointCloud(position=points_all, color=colors, size=0.02)
                        )
            finally:        
                self.logger.info("[INFO] Display NLF Process with triangul terminated")
        else:
            cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL)

            try: 
                while not self.stop_event.is_set():
                    frames = []
                    new_counters = []
                    for i, (lock, buffer, cam_ts, frame_counter) in enumerate(zip(self.camera_locks, self.camera_buffers, self.timestamp_buffers, self.frame_counters)):
                        with lock:
                            #  Only accept data if this camera has produced a new frame
                            if frame_counter.value > self.last_frame_counters[i]:
                                # Read and copy shared data atomically
                                arr = np.frombuffer(buffer, dtype=np.uint8)
                                frame = arr.reshape(self.frame_shape).copy()
                                # Get current timestamp
                                timestamp = bytes(cam_ts[:]).decode().strip('\x00')

                                if timestamp == '': # empty data
                                    continue
                                else:
                                    frames.append(frame)
                                new_counters.append(frame_counter.value)
                    
                    if len(frames)!=self.num_cameras:
                        if self.logger:
                            self.logger.debug(f"[WARN] one of the camera frames is missing, skip")
                        continue

                    self.last_frame_counters = new_counters.copy()

                    nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)

                    print(f"Timings to perform inference = {infer_ms}")

                    vis_frames = est.visualize_frames(
                        frames,
                        nlf_out,
                        boxes=boxes,
                        draw_boxes=True,
                        put_text=True,
                        text_prefix="cam",
                    )
                    vis = np.hstack(vis_frames)

                    cv2.imshow("Visualization", vis)
                        
                    # Break on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:        
                cv2.destroyAllWindows()
                self.logger.info("[INFO] Display NLF Process terminated")