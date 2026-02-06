import numpy as np
import torch
from ultralytics import YOLO
import logging
import time
import cv2

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
        K_stack = np.stack(cam_Ks, axis=0)  # (C,3,3)
        self.Kt = torch.from_numpy(K_stack).to(self.device, dtype=self.geom_dtype).unsqueeze(1)  # (C,1,3,3)

        self.Et = torch.eye(4, dtype=self.geom_dtype, device=self.device).unsqueeze(0)
        self.world_up = torch.tensor([0.0, -1.0, 0.0], device=self.device, dtype=self.geom_dtype)

        if warmup:
            self.logger.info("[INFO] Starting to warm up all the models")
            self._warmup(iters=warmup_iters)
    
    def load_nlf(self, path: str):
        model = torch.jit.load(path).eval().to(self.device)

        try:
            model = torch.jit.optimize_for_inference(
                model,
                other_methods=["estimate_poses_batched", "get_weights_for_canonical_points"],
            )
        except RuntimeError as exc:
            self.logger.warning("[WARN] NLF optimize_for_inference skipped: %s", exc)

        return model

    def _warmup(self, iters: int = 10):
        """
        Warm up YOLO + NLF to reduce first-call latency.
        Uses synthetic data but matches your real tensor shapes & camera count.
        """
        C = len(self.Kt)
        H, W = self.height, self.width

        # Fake frames (uint8 BGR)
        frames = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(C)]

        # Dummy bbox: xywh + score (your NLF call uses xywh_score)
        # Here: full image box with a high score
        dummy_bbox = torch.tensor([[0.0, 0.0, float(W - 1), float(H - 1), 1.0]],
                                  device=self.device, dtype=self.geom_dtype)

        # Warm up YOLO (batched)
        with torch.inference_mode():
            for _ in range(iters):
                _ = self.yolo.predict(
                    frames,
                    imgsz=self.imgsz,
                    classes=0,
                    conf=self.conf,
                    device=self.device,
                    verbose=False,
                    half=True,
                )

        # Warm up NLF (per-camera, since NLF isn't batched)
        with torch.inference_mode():
            imgs = self.preprocess_batch(frames)  # (C,3,H,W) fp16
            for _ in range(iters):
                img_0 = imgs[0:1]     # (1,3,H,W)
                Kt_el = self.Kt[0]      # (1,3,3)
                _ = self.nlf0.estimate_poses_batched(
                    img_0,
                    [dummy_bbox],
                    intrinsic_matrix=Kt_el,
                    extrinsic_matrix=self.Et,
                    world_up_vector=self.world_up,
                    weights=self.weights,
                    num_aug=1,
                )

        torch.cuda.synchronize()

    def top1_box_xywh_score(self, res):
        if len(res.boxes) == 0:
            return torch.zeros((0, 5), device=self.device, dtype=self.geom_dtype)
        j = int(torch.argmax(res.boxes.conf).item())
        boxes = res.boxes.xyxy[j:j + 1].to(self.device)
        scores = res.boxes.conf[j:j + 1].to(self.device).unsqueeze(1)
        wh = boxes[:, 2:] - boxes[:, :2]
        return torch.cat([boxes[:, :2], wh, scores], dim=1).contiguous().to(self.geom_dtype)

    def preprocess_batch(self, frames_bgr):
        arr = np.ascontiguousarray(np.stack(frames_bgr, axis=0))   # (C,H,W,3) uint8
        x = torch.from_numpy(arr).to(self.device)  # single transfer
        x = x[..., [2, 1, 0]]                                       # BGR->RGB
        x = x.permute(0, 3, 1, 2).contiguous()                      # (C,3,H,W)
        x = x.to(torch.float16).mul_(1.0 / 255.0)                   # try fp16 for speed
        return x

    
    @torch.inference_mode()
    def estimate_from_frames(self, frames_bgr):
        t0 = time.perf_counter()

        # YOLO batched
        yres = self.yolo.predict(
            frames_bgr,
            imgsz=self.imgsz,
            classes=0,
            conf=self.conf,
            device=self.device,   # keep consistent with user input
            verbose=False,
            half=True,
        )

        # Top-1 bbox per image (each b[i] is (1,5) or (0,5))
        b = [self.top1_box_xywh_score(y) for y in yres]

        C = len(frames_bgr)
        if C != len(self.Kt):
            raise ValueError(f"Need {len(self.Kt)} frames, got {C}")

        # One GPU upload + format for all images
        imgs = self.preprocess_batch(frames_bgr)  # (C,3,H,W)

        out = [None] * C
        for i in range(C):
            b_el = b[i]
            if b_el.numel() == 0:
                continue

            img_i = imgs[i:i+1]      # (1,3,H,W) keeps batch dim
            Kt_el = self.Kt[i]       # you stored (1,3,3)
            # Et is already (1,4,4)

            # Keep whatever NLF expects here; common pattern is list-of-boxes
            out[i] = self.nlf0.estimate_poses_batched(
                img_i,
                [b_el],
                intrinsic_matrix=Kt_el,
                extrinsic_matrix=self.Et,
                world_up_vector=self.world_up,
                weights=self.weights,
                num_aug=1,
            )

        t1 = time.perf_counter()
        return out, (t1 - t0) * 1000.0, yres, b

    @staticmethod
    def _extract_poses3d(nlf_out):
        """Best-effort extraction of a (P,J,3) torch.Tensor from NLF output."""
        if nlf_out is None:
            return None

        # Common case in your bench: dict with key "poses3d"
        if isinstance(nlf_out, dict) and "poses3d" in nlf_out:
            poses = nlf_out["poses3d"]
            # Often wrapped as length-1 list/tuple (batch)
            if isinstance(poses, (list, tuple)) and len(poses) > 0:
                poses = poses[0]
            if torch.is_tensor(poses):
                # Accept (P,J,3) or (1,P,J,3)
                if poses.ndim == 4 and poses.shape[0] == 1:
                    poses = poses[0]
                return poses

        # If TorchScript returns a tuple/list, try to find a plausible tensor
        if isinstance(nlf_out, (list, tuple)):
            for item in nlf_out:
                if torch.is_tensor(item):
                    if item.ndim == 4 and item.shape[-1] == 3:
                        return item[0] if item.shape[0] == 1 else item
                    if item.ndim == 3 and item.shape[-1] == 3:
                        return item

        return None

    @staticmethod
    def draw_projection(frame_bgr, poses_3d, K, color=(0, 255, 255)):
        """Draws projected 3D points (no skeleton) on a BGR image."""
        if poses_3d is None:
            return frame_bgr

        img = frame_bgr.copy()
        pts = poses_3d.detach().float().cpu().numpy()
        h, w = img.shape[:2]

        K_new = K.detach().float().cpu().numpy()[0]

        # pts: (P,J,3)
        for person in pts:
            radius = 1 if len(person) > 100 else 3
            for x, y, z in person:
                if abs(z) < 1e-9:
                    continue
                u = (K_new[0, 0] * x / z) + K_new[0, 2]
                v = (K_new[1, 1] * y / z) + K_new[1, 2]
                ui, vi = int(round(u)), int(round(v))
                if 0 <= ui < w and 0 <= vi < h:
                    cv2.circle(img, (ui, vi), radius, color, -1, lineType=cv2.LINE_AA)
        return img

    @staticmethod
    def draw_bbox_xywh(frame_bgr, bbox_xywh_score, color=(0, 255, 0), thickness=2):
        """Optional helper to draw the top-1 bbox used for NLF."""
        if bbox_xywh_score is None or bbox_xywh_score.numel() == 0:
            return frame_bgr
        img = frame_bgr.copy()
        b = bbox_xywh_score.detach().float().cpu().numpy()[0]
        x, y, w, h = b[:4]
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
        if len(frames_bgr) != len(nlf_outputs):
            raise ValueError(f"frames_bgr and nlf_outputs length mismatch: {len(frames_bgr)} vs {len(nlf_outputs)}")
        if boxes is not None and len(boxes) != len(frames_bgr):
            raise ValueError(f"boxes length mismatch: {len(boxes)} vs {len(frames_bgr)}")
        if len(frames_bgr) != self.Kt.shape[0]:
            raise ValueError(f"Need {self.Kt.shape[0]} frames (one per camera), got {len(frames_bgr)}")

        out_frames = []
        for i, (frm, nlf_out) in enumerate(zip(frames_bgr, nlf_outputs)):
            img = frm.copy()
            if draw:
                poses3d = self._extract_poses3d(nlf_out)
                img = self.draw_projection(img, poses3d, self.Kt[i])
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

