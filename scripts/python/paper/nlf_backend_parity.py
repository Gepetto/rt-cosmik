#!/usr/bin/env python3
"""Parity of the NLF TensorRT backend against TorchScript on the same frames and boxes.

    python3 scripts/python/paper/nlf_backend_parity.py --dataset /root/workspace/COMFI \
        --participant 1012 --task Lifting --out results/nlf_backend_parity.json

Compares the estimator's output (per-view landmarks in each camera frame,
uncertainties), then the joint angles after fusion, filter and IK, over the
first frames of one trial with the paper settings. ``--engine`` adds further
engines (an fp32 build, say) next to ``settings.nlf_engine_path``. The
reference is TorchScript with per-view detection whatever ``settings.py`` says;
the TorchScript file loaded without ``optimize_for_inference`` is compared too.
"""
import argparse
import copy
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np

FRAMES = 300
CAMERAS = [0, 2, 4, 6]


class Trial:
    """One trial's cameras, subject and videos, with the participant's landmark map applied."""

    def __init__(self, dataset, participant, task, cameras, settings):
        import yaml
        from rtcosmik import landmark_maps
        from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation

        self.settings = landmark_maps.apply(landmark_maps.select(settings, participant), settings)
        root = Path(dataset)
        self.participant, self.task, self.cameras = participant, task, cameras
        self.meta = yaml.safe_load((root / "metadata" / f"{participant}.yaml").read_text())
        cam_dir = root / "cam_params" / participant
        self.mtxs, _, self.projections, _, _ = load_camera_parameters(cam_dir, cameras)
        self.world_R, self.world_T = load_world_transformation(cam_dir, cameras[0])
        self.paths = [str(root / "videos" / participant / task / f"camera_{c}.mp4")
                      for c in cameras]

    def frames(self, count):
        from rtcosmik.utils.VideoReader import OfflineVideoSource
        source = OfflineVideoSource(paths=self.paths, loop=False,
                                    size_wh=(self.settings.width, self.settings.height))
        try:
            for _ in range(count):
                images = source.read()
                if images is None:
                    break
                yield images
        finally:
            source.release()

    def estimator(self, nlf_backend, detection_mode, **overrides):
        """A fresh estimator with this backend and detection mode, whatever the settings say."""
        from sweep import set_torch_flags
        from rtcosmik.model_weights import estimator_arguments
        from rtcosmik.nlf.nlf import NLFEstimator
        s = copy.copy(self.settings)
        vars(s).update(overrides, nlf_backend=nlf_backend, detection_mode=detection_mode)
        set_torch_flags()
        return NLFEstimator(
            **estimator_arguments(s, len(self.cameras)), cano_path=s.cano_path,
            image_size=(s.width, s.height), cam_Ks=self.mtxs, indices=s.nlf_indices,
            conf=s.yolo_conf, imgsz=s.yolo_imgsz, device=s.device)

    def joint_angles(self, views_per_frame):
        """Joint coordinates after fusion, filter and IK, one row per fused frame."""
        from rtcosmik.filtering.iir import MarkerFilter
        from rtcosmik.pipeline.solver import HumanSolver
        from rtcosmik.triangulation.triangulation import reconstruct_3d
        s = self.settings
        solver = HumanSolver(s, gender=self.meta["gender"][0], height=self.meta["height"],
                             weight=self.meta["weight"], logger=logging.getLogger("solve"))
        marker_filter = MarkerFilter(len(s.marker_names), s)
        rows = []
        for views in views_per_frame:
            p3d = reconstruct_3d(views, self.projections)
            if len(p3d) == 0:
                continue
            p3d = np.asarray(p3d) @ np.asarray(self.world_R).T + np.asarray(self.world_T)
            rows.append(solver.solve(dict(zip(s.marker_names, marker_filter(p3d)))))
        return np.asarray(rows, dtype=float)

    def differences(self, reference, other):
        """Per-view landmark, pixel, uncertainty and joint angle differences of two recordings."""
        both = [(a, b) for a, b in zip(reference, other)
                if len(a.valid_cam_ids) == len(b.valid_cam_ids) == len(self.cameras)]

        def stacked(side, field):
            return np.stack([np.stack([np.asarray(v) for v in getattr(pair[side], field)])
                             for pair in both])

        d_mm = 1000 * np.linalg.norm(stacked(1, "poses3d") - stacked(0, "poses3d"), axis=-1)
        d_px = np.linalg.norm(stacked(1, "keypoints") - stacked(0, "keypoints"), axis=-1)
        sigma = stacked(0, "uncertainties")
        d_sigma = np.abs(stacked(1, "uncertainties") - sigma)
        is_angle = np.array([n.endswith("[rad]") for n in self.settings.joint_angles_names])
        q_ref, q = self.joint_angles(reference), self.joint_angles(other)
        n = min(len(q), len(q_ref))
        d_q = np.degrees(np.abs(q[:n, is_angle] - q_ref[:n, is_angle]))
        return {
            "frames_compared": len(both),
            "landmark_mm": {"max": float(d_mm.max()), "p99": float(np.percentile(d_mm, 99)),
                            "median": float(np.median(d_mm))},
            "pixel": {"max": float(d_px.max()), "median": float(np.median(d_px))},
            "uncertainty": {"max": float(d_sigma.max()), "median": float(np.median(d_sigma)),
                            "median_relative": float(np.median(d_sigma / sigma))},
            "joint_angle_deg": {"max": float(d_q.max()), "median": float(np.median(d_q)),
                                "rms_per_dof_mean": float(np.sqrt((d_q ** 2).mean(0)).mean())},
        }


def measure(dataset, participant, task, engines, settings, frames=FRAMES, cameras=CAMERAS,
            noise_floor=True):
    """Differences to TorchScript of each engine in ``engines`` (name -> path)."""
    from rtcosmik.nlf.nlf import extract_views

    trial = Trial(dataset, participant, task, cameras, settings)
    settings = trial.settings
    reference = trial.estimator("torchscript", "per_view")
    others = {name: trial.estimator("tensorrt", "per_view", nlf_engine_path=path)
              for name, path in engines.items()}
    if noise_floor:
        import torch
        plain = trial.estimator("torchscript", "per_view")
        plain.nlf0 = torch.jit.load(settings.nlf_path).eval().to(settings.device)
        others["torchscript_not_optimized"] = plain
    recorded = {name: [] for name in ["torchscript", *others]}
    for images in trial.frames(frames):
        out, _, _, boxes = reference.estimate_from_frames(images)
        recorded["torchscript"].append(extract_views(out, len(cameras)))
        for name, est in others.items():
            out = est.landmarks(est.preprocess_batch(images), boxes, {})
            recorded[name].append(extract_views(out, len(cameras)))

    report = {"participant": participant, "task": task, "cameras": cameras,
              "reference": {"nlf_backend": reference.backend,
                            "detection_mode": "per_view" if reference.layout is None else "tiled"},
              "frames": len(recorded["torchscript"]), "landmarks": len(settings.marker_names)}
    for name in others:
        report[name] = trial.differences(recorded["torchscript"], recorded[name])
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--engine", action="append", default=[], help="further engine to compare")
    ap.add_argument("--out", help="JSON report")
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    engines = {Path(p).name: p for p in [settings.nlf_engine_path, *args.engine]}
    report = measure(args.dataset, args.participant, args.task, engines, settings)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
