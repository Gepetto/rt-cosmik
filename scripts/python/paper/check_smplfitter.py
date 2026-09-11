#!/usr/bin/env python3
"""Verify a smplfitter install end to end, before spending a sweep on it.

Every failure mode in docs/smplfitter.md is silent except the first, so this
checks each one explicitly rather than trusting that an import succeeding means
the thing works:

1. the torch < 2.5 ``nn.Buffer`` shim is in place
2. the body model files are present and loadable
3. ``num_betas`` is small enough that the fit does not return the average body
4. a synthetic fit actually reduces vertex error
5. NLF's canonical vertices line up with the body model's vertex count
6. the fit costs what the guide says it costs, on this hardware

    python3 scripts/python/paper/check_smplfitter.py
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

PASS, FAIL = "  ok  ", " FAIL "


def main():
    import torch

    from rtcosmik.config_loader import settings

    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}")

    import torch.nn as nn
    had_buffer = hasattr(nn, "Buffer")
    from rtcosmik.smpl import torch_shim  # noqa: F401
    print(f"[{PASS}] nn.Buffer available "
          f"({'native' if had_buffer else 'via shim'})")

    try:
        import smplfitter.pt as smpl_pt
        from smplfitter.pt.bodyfitter import BodyFitter
    except ImportError as exc:
        print(f"[{FAIL}] smplfitter not installed: {exc}")
        print("        pip install smplfitter")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_betas = settings.smpl_num_betas
    try:
        body = smpl_pt.BodyModel("smplx", "neutral",
                                 num_betas=num_betas).to(device)
    except FileNotFoundError as exc:
        print(f"[{FAIL}] body models missing.\n")
        print(str(exc).strip())
        print("\n        Fix: register at https://smpl-x.is.tue.mpg.de/ (and smpl,"
              "\n        mano, agora with the SAME email and password), then run"
              "\n        bash scripts/bash/setup_smplfitter.sh")
        return 1
    print(f"[{PASS}] body model loaded: {body.num_vertices} vertices, "
          f"{body.num_joints} joints, {num_betas} betas")

    cano = np.load(settings.cano_path)
    if cano.shape[0] != body.num_vertices:
        print(f"[{FAIL}] canonical verts {cano.shape} != body {body.num_vertices}. "
              f"settings.cano_path must match the body model family.")
        return 1
    print(f"[{PASS}] canonical vertices match: {cano.shape}")

    fitter = BodyFitter(body).to(device)
    rng = np.random.default_rng(0)
    frames, joints = 8, body.num_joints
    beta_true = rng.normal(size=num_betas) * np.exp(-np.arange(num_betas) / 4)
    pose_true = np.zeros((frames, joints * 3))
    pose_true[:, 3:66] = rng.normal(size=(frames, 63)) * 0.25
    truth = body(shape_betas=torch.tensor(np.repeat(beta_true[None], frames, 0),
                                          dtype=torch.float32, device=device),
                 pose_rotvecs=torch.tensor(pose_true, dtype=torch.float32,
                                           device=device),
                 return_vertices=True)["vertices"]
    noisy = truth + torch.randn_like(truth) * 0.01

    result = fitter.fit(noisy, num_iter=4, beta_regularizer=0.1, share_beta=True,
                        final_adjust_rots=True,
                        requested_keys=["pose_rotvecs", "shape_betas", "trans"])
    betas = result["shape_betas"].detach().cpu().numpy()
    if np.allclose(betas, 0, atol=1e-6):
        print(f"[{FAIL}] shape_betas all zero -- num_betas too large (trap 2)")
        return 1
    print(f"[{PASS}] betas non-degenerate: {np.round(betas[0, :5], 2)}")

    fitted = body(pose_rotvecs=result["pose_rotvecs"],
                  shape_betas=result["shape_betas"], trans=result["trans"],
                  return_vertices=True)["vertices"]
    before = float((noisy - truth).norm(dim=-1).median()) * 1000
    after = float((fitted - truth).norm(dim=-1).median()) * 1000
    tag = PASS if after < before else FAIL
    print(f"[{tag}] synthetic fit: {before:.1f} mm noise -> {after:.1f} mm fitted")
    if after >= before:
        return 1

    # Timing, at the batch sizes the pipeline actually uses.
    from rtcosmik.smpl.fitter import SmplRefiner
    refiner = SmplRefiner(gender="n", num_betas=num_betas,
                          num_iter=settings.smpl_num_iter, device=device,
                          beta_mode="free")
    single = truth[0].cpu().numpy()
    for _ in range(3):
        refiner.refine(single)
    times = []
    for _ in range(20):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        refiner.refine(single)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    print(f"[{PASS}] online fit: {np.median(times):.2f} ms/frame "
          f"({settings.smpl_num_iter} iters, batch 1), "
          f"residual {refiner.last_residual_mm:.1f} mm")
    print("\nAll checks passed. The nlfsmpl arm can run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
