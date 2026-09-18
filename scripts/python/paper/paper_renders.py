#!/usr/bin/env python3
"""The paper's 3D renderings, drawn in the COMFI scene (``rtcosmik.viewer.comfi_scene``).

Every body is a run's own model, rebuilt as its IK built it (scaled to the first
frame of its markers.csv, as ``robot_distance.Skeleton`` does) and posed with
its joint_angles.csv; the reference is the mocap-driven run. Frames are the
REBA events of ``reba_events.py`` (``reba_events/events.csv``): reference frame
``f`` is arm row ``f + lag`` and, because COMFI's video trails its mocap by
``VIDEO_OFFSET`` frames, video frame ``f + VIDEO_OFFSET``.

Modes:

``check``      the REBA event of each trial: the video frame with the reference
               model laid over it through the calibrated camera, one tile per
               trial, a contact sheet per task (``reba_events_check/``)
``snapshots``  the paper's Fig. 3: the six events of a participant, camera image
               with an arm's estimated model overlaid, plus the colour-coded 3D
               mannequins of the reference and the pipelines
               (``snapshots/<participant>/``)

    python3 scripts/python/paper/paper_renders.py check --participants 1012 1118
    python3 scripts/python/paper/paper_renders.py snapshots --participants 1012
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

DATASET = Path("/root/workspace/COMFI")
REFERENCE_TAG = "mocap_reference"
TASKS = ("Screwing", "Polishing", "SideOverhead", "RobotPolishing", "RobotWelding", "Lifting")
TASK_LABELS = {"Screwing": "Screwing", "Polishing": "Polishing", "SideOverhead": "Overhead",
               "RobotPolishing": "Robot polishing", "RobotWelding": "Robot welding",
               "Lifting": "Lifting"}
#: COMFI's video trails its mocap by about 3 frames (75 ms), a recording offset.
VIDEO_OFFSET = 3
#: One viewpoint for every task: camera 2 sees each of them from the side or front-side.
CAMERA = 2
#: The paper's colours (RGB 0-1), shared with the plots.
COLOURS = {"reference": (0.0, 0.0, 0.0), "nlf_0-2-4-6": (0.18, 0.63, 0.17),
           "mmpose_0-2-4-6": (0.12, 0.35, 0.80), "nlf2d_0-2-4-6": (0.95, 0.55, 0.05),
           "fastsam_0-2-4-6": (0.50, 0.25, 0.65)}
GREY = (0.82, 0.82, 0.82, 1.0)
TILE = (300, 400)        # tile width, height (px) of a contact-sheet cell


def read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


class Run:
    """One run of one trial: its scaled model and its joint angles."""

    def __init__(self, run_dir, meta):
        import example_robot_data as robex
        import pandas as pd
        from rtcosmik.human_model.model_utils import scale_human_model
        markers = pd.read_csv(run_dir / "markers.csv", nrows=1)
        names = sorted({c[:-2] for c in markers.columns if c.endswith("_x")})
        first = {n: markers.loc[0, [f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(float) for n in names}
        gender, height = meta["gender"][0], float(meta["height"])
        human = robex.human.HumanLoader(height=height, weight=meta["weight"], gender=gender).robot
        self.model = scale_human_model(human.model, first, gender=gender, subject_height=height)
        self.visual_model = human.visual_model
        self.q = pd.read_csv(run_dir / "joint_angles.csv").to_numpy(float)
        self.markers = pd.read_csv(run_dir / "markers.csv")


def meta_of(participant):
    import yaml
    return yaml.safe_load((DATASET / "metadata" / f"{participant}.yaml").read_text())


def video_frame(participant, task, camera, index):
    """(RGB image undistorted with the camera's own calibration, K)."""
    import cv2
    from rtcosmik.viewer.comfi_scene import TrialAssets
    cap = cv2.VideoCapture(str(DATASET / "videos" / participant / task / f"camera_{camera}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {index} of {participant}/{task} camera {camera}")
    return image


def lags(results, arm):
    return {(r["participant"], r["task"]): int(r["lag_frames"])
            for r in read(results / "paper" / "per_trial" / f"{arm}.csv")}


def crop_box(alpha, aspect, pad=0.12):
    """A box of the given width/height aspect around the non-transparent pixels."""
    ys, xs = np.nonzero(alpha > 0)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    h = (ys.max() - ys.min()) * (1 + 2 * pad)
    w = max((xs.max() - xs.min()) * (1 + 2 * pad), h * aspect)
    h = w / aspect
    H, W = alpha.shape
    x0 = int(np.clip(cx - w / 2, 0, max(0, W - w)))
    y0 = int(np.clip(cy - h / 2, 0, max(0, H - h)))
    return x0, y0, int(min(w, W)), int(min(h, H))


def overlay(renderer, assets, runs, rows, frame, camera, opacity=0.65):
    """Camera image at video ``frame`` with each run posed at its row, drawn
    through the calibrated camera. ``runs``: name -> (Run, rgba)."""
    import cv2
    from rtcosmik.viewer.comfi_scene import ComfiScene
    from scene_render import composite
    K, dist = assets.intrinsics[camera]
    image = cv2.cvtColor(cv2.undistort(video_frame(assets.participant, assets.task, camera, frame), K, dist),
                         cv2.COLOR_BGR2RGB)
    renderer.vis["bodies"].delete()
    renderer.vis["scene"].delete()
    scene = ComfiScene(renderer.vis, None, show_cameras=False)
    for name, (run, rgba) in runs.items():
        scene.add_body(name, run.model, run.visual_model, rgba=rgba)
    scene.show(None, {name: runs[name][0].q[rows[name]] for name in runs})
    renderer.calibrated(assets.cameras[camera], K, image.shape[1::-1])
    layer = renderer.shot(alpha=True, grid=False)
    return composite(image, layer, opacity), layer[..., 3]


def label_tile(tile, lines):
    """Write short caption lines under a tile."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    extra = 20 * len(lines) + 6
    canvas = Image.new("RGB", (tile.shape[1], tile.shape[0] + extra), "white")
    canvas.paste(Image.fromarray(tile), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((6, tile.shape[0] + 4 + 20 * i), line, fill="black", font=font)
    return np.asarray(canvas)


def sheet(tiles, columns):
    rows = [np.hstack(tiles[i:i + columns] + [np.full_like(tiles[0], 255)] * (columns - len(tiles[i:i + columns])))
            for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


# --------------------------------------------------------------------------- modes

def check(args):
    """Reference model over the video at each trial's REBA event, per task."""
    import cv2
    from rtcosmik.viewer.comfi_scene import TrialAssets
    from scene_render import Renderer
    events = [e for e in read(args.events) if not args.participants or e["participant"] in args.participants]
    out = args.out / "reba_events_check"
    out.mkdir(parents=True, exist_ok=True)
    tiles = {t: [] for t in TASKS}
    with Renderer(size=(1280, 720), scale=1) as renderer:
        for e in events:
            p, t, f = e["participant"], e["task"], int(e["ref_frame"])
            assets = TrialAssets.resolve(DATASET, p, t)
            ref = Run(args.output_dir / p / t / REFERENCE_TAG, meta_of(p))
            image, alpha = overlay(renderer, assets, {"reference": (ref, (0.05, 0.05, 0.05, 0.55))},
                                   {"reference": f}, f + VIDEO_OFFSET, CAMERA, opacity=1.0)
            x, y, w, h = crop_box(alpha, TILE[0] / TILE[1])
            tile = cv2.resize(image[y:y + h, x:x + w], TILE, interpolation=cv2.INTER_AREA)
            value = float(e["value"])
            quantity = f"{value:.0f} deg" if e["unit"] == "deg" else f"{value:.2f} m"
            tile = label_tile(tile, [f"{p} {TASK_LABELS[t]}", f"frame {f} ({float(e['time_s']):.1f} s)",
                                     f"{e['rule']}: {quantity}", f"reference REBA {float(e['reba_ref']):.0f}"])
            cv2.imwrite(str(out / f"{p}_{t}.png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            tiles[t].append(tile)
            print(f"  {p}/{t}: frame {f}", flush=True)
    for t, items in tiles.items():
        if items:
            cv2.imwrite(str(out / f"sheet_{t}.png"), cv2.cvtColor(sheet(items, 6), cv2.COLOR_RGB2BGR))
    everything = [tile for t in TASKS for tile in tiles[t]]
    if everything:
        cv2.imwrite(str(out / "sheet_all.png"),
                    cv2.cvtColor(sheet(everything, max(len(v) for v in tiles.values())), cv2.COLOR_RGB2BGR))
    print(f"{len(everything)} renderings -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check",))
    ap.add_argument("--output-dir", type=Path, default=REPO / "output" / "campaign")
    ap.add_argument("--results", type=Path, default=REPO / "results" / "campaign")
    ap.add_argument("--events", type=Path, default=None, help="events.csv from reba_events.py")
    ap.add_argument("--out", type=Path, default=None, help="where renderings go")
    ap.add_argument("--participants", nargs="*", default=None)
    args = ap.parse_args()
    args.events = args.events or args.results / "paper" / "reba_events" / "events.csv"
    args.out = args.out or args.results / "paper" / "figures"
    {"check": check}[args.mode](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
