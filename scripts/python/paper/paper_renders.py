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
from paper_figures import STYLE, FOUR_CAMERAS, MM, TWO_COLUMNS, setup as plot_setup

def _rgb(hex_colour):
    return tuple(int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))

#: The plots' colours (RGB 0-1), so a front end looks the same in every figure.
COLOURS = {"reference": _rgb(STYLE["reference"]["color"]),
           **{arm: _rgb(STYLE[key]["color"]) for key, arm in FOUR_CAMERAS.items()}}
LABELS = {"reference": "Reference", **{arm: STYLE[key]["label"] for key, arm in FOUR_CAMERAS.items()}}
GREY = (0.82, 0.82, 0.82, 1.0)
ESTIMATE_OPACITY = 0.45
#: 3D variants of the snapshots: which bodies each shows (at most three).
VARIANTS = {"reference": ("reference",), "nlf3d": ("reference", "nlf_0-2-4-6"),
            "rtmpose": ("reference", "mmpose_0-2-4-6"),
            "three": ("reference", "nlf_0-2-4-6", "mmpose_0-2-4-6")}
FIGURE_VARIANT = "three"
#: 3D view of a snapshot, the same relative to the body in every column: from
#: its right side turned this far towards its front (trunk flexion reads in
#: profile, the arms stay visible), this high above the horizontal, this far
#: from the pelvis.
VIEW_FROM_SIDE_DEG, VIEW_ELEVATION_DEG, VIEW_DISTANCE_M, VIEW_FOV_DEG = 35.0, 15.0, 4.2, 30.0
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

    def pelvis(self, row):
        """World position of the model's root (pelvis) at a row, by forward kinematics."""
        import pinocchio as pin
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, self.q[row])
        return np.array(data.oMi[1].translation)


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


def typical_participants(results, count=3):
    """The participants whose NLF-3D (4 cameras) whole-body RMSE is closest to the median."""
    per = {}
    for r in read(results / "paper" / "per_trial" / "nlf_0-2-4-6.csv"):
        per.setdefault(r["participant"], []).append(float(r["joint_rmse_deg"]))
    means = {p: np.mean(v) for p, v in per.items()}
    median = np.median(list(means.values()))
    return sorted(means, key=lambda p: abs(means[p] - median))[:count]


def view_of_body(markers, row, target):
    """Eye position looking at ``target`` from the body's right-front, from the
    pelvis markers at ``row`` (forward: PSIS to ASIS, horizontal)."""
    xyz = lambda n: markers[[f"{n}_x", f"{n}_y", f"{n}_z"]].iloc[row].to_numpy(float)
    forward = (xyz("RASI") + xyz("LASI")) / 2 - (xyz("RPSI") + xyz("LPSI")) / 2
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    a = np.radians(VIEW_FROM_SIDE_DEG)
    towards = np.cos(a) * right + np.sin(a) * forward
    e = np.radians(VIEW_ELEVATION_DEG)
    return target + VIEW_DISTANCE_M * (np.cos(e) * towards + np.array([0.0, 0.0, np.sin(e)]))


def snapshots(args):
    """Fig. 3: the six REBA events of a participant, camera image with NLF-3D
    overlaid (a) and the 3D scene with the colour-coded bodies (b)."""
    import cv2
    from rtcosmik.viewer.comfi_scene import ComfiScene, TrialAssets
    from scene_render import Renderer
    events = {(e["participant"], e["task"]): e for e in read(args.events)}
    participants = args.participants or typical_participants(args.results)
    lag = {arm: lags(args.results, arm) for arm in FOUR_CAMERAS.values()}
    shown = ["reference", "nlf_0-2-4-6", "mmpose_0-2-4-6"]
    for p in participants:
        out = args.out / "snapshots" / p
        out.mkdir(parents=True, exist_ok=True)
        overlays, views, rows = [], {v: [] for v in VARIANTS}, []
        meta = meta_of(p)
        with Renderer(size=(1280, 720), scale=1) as cam_view, Renderer(size=(900, 1200), scale=1) as free:
            for t in TASKS:
                e = events[(p, t)]
                f = int(e["ref_frame"])
                assets = TrialAssets.resolve(DATASET, p, t)
                runs = {"reference": Run(args.output_dir / p / t / REFERENCE_TAG, meta),
                        **{a: Run(args.output_dir / p / t / a, meta) for a in shown[1:]}}
                row_of = {"reference": f, **{a: f + lag[a][(p, t)] for a in shown[1:]}}
                # (a) camera image with NLF-3D laid over it.
                image, alpha = overlay(cam_view, assets,
                                       {"nlf": (runs["nlf_0-2-4-6"], (*COLOURS["nlf_0-2-4-6"], 0.6))},
                                       {"nlf": row_of["nlf_0-2-4-6"]}, f + VIDEO_OFFSET, CAMERA, opacity=1.0)
                x, y, w, h = crop_box(alpha, 3 / 4)
                overlays.append(cv2.resize(image[y:y + h, x:x + w], (450, 600), interpolation=cv2.INTER_AREA))
                # (b) the scene, every variant, same view direction and distance.
                free.vis["bodies"].delete()
                free.vis["scene"].delete()
                scene = ComfiScene(free.vis, assets, show_cameras=False, table_rgba=GREY, robot_rgba=GREY)
                for name in shown:
                    rgba = (*COLOURS[name], 1.0 if name == "reference" else ESTIMATE_OPACITY)
                    scene.add_body(name, runs[name].model, runs[name].visual_model, rgba=rgba)
                scene.show(f + VIDEO_OFFSET, {n: runs[n].q[row_of[n]] for n in shown})
                target = runs["reference"].pelvis(f)
                target[2] = 0.85
                free.look_at(view_of_body(runs["reference"].markers, f, target), target, fov_deg=VIEW_FOV_DEG)
                for variant, bodies in VARIANTS.items():
                    for name in shown:
                        free.vis["bodies"][name].set_property("visible", name in bodies)
                    views[variant].append(free.shot()[..., :3])
                rows.append([p, t, e["rule"], f, f + VIDEO_OFFSET, CAMERA] + [row_of[a] for a in shown[1:]]
                            + [e["reba_ref"]])
                print(f"  {p}/{t}: frame {f}", flush=True)
        for variant, tiles in views.items():
            cv2.imwrite(str(out / f"3d_{variant}.png"), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "camera_nlf3d.png"), cv2.cvtColor(np.hstack(overlays), cv2.COLOR_RGB2BGR))
        compose_snapshots(args, p, overlays, views[FIGURE_VARIANT], VARIANTS[FIGURE_VARIANT], rows)


def compose_snapshots(args, participant, top, bottom, bodies, rows):
    """Two rows of six, 181 mm wide: (a) camera images, (b) 3D views; task names
    under the columns, the colour key above."""
    from matplotlib.patches import Patch
    plt = plot_setup()
    fig, axes = plt.subplots(2, 6, figsize=(TWO_COLUMNS, 94 * MM),
                             gridspec_kw={"hspace": 0.03, "wspace": 0.03})
    for k, t in enumerate(TASKS):
        axes[0, k].imshow(top[k])
        axes[1, k].imshow(bottom[k])
        axes[1, k].set_xlabel(TASK_LABELS[t], labelpad=2)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for r, text in enumerate(("(a)", "(b)")):
        axes[r, 0].set_ylabel(text, rotation=0, ha="right", va="center", labelpad=4)
    handles = [Patch(facecolor=COLOURS[b], alpha=1.0 if b == "reference" else 0.6, label=LABELS[b])
               for b in bodies]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=len(handles),
               frameon=False, handlelength=1.2, columnspacing=1.5)
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.06, top=0.93)
    out = args.out / "snapshots"
    name = f"snapshots_{participant}"
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=200)
    plt.close(fig)
    with open(out / f"{name}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "task", "rule", "reference_frame", "video_frame", "camera",
                         "NLF-3D 4 row", "RTMPose+LSTM 4 row", "reference_REBA"])
        writer.writerows(rows)
    print(f"  {name}: pdf, png, csv -> {out}", flush=True)


SETUP_TRIAL = ("1118", "RobotWelding")   # a session with the robot in its usual place
CAPTURE_HEIGHT_M = 1.0                     # the capture volume is drawn at this height
CONFIGURATIONS = (("2S", (0, 2)), ("2F", (0, 4)))


def seen_by(assets, cameras, points):
    """Boolean per point: inside the image of every camera listed, in front of it."""
    ok = np.ones(len(points), bool)
    for k in cameras:
        T = np.linalg.inv(assets.cameras[k])
        K, _ = assets.intrinsics[k]
        pc = (T[:3, :3] @ points.T + T[:3, 3:4]).T
        uv = (K @ (pc / np.maximum(pc[:, 2:3], 1e-9)).T).T
        ok &= (pc[:, 2] > 0.2) & (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 720)
    return ok


def setup_figure(args):
    """Fig. 4: the workspace seen from above -- force plates, the robot on its
    table, the four cameras with their optical axes and horizontal field of
    view, the floor area all four see at CAPTURE_HEIGHT_M, and the 2S / 2F pairs."""
    import cv2
    from matplotlib.patches import Polygon
    from rtcosmik.viewer.comfi_scene import ComfiScene, TrialAssets
    from scene_render import Renderer
    p, t = SETUP_TRIAL
    assets = TrialAssets.resolve(DATASET, p, t)
    ref = Run(args.output_dir / p / t / REFERENCE_TAG, meta_of(p))
    from scene_render import look_at_cv
    # Seen from above with the long, camera-to-camera axis horizontal (image up = world +x).
    points = [T[:3, 3] for T in assets.cameras.values()] + [assets.robot_base[:3, 3]]
    table = assets.table
    (L, Wd), pose = table["size"], table["pose"]
    points += [pose[:3, :3] @ np.array([sx * L / 2, sy * Wd / 2, 0]) + pose[:3, 3]
               for sx in (-1, 1) for sy in (-1, 1)]
    centre = np.r_[np.mean([p_[:2] for p_ in points], axis=0), 0.0]
    eye = centre + [0.0, 0.0, 8.0]
    view = np.linalg.inv(look_at_cv(eye, centre, up=(1.0, 0.0, 0.0)))
    local = np.array([(view[:3, :3] @ p_ + view[:3, 3])[:2] for p_ in points])
    half = np.abs(local).max(axis=0) + 0.45
    size = (1600, int(1600 * half[1] / half[0]))
    with Renderer(size=size, scale=1) as r:
        scene = ComfiScene(r.vis, assets, show_cameras=True, table_rgba=GREY, robot_rgba=GREY)
        scene.add_body("reference", ref.model, ref.visual_model, rgba=(*COLOURS["reference"], 1.0))
        scene.show(min(assets.robot_joints), {"reference": ref.q[0]})
        r.look_at(eye, centre, up=(1.0, 0.0, 0.0), ortho_half_height=half[1])
        image = r.shot(grid=False)[..., :3]
    W, H = size

    def px(xy):
        xy = np.asarray(xy, float)
        pts = np.column_stack([xy, np.zeros(len(xy))])
        c = (view[:3, :3] @ pts.T + view[:3, 3:4]).T
        return np.column_stack([W / 2 + c[:, 0] / half[0] * W / 2, H / 2 + c[:, 1] / half[1] * H / 2])
    lo = np.min([p_[:2] for p_ in points], axis=0) - 0.6
    hi = np.max([p_[:2] for p_ in points], axis=0) + 0.6

    plt = plot_setup()
    fig, ax = plt.subplots(figsize=(88 * MM, 88 * MM * H / W))
    ax.imshow(image)
    ax.set_axis_off()
    # Capture volume: the floor area every camera sees at CAPTURE_HEIGHT_M.
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, CAPTURE_HEIGHT_M)])
    mask = seen_by(assets, sorted(assets.cameras), grid).reshape(gx.shape).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = max(contours, key=cv2.contourArea)[:, 0, :]
    region_xy = np.column_stack([gx[0, region[:, 0]], gy[region[:, 1], 0]])
    ax.add_patch(Polygon(px(region_xy), closed=True, facecolor=(0.2, 0.2, 0.2, 0.08),
                         edgecolor="0.3", lw=0.8, ls="--"))
    # Cameras: optical axis and horizontal field of view.
    for k, T in sorted(assets.cameras.items()):
        K, _ = assets.intrinsics[k]
        o, axis = T[:3, 3], T[:3, 2]
        half_fov = np.arctan(640.0 / K[0, 0])
        heading = np.arctan2(axis[1], axis[0])
        for sign, style in ((0, "-"), (-1, ":"), (1, ":")):
            a = heading + sign * half_fov
            end = o[:2] + (1.6 if sign == 0 else 1.2) * np.array([np.cos(a), np.sin(a)])
            seg = px([o[:2], end])
            ax.plot(seg[:, 0], seg[:, 1], color="k", lw=0.9 if sign == 0 else 0.6, ls=style)
        label = px([o[:2] - 0.35 * axis[:2] / np.linalg.norm(axis[:2])])[0]
        ax.text(*label, str(k), ha="center", va="center", fontsize=8)
    # Camera configurations.
    for name, (a, b) in CONFIGURATIONS:
        pa, pb = assets.cameras[a][:3, 3][:2], assets.cameras[b][:3, 3][:2]
        seg = px([pa, pb])
        ax.plot(seg[:, 0], seg[:, 1], color="0.35", lw=0.8, ls=(0, (3, 2)))
        at = seg[0] + (0.5 if name == "2S" else 0.3) * (seg[1] - seg[0])    # off the body and axes
        ax.text(at[0], at[1], name, ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    # Scale bar, 1 m, in the upper left corner.
    bar = np.array([[40, 50], [40 + W / half[0] / 2, 50]])
    ax.plot(bar[:, 0], bar[:, 1], color="k", lw=1.2)
    ax.text(bar[:, 0].mean(), bar[0, 1] - 10, "1 m", ha="center", va="bottom", fontsize=8)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out / "setup.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(out / "setup.png", bbox_inches="tight", pad_inches=0.01, dpi=200)
    plt.close(fig)
    with open(out / "setup.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "camera", "x_m", "y_m", "z_m", "axis_x", "axis_y", "axis_z", "hfov_deg"])
        for k, T in sorted(assets.cameras.items()):
            K, _ = assets.intrinsics[k]
            writer.writerow(["camera", k, *np.round(T[:3, 3], 4), *np.round(T[:3, 2], 4),
                             round(float(np.degrees(2 * np.arctan(640.0 / K[0, 0]))), 1)])
        writer.writerow(["robot_base", "", *np.round(assets.robot_base[:3, 3], 4), "", "", "", ""])
        for x, y in region_xy:
            writer.writerow([f"capture_region_at_{CAPTURE_HEIGHT_M:.1f}m", "", round(x, 3), round(y, 3), CAPTURE_HEIGHT_M, "", "", "", ""])
    print(f"  setup: pdf, png, csv -> {out} ({p}/{t})", flush=True)


MARKERSET_PARTICIPANT, MARKERSET_TASK = "1847", "SideOverhead"   # frame 0: standing, calibration pose
FASTSAM_EXPORT = DATASET / "fastsam" / "results_multicam"
FASTSAM_NAMES = {"1847": ("Maxime", "overhead")}      # export folders are named by first name
FROZEN = (("middle_thoracic_Z", "thoracic (3 DoF)"), ("right_wrist_Z", "wrist (2 DoF)"),
          ("left_wrist_Z", "wrist (2 DoF)"))


def upright(vertices, left, right, pelvis_centre):
    """Rotate about z so left-right runs along +x, the body facing -y; feet on z = 0."""
    d = right - left
    a = np.arctan2(d[1], d[0]) - np.pi          # right side to image left for a body facing the viewer
    c, s_ = np.cos(-a), np.sin(-a)
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    v = (R @ (vertices - pelvis_centre).T).T
    v[:, 2] -= v[:, 2].min()
    return v, R


def markerset_figure(args):
    """Fig. 5: the markers the evaluation uses -- (a) on our model, as the mocap
    markers, with the 7 frozen DoFs; (b) the SMPL-X vertices NLF is queried at;
    (c) the MHR vertices of FastSAM-3D -- from the front and from the back."""
    import json
    import cv2
    import meshcat.geometry as g
    import pinocchio as pin
    from rtcosmik.config_loader import settings
    from rtcosmik.viewer.comfi_scene import ComfiScene
    from scene_render import Renderer, look_at_cv
    names = list(settings.marker_names)
    p, t = MARKERSET_PARTICIPANT, MARKERSET_TASK
    ref = Run(args.output_dir / p / t / REFERENCE_TAG, meta_of(p))
    xyz = lambda m, n: m[[f"{n}_x", f"{n}_y", f"{n}_z"]].iloc[0].to_numpy(float)

    # (a) our model at frame 0 with the mocap markers the IK used.
    data = ref.model.createData()
    pin.forwardKinematics(ref.model, data, ref.q[0])
    mocap = {n: xyz(ref.markers, n) for n in names if f"{n}_x" in ref.markers}
    left, right = xyz(ref.markers, "LASI"), xyz(ref.markers, "RASI")
    centre = (left + right + xyz(ref.markers, "LPSI") + xyz(ref.markers, "RPSI")) / 4
    _, R_a = upright(np.array([centre]), left, right, centre)
    floor = min(mocap[n][2] for n in ("RHEE", "LHEE", "RTOE", "LTOE")) - 0.03   # skin markers sit ~3 cm up
    T_a = np.eye(4)
    T_a[:3, :3] = R_a
    T_a[:3, 3] = -R_a @ centre
    T_a[2, 3] = -floor
    place = lambda P: (T_a[:3, :3] @ np.asarray(P).T).T + T_a[:3, 3]
    frozen = [(place([data.oMi[ref.model.getJointId(j)].translation])[0], text) for j, text in FROZEN]

    # (b) SMPL-X template and the vertices NLF is queried at (same order as the markers).
    smplx = np.load(REPO / "weights" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    vs = smplx["v_template"] @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])    # y up, facing +z -> z up, facing -y
    ids = list(settings.nlf_indices)
    idx = {n: i for n, i in zip(names, ids)}
    vs, _ = upright(vs, vs[idx["LASI"]], vs[idx["RASI"]], (vs[idx["LASI"]] + vs[idx["RASI"]]) / 2)
    nlf = {n: vs[i] for n, i in idx.items()}

    # (c) FastSAM-3D's MHR mesh, frame 0 of camera 2 (facing the camera), and its marker map.
    folder, task_dir = FASTSAM_NAMES[p]
    base = FASTSAM_EXPORT / folder / task_dir / "camera_2" / "calibrated"
    vm = np.load(base / "vertices_cam.npy", mmap_mode="r")[0].astype(float)
    faces_m = np.load(base / "faces.npy")
    vm = vm @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])       # camera (x right, y down, z ahead) -> z up
    mmap = json.load(open(FASTSAM_EXPORT / "cosmik_mhr_marker_map_17subjects_tv8_tv12.json"))["markers"]
    fidx = {n: mmap[n]["vertex_index"] for n in names if n in mmap}
    vm, _ = upright(vm, vm[fidx["LASI"]], vm[fidx["RASI"]], (vm[fidx["LASI"]] + vm[fidx["RASI"]]) / 2)
    fast = {n: vm[i] for n, i in fidx.items()}

    panels = []
    size = (700, 1400)
    with Renderer(size=size, scale=1) as r:
        for key in ("mocap", "nlf", "fastsam"):
            r.vis["bodies"].delete()
            r.vis["markers"].delete()
            r.vis["scene"].delete()
            scene = ComfiScene(r.vis, None, show_cameras=False)
            if key == "mocap":
                scene.add_body("model", ref.model, ref.visual_model, rgba=(0.80, 0.80, 0.80, 1.0))
                scene.show(None, {"model": ref.q[0]})
                r.vis["bodies"].set_transform(T_a)
                points, colour = [place([v])[0] for v in mocap.values()], COLOURS["reference"]
            else:
                verts, faces = (vs, smplx["f"]) if key == "nlf" else (vm, faces_m)
                r.vis["bodies"]["mesh"].set_object(g.TriangularMeshGeometry(verts, faces),
                                                   g.MeshLambertMaterial(color=0xcccccc))
                points = list((nlf if key == "nlf" else fast).values())
                colour = COLOURS["nlf_0-2-4-6" if key == "nlf" else "fastsam_0-2-4-6"]
            scene.set_markers("m", np.array(points), rgba=(*colour, 1.0), radius=0.018)
            views = []
            for side in (-1, 1):                      # front (viewer at -y), then back
                eye = np.array([0.0, 6.0 * side, 0.9])
                r.look_at(eye, (0.0, 0.0, 0.9), up=(0, 0, 1), ortho_half_height=1.0)
                views.append(r.shot(grid=False)[..., :3])
            panels.append((key, views, points))

    plt = plot_setup()
    fig, axes = plt.subplots(2, 3, figsize=(88 * MM, 118 * MM), gridspec_kw={"hspace": 0.02, "wspace": 0.0})
    W, H = size
    to_px = lambda P, side: np.column_stack([W / 2 + (-side) * np.asarray(P)[:, 0] / (W / H) * W / 2,
                                             H / 2 - (np.asarray(P)[:, 2] - 0.9) * H / 2])
    captions = {"mocap": "(a) Mocap", "nlf": "(b) NLF", "fastsam": "(c) FastSAM-3D"}
    for c, (key, views, points) in enumerate(panels):
        for r_, image in enumerate(views):
            ax = axes[r_, c]
            ax.imshow(image[int(H * 0.02):int(H * 0.98)], extent=(0, W, H * 0.98, H * 0.02))
            ax.set_axis_off()
        axes[1, c].text(0.5, -0.02, captions[key], transform=axes[1, c].transAxes, ha="center", va="top")
    for (P, text), dy in zip(frozen, (0, 0, 0)):
        u, v = to_px([P], -1)[0]
        axes[0, 0].plot(u, v, "o", ms=7, mfc="none", mec="k", mew=0.8)
    axes[0, 0].text(0.02, 0.995, "\u25cb frozen:\nthoracic (3 DoF)\nwrists (2\u00d72 DoF)",
                    transform=axes[0, 0].transAxes, ha="left", va="top", fontsize=8, linespacing=1.1)
    out = args.out
    fig.savefig(out / "markerset.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(out / "markerset.png", bbox_inches="tight", pad_inches=0.01, dpi=200)
    plt.close(fig)
    with open(out / "markerset.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["marker", "mocap_in_our_model_frame0", "nlf_smplx_vertex", "fastsam_mhr_vertex"])
        for n in names:
            writer.writerow([n, "yes" if n in mocap else "no", idx.get(n, ""), fidx.get(n, "reconstructed" if n == "Head" else "")])
    print(f"  markerset: pdf, png, csv -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "snapshots", "setup", "markerset"))
    ap.add_argument("--output-dir", type=Path, default=REPO / "output" / "campaign")
    ap.add_argument("--results", type=Path, default=REPO / "results" / "campaign")
    ap.add_argument("--events", type=Path, default=None, help="events.csv from reba_events.py")
    ap.add_argument("--out", type=Path, default=None, help="where renderings go")
    ap.add_argument("--participants", nargs="*", default=None)
    args = ap.parse_args()
    args.events = args.events or args.results / "paper" / "reba_events" / "events.csv"
    args.out = args.out or args.results / "paper" / "figures"
    {"check": check, "snapshots": snapshots, "setup": setup_figure, "markerset": markerset_figure}[args.mode](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
