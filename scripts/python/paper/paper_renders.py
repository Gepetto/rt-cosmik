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
CAPTURE_HEIGHT_M = 1.0                     # the capture region is taken at this height ...
CAPTURE_TOP_M = 2.0                        # ... and drawn as a volume from the floor to this height
CONFIGURATIONS = (("2S", (0, 2)), ("2F", (0, 4)))
#: The inclined view of the setup: from above the participant's front-left, so
#: the body is seen in profile at the robot's table, the force plates in front.
SETUP_EYE, SETUP_TARGET, SETUP_FOV_DEG = (-5.5, -4.2, 4.4), (0.0, -0.3, 0.45), 36.0
AXIS_RGB = ((0.85, 0.12, 0.12), (0.10, 0.65, 0.15), (0.12, 0.25, 0.90))    # x, y, z
FRUSTUM_DEPTH_M = 1.0


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


def draw_triad(node, T, length=0.3, radius=0.012):
    """An xyz frame (red, green, blue cylinders) at the 4x4 pose ``T``."""
    import meshcat.geometry as g
    from rtcosmik.viewer.comfi_scene import _material
    turn = {0: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),     # meshcat cylinders run along y
            1: np.eye(3), 2: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])}
    for axis in range(3):
        local = np.eye(4)
        local[:3, :3] = turn[axis]
        local[axis, 3] = length / 2
        node[f"axis_{axis}"].set_object(g.Cylinder(length, radius), _material((*AXIS_RGB[axis], 1.0)))
        node[f"axis_{axis}"].set_transform(local)
    node.set_transform(T)


def draw_frustum(scene, path, T, K, depth=FRUSTUM_DEPTH_M, rgba=(0.2, 0.2, 0.2, 0.9)):
    """The field of view of a camera as a wire pyramid ``depth`` deep."""
    corners = [np.linalg.inv(K) @ np.array([u, v, 1.0]) * depth for u, v in ((0, 0), (1280, 0), (1280, 720), (0, 720))]
    world = [T[:3, :3] @ c + T[:3, 3] for c in corners]
    for i, c in enumerate(world):
        scene._bar(f"{path}/ray_{i}", T[:3, 3], c, 0.008, rgba)
        scene._bar(f"{path}/edge_{i}", c, world[(i + 1) % 4], 0.008, rgba)


def draw_prism(node, polygon_xy, bottom, top, rgba=(0.45, 0.70, 0.95, 0.14)):
    """A translucent vertical prism over a convex floor polygon."""
    import meshcat.geometry as g
    n = len(polygon_xy)
    ring = np.asarray(polygon_xy, float)
    verts = np.vstack([np.column_stack([ring, np.full(n, bottom)]), np.column_stack([ring, np.full(n, top)]),
                       [[*ring.mean(axis=0), bottom], [*ring.mean(axis=0), top]]])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i],             # side
                  [2 * n, j, i], [2 * n + 1, n + i, n + j]]      # bottom and top fans
    colour = int(rgba[0] * 255) * 65536 + int(rgba[1] * 255) * 256 + int(rgba[2] * 255)
    node.set_object(g.TriangularMeshGeometry(verts, np.array(faces)),
                    g.MeshLambertMaterial(color=colour, opacity=rgba[3], transparent=True, side=2))


def setup_figure(args):
    """Fig. 4: the workspace in the COMFI scene, seen from above at an angle --
    force plates, the robot on its table, a participant at work (in the model's
    own colours), the four cameras with their frames and fields of view, the
    capture volume every camera sees, the 2S and 2F pairs, and xyz frames."""
    import cv2
    from rtcosmik.viewer.comfi_scene import ComfiScene, TrialAssets
    from scene_render import Renderer
    p, t = SETUP_TRIAL
    assets = TrialAssets.resolve(DATASET, p, t)
    ref = Run(args.output_dir / p / t / REFERENCE_TAG, meta_of(p))
    event = [e for e in read(args.events) if e["participant"] == p and e["task"] == t]
    frame = int(event[0]["ref_frame"]) if event else len(ref.q) // 2

    # Capture region: the convex floor polygon every camera sees at CAPTURE_HEIGHT_M.
    gx, gy = np.meshgrid(np.linspace(-4, 4, 400), np.linspace(-4, 4, 400))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, CAPTURE_HEIGHT_M)])
    mask = seen_by(assets, sorted(assets.cameras), grid).reshape(gx.shape).astype(np.uint8)
    contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    hull = cv2.approxPolyDP(cv2.convexHull(contour), 2.0, True)[:, 0, :]
    region = np.column_stack([gx[0, hull[:, 0]], gy[hull[:, 1], 0]])

    size = (1600, 1000)
    with Renderer(size=size, scale=1) as r:
        scene = ComfiScene(r.vis, assets, show_cameras=True)
        scene.add_body("reference", ref.model, ref.visual_model)            # the model's own colours
        scene.show(frame + VIDEO_OFFSET, {"reference": ref.q[frame]})
        extras = r.vis["scene"]["extras"]
        draw_triad(extras["world"], np.eye(4), length=0.5, radius=0.015)
        draw_triad(extras["robot"], assets.robot_base, length=0.35)
        for k, T in assets.cameras.items():
            draw_triad(extras[f"camera_{k}"], T, length=0.3)
            draw_frustum(scene, f"scene/extras/frustum_{k}", T, assets.intrinsics[k][0])
        draw_prism(extras["capture"], region, 0.005, CAPTURE_TOP_M)
        r.look_at(SETUP_EYE, SETUP_TARGET, fov_deg=SETUP_FOV_DEG)
        image = r.shot()[..., :3]
        at = {k: r.project(T[:3, 3])[0] for k, T in assets.cameras.items()}
        robot_px = r.project(assets.robot_base[:3, 3] + [0, 0, 0.55])[0]
        capture_px = r.project([*region[np.argmin(region[:, 0])], CAPTURE_TOP_M])[0]
        big = max(assets.force_plates, key=lambda plate: plate[0][0] * plate[0][1])
        plates_px = r.project([big[1][0] - big[0][0] / 2, big[1][1] - big[0][1] / 2, 0.0])[0]

    W, H = size
    plt = plot_setup()
    fig, ax = plt.subplots(figsize=(88 * MM, 88 * MM * H / W))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    # Camera ids pushed away from the middle of their support, so a pair seen
    # end-on keeps its two labels apart.
    supports = {0: (0, 2), 2: (0, 2), 4: (4, 6), 6: (4, 6)}
    for k, (u, v) in at.items():
        centre = (at[supports[k][0]] + at[supports[k][1]]) / 2
        away = np.array([u, v]) - centre
        away = away / max(np.linalg.norm(away), 1e-9)
        ax.text(u + 45 * away[0], v - 30 + 25 * away[1], str(k), ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))
    for name, (a, b) in CONFIGURATIONS:
        seg = np.array([at[a], at[b]])
        if name == "2F":
            ax.plot(seg[:, 0], seg[:, 1], color="0.2", lw=0.8, ls=(0, (3, 2)))
            mid = seg[0] + 0.3 * (seg[1] - seg[0])
        else:
            mid = seg.mean(axis=0) + [0, 55]                       # just below the support
        ax.text(mid[0], mid[1], name, ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.text(robot_px[0] + 40, robot_px[1], "robot", ha="left", va="center", fontsize=8)
    ax.text(*capture_px, "capture volume", ha="right", va="bottom", fontsize=8)
    ax.text(plates_px[0], plates_px[1] + 12, "force plates", ha="center", va="top", fontsize=8)
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
        for (sx, sy), (cx, cy) in assets.force_plates:
            writer.writerow(["force_plate", "", cx, cy, 0.0, "", "", "", f"{sx} x {sy} m"])
        for x, y in region:
            writer.writerow([f"capture_region_at_{CAPTURE_HEIGHT_M:.1f}m", "", round(x, 3), round(y, 3),
                             CAPTURE_HEIGHT_M, "", "", "", ""])
    print(f"  setup: pdf, png, csv -> {out} ({p}/{t}, reference frame {frame})", flush=True)


#: Panel (a): the marker figure of the COMFI paper, which names every marker
#: (anatomical in red, technical in green). Used as published, without a list:
#: which of them the IK uses is in markerset.csv.
COMFI_MARKER_FIGURE = Path("/root/workspace/Figure_4_markers_comfi.pdf")
#: Panel (c): a FastSAM export whose mesh is also published, at its first frame
#: (the participant stands in the calibration pose).
FASTSAM_EXPORT = DATASET / "fastsam" / "results_multicam"
MARKERSET_FASTSAM = ("2198", "SideOverhead", "Batiste", "overhead", 2)   # id, task, folder, task dir, camera
ARMS_DOWN_DEG = 70.0      # SMPL-X template: shoulders lowered from the T pose
#: Where each marker of the evaluation comes from in the mocap reference
#: (``mocap_reference.py``): COMFI's head cluster stands in for the head, and
#: the three facial landmarks have no mocap counterpart.
MOCAP_SOURCE = {"REar": "RHD", "LEar": "LHD",
                "Head": "centroid of FHD, BHD, LHD, RHD",
                "Nose": "not in mocap", "REye": "not in mocap", "LEye": "not in mocap"}


def comfi_marker_figure():
    """The COMFI paper's marker figure, rendered and trimmed of its margins."""
    import subprocess
    import tempfile
    import cv2
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "500", "-png", "-singlefile", str(COMFI_MARKER_FIGURE),
                        f"{tmp}/page"], check=True)
        page = cv2.cvtColor(cv2.imread(f"{tmp}/page.png"), cv2.COLOR_BGR2RGB)
    ink = page.mean(axis=2) < 245
    ys, xs = np.nonzero(ink)
    return page[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def smplx_arms_down(path, angle_deg=ARMS_DOWN_DEG):
    """SMPL-X template vertices (y up, facing +z) with the arms lowered along the
    body: linear blend skinning of the model's own joints, weights and pose
    corrective blend shapes, shoulders rotated by ``angle_deg``."""
    d = np.load(path, allow_pickle=True)
    v = d["v_template"].astype(float)
    J = d["J_regressor"] @ v
    parents = d["kintree_table"][0].astype(np.int64)
    n = len(J)
    rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    R = np.tile(np.eye(3), (n, 1, 1))
    R[16], R[17] = rz(-np.radians(angle_deg)), rz(np.radians(angle_deg))    # left, right shoulder
    v = v + np.einsum("vcp,p->vc", d["posedirs"], (R[1:] - np.eye(3)).reshape(-1))
    G = np.zeros((n, 4, 4))
    for j in range(n):
        T = np.eye(4)
        T[:3, :3] = R[j]
        T[:3, 3] = J[j] - (J[parents[j]] if j else 0.0)
        G[j] = T if j == 0 else G[parents[j]] @ T
    G[:, :3, 3] -= np.einsum("jab,jb->ja", G[:, :3, :3], J)
    T_v = np.einsum("vj,jab->vab", d["weights"], G)
    return np.einsum("vab,vb->va", T_v[:, :3, :3], v) + T_v[:, :3, 3], d["f"]


def upright_transform(left, right, centre, floor_of):
    """A transform putting a body upright and facing -y: left-right along +x,
    the pelvis centred, the lowest of ``floor_of`` on z = 0."""
    d = np.asarray(right) - np.asarray(left)
    a = np.arctan2(d[1], d[0]) - np.pi          # the body's right lands on image left
    c, s_ = np.cos(-a), np.sin(-a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]
    T[:3, 3] = -T[:3, :3] @ np.asarray(centre)
    T[2, 3] -= ((T[:3, :3] @ np.asarray(floor_of).T).T + T[:3, 3])[:, 2].min()
    return T


def apply(T, points):
    return (T[:3, :3] @ np.asarray(points, float).reshape(-1, 3).T).T + T[:3, 3]


def fastsam_markers(names):
    """The 35 markers FastSAM-3D feeds the IK, and its body mesh, in one frame.

    Read with the pipeline's own reader (``fastsam_source``), so the points are
    exactly the ones the IK consumes: the exported markers, minus the two extra
    thoracic ones, with Head placed from the facial landmarks.
    """
    import importlib
    import sys as _sys
    _sys.path.insert(0, str(REPO / "src"))
    fs = importlib.import_module("rtcosmik.paper.fastsam_source")
    participant, task, folder, task_dir, camera = MARKERSET_FASTSAM
    export = DATASET / "fastsam" / participant / task / fs.FASTSAM_FILE.format(camera=camera)
    exported, xyz, valid = fs.load_fastsam_markers(export)
    frame = int(np.flatnonzero(valid)[0])
    markers = {name: xyz[frame, i] for i, name in enumerate(exported) if name not in fs.DROPPED_MARKERS}
    markers["Head"] = fs.derive_head(markers)
    missing = set(names) - set(markers)
    if missing:
        raise ValueError(f"FastSAM export is missing {sorted(missing)}")
    base = FASTSAM_EXPORT / folder / task_dir / f"camera_{camera}" / "calibrated"
    mesh = np.load(base / "vertices_cam.npy", mmap_mode="r")[frame].astype(float)
    return markers, mesh, np.load(base / "faces.npy")


def markerset_figure(args):
    """Fig. 5: the markers of the evaluation. (a) the motion capture markers, as
    the COMFI paper shows them; (b) the SMPL-X vertices NLF is queried at;
    (c) the points FastSAM-3D feeds the IK, on its own body mesh. (b) and (c)
    carry the same 35 markers as the parity set, front and back."""
    import json
    import meshcat.geometry as g
    from rtcosmik.config_loader import settings
    from rtcosmik.viewer.comfi_scene import ComfiScene
    from scene_render import Renderer
    names = list(settings.marker_names)

    # (b) SMPL-X, arms along the body, at the vertices NLF is queried at.
    vs, faces_s = smplx_arms_down(REPO / "weights" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz")
    vs = vs @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])        # y up, facing +z -> z up, facing -y
    nlf_vertex = dict(zip(names, settings.nlf_indices))
    T = upright_transform(vs[nlf_vertex["LASI"]], vs[nlf_vertex["RASI"]],
                          (vs[nlf_vertex["LASI"]] + vs[nlf_vertex["RASI"]]) / 2, vs)
    vs = apply(T, vs)
    nlf_points = np.array([vs[nlf_vertex[n]] for n in names])

    # (c) FastSAM-3D: the points its source hands the IK, on the published mesh.
    fast, vm, faces_m = fastsam_markers(names)
    vm = vm @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])        # camera (x right, y down, z ahead) -> z up
    fast = {n: p @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) for n, p in fast.items()}
    T = upright_transform(fast["LASI"], fast["RASI"], (fast["LASI"] + fast["RASI"]) / 2, vm)
    vm = apply(T, vm)
    fast_points = apply(T, [fast[n] for n in names])
    mhr_vertex = json.load(open(FASTSAM_EXPORT / "cosmik_mhr_marker_map_17subjects_tv8_tv12.json"))

    size = (560, 1400)
    renders = {}
    with Renderer(size=size, scale=1) as r:
        for key, verts, faces, points, colour in (
                ("nlf", vs, faces_s, nlf_points, COLOURS["nlf_0-2-4-6"]),
                ("fastsam", vm, faces_m, fast_points, COLOURS["fastsam_0-2-4-6"])):
            r.vis["bodies"].delete()
            r.vis["markers"].delete()
            scene = ComfiScene(r.vis, None, show_cameras=False)
            r.vis["bodies"]["mesh"].set_object(g.TriangularMeshGeometry(verts, faces),
                                               g.MeshLambertMaterial(color=0xd9d9d9))
            scene.set_markers("m", points, rgba=(*colour, 1.0), radius=0.02)
            renders[key] = []
            for side in (-1, 1):                              # front (viewer at -y), then back
                r.look_at((0.0, 6.0 * side, 0.9), (0.0, 0.0, 0.9), up=(0, 0, 1), ortho_half_height=1.0)
                image = r.shot(grid=False)[..., :3]
                renders[key].append(image[int(0.03 * size[1]):int(0.97 * size[1])])

    comfi = comfi_marker_figure()
    plt = plot_setup()
    # One column, two rows, laid out in millimetres: the COMFI figure on top,
    # the two templates below, front and back.
    W_MM, GAP_MM, CAPTION_MM = 88.0, 3.0, 5.0
    ratio = lambda im: im.shape[1] / im.shape[0]
    h1 = W_MM / ratio(comfi)
    views = [*renders["nlf"], *renders["fastsam"]]
    h2 = (W_MM - GAP_MM) / sum(ratio(v) for v in views)
    H_MM = h1 + CAPTION_MM + h2 + CAPTION_MM
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    box = lambda x, y, w, h: fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])

    def image_at(x, y, image, height):
        ax = box(x, y, ratio(image) * height, height)
        ax.imshow(image)
        ax.set_axis_off()
        return x + ratio(image) * height

    image_at(0.0, 0.0, comfi, h1)
    fig.text(0.5, 1 - (h1 + 0.6) / H_MM, "(a) Motion capture markers (COMFI)", ha="center", va="top", fontsize=8)
    y2 = h1 + CAPTION_MM
    x = image_at(0.0, y2, views[0], h2)
    x = image_at(x, y2, views[1], h2)
    split = x + GAP_MM / 2
    x = image_at(x + GAP_MM, y2, views[2], h2)
    image_at(x, y2, views[3], h2)
    for centre, text in ((split / 2, "(b) NLF (SMPL-X)"), ((split + W_MM) / 2, "(c) FastSAM-3D (MHR)")):
        fig.text(centre / W_MM, 1 - (y2 + h2 + 0.6) / H_MM, text, ha="center", va="top", fontsize=8)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "markerset.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(out / "markerset.png", bbox_inches="tight", pad_inches=0.01, dpi=200)
    plt.close(fig)
    markers_map = mhr_vertex["markers"]
    faces_map = mhr_vertex["face_keypoints"]
    with open(out / "markerset.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "marker", "mocap reference (COMFI marker)", "NLF: SMPL-X vertex",
                         "FastSAM-3D: MHR source"])
        for i, n in enumerate(names, start=1):
            if n in markers_map:
                source = f"vertex {markers_map[n]['vertex_index']}"
            elif n in faces_map:
                source = f"mhr70 keypoint {faces_map[n]['keypoint_index']}"
            else:
                source = "placed from Nose, REar, LEar"
            writer.writerow([i, n, MOCAP_SOURCE.get(n, n), nlf_vertex[n], source])
    print(f"  markerset: pdf, png, csv -> {out} ({len(names)} markers)", flush=True)



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
