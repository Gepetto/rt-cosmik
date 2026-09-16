#!/usr/bin/env python3
"""Human-robot distances from every arm, against the mocap reference.

On the robot tasks (RobotPolishing, RobotWelding) COMFI records the Franka Panda's
joint positions and the robot base pose in the world frame. In both tasks the
participant hand-guides the robot by the tool on its flange, so the hands are in
contact for most of the trial and a whole-body minimum distance sits at zero.
Four distances are therefore computed for every frame, twice -- from the arm's
estimate and from the mocap reference, with the robot in the same state:

``whole``       minimum distance, whole body to robot (mostly contact)
``body``        minimum distance excluding forearms and hands: head, trunk,
                pelvis, clavicles, upper arms, legs -- what should stay clear
``left_hand_ee``, ``right_hand_ee``
                hand centre (half a hand length past the wrist) to the origin
                of the Panda's hand frame -- where the co-manipulation happens

Robot: example-robot-data's Panda collision meshes, placed by forward
kinematics from COMFI's joint positions (fingers closed; the gripper is not
recorded) and ``robot_in_world/<participant>/robot_base_pose.yaml``. COMFI's
robot rows carry camera 0's own timestamps, so each row is matched to its video
frame exactly; recordings that start late simply cover fewer frames.

Human: a skeleton, not a surface. Each run's model is rebuilt exactly as the IK
built it -- scaled to the first frame of its own markers.csv, which is what
``HumanSolver.calibrate`` does -- and posed with its joint_angles.csv. The body
is the set of segments between joint centres (pelvis, thighs, shanks, lumbar and
thoracic spine, clavicles, upper arms, forearms), plus two extensions the joint
centres do not reach: each hand, 0.108 H along the forearm past the wrist, and
the head, from the cervical joint up to the subject's stature (ankle height
0.039 H; proportions from Winter, Biomechanics and Motor Control of Human
Movement). Distances are to these segments, so they read larger than the
distance to the skin by roughly a limb's radius; the error between an arm and
the reference is what the metric is for, and that offset cancels in it.

Per trial, arm and distance, after the knee-flexion lag alignment used
everywhere else: bias, MAE, RMSE, SD of the distance error, Pearson r, the error
on the trial's closest approach, agreement on whether the distance is below each
threshold in ``THRESHOLDS_M`` (contact: ``CONTACT_M``), and for the two minimum
distances, agreement on which body segment is closest.

    python3 scripts/python/paper/robot_distance.py --arms nlf_0-2-4-6 fastsam_0
"""
import argparse
import csv
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

DATASET = Path("/root/workspace/COMFI")
REFERENCE_TAG = "mocap_reference"
ROBOT_TASKS = ("RobotPolishing", "RobotWelding")
THRESHOLDS_M = (0.10, 0.20, 0.30)
CONTACT_M = 0.01
MEASURES = ("whole", "body", "left_hand_ee", "right_hand_ee")
DISTAL = ("left_forearm", "right_forearm", "left_hand", "right_hand")
HAND_LENGTH_H, ANKLE_HEIGHT_H = 0.108, 0.039

#: (segment, from joint, to joint): joint centres of the scaled model.
SEGMENTS = (
    ("pelvis_left", "root_joint", "left_hip_Z"),
    ("pelvis_right", "root_joint", "right_hip_Z"),
    ("left_thigh", "left_hip_Z", "left_knee_Z"),
    ("left_shank", "left_knee_Z", "left_ankle_Z"),
    ("right_thigh", "right_hip_Z", "right_knee_Z"),
    ("right_shank", "right_knee_Z", "right_ankle_Z"),
    ("lumbar", "middle_lumbar_Z", "middle_thoracic_Z"),
    ("thorax", "middle_thoracic_Z", "middle_cervical_Z"),
    ("left_clavicle", "left_clavicle_joint_X", "left_shoulder_Z"),
    ("right_clavicle", "right_clavicle_joint_X", "right_shoulder_Z"),
    ("left_upperarm", "left_shoulder_Z", "left_elbow_Z"),
    ("left_forearm", "left_elbow_Z", "left_wrist_Z"),
    ("right_upperarm", "right_shoulder_Z", "right_elbow_Z"),
    ("right_forearm", "right_elbow_Z", "right_wrist_Z"),
)
#: (segment, distal joint, proximal joint): extensions past a distal joint.
HANDS = (("left_hand", "left_wrist_X", "left_elbow_Z"),
         ("right_hand", "right_wrist_X", "right_elbow_Z"))
SEGMENT_NAMES = tuple(s[0] for s in SEGMENTS) + tuple(h[0] for h in HANDS) + ("head",)

FIELDS = (["arm", "participant", "task", "measure", "frames", "lag_frames",
           "ref_mean_mm", "ref_min_mm", "arm_mean_mm", "arm_min_mm",
           "bias_mm", "mae_mm", "rmse_mm", "sd_mm", "r", "closest_approach_err_mm",
           "closest_segment_agree_pct", "contact_agree_pct"]
          + [f"below_{int(t * 1000)}mm_agree_pct" for t in THRESHOLDS_M])


def load_eval():
    path = REPO / "scripts" / "python" / "eval" / "compare_to_mocap.py"
    spec = importlib.util.spec_from_file_location("compare_to_mocap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- robot

class Robot:
    """The Panda's collision geometry in the world, for one participant's rig."""

    def __init__(self, participant):
        import example_robot_data as erd
        import yaml
        panda = erd.load("panda")
        self.model, self.geometry = panda.model, panda.collision_model
        self.data = self.model.createData()
        self.geometry_data = self.geometry.createData()
        pose = yaml.safe_load((DATASET / "robot" / "robot_in_world" / participant
                               / "robot_base_pose.yaml").read_text())["world_T_robot"]
        self.world_T_robot = np.asarray(pose["matrix_4x4"], dtype=float)
        self.shapes = [g.geometry for g in self.geometry.geometryObjects]
        for shape in self.shapes:
            shape.computeLocalAABB()
        self.centres = np.array([s.aabb_center for s in self.shapes])
        self.radii = np.array([s.aabb_radius for s in self.shapes])
        self.hand_frame = self.model.getFrameId("panda_hand")

    def place(self, joints):
        """World placements (R, t) of every collision object, and the hand frame's
        origin, for 7 joint positions."""
        import pinocchio as pin
        q = pin.neutral(self.model)
        q[:7] = joints
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.geometry, self.geometry_data)
        pin.updateFramePlacement(self.model, self.data, self.hand_frame)
        R0, t0 = self.world_T_robot[:3, :3], self.world_T_robot[:3, 3]
        placements = [(R0 @ m.rotation, R0 @ m.translation + t0) for m in self.geometry_data.oMg]
        return placements, R0 @ self.data.oMf[self.hand_frame].translation + t0


def robot_frames(participant, task):
    """Video frame index -> 7 joint positions, from COMFI's aligned robot states."""
    import pandas as pd
    path = DATASET / "robot" / "aligned" / participant / f"{participant}_{task}.csv"
    if not path.exists():
        return {}
    robot = pd.read_csv(path)
    if robot.empty:
        return {}
    video = pd.read_csv(DATASET / "videos" / participant / task / "camera_0_timestamps.csv")
    video_ns = pd.to_datetime(video["timestamp"], utc=True).astype("int64").to_numpy()
    robot_ns = pd.to_datetime(robot["_cam_time"], utc=True).astype("int64").to_numpy()
    index = np.clip(np.searchsorted(video_ns, robot_ns), 0, len(video_ns) - 1)
    exact = np.abs(video_ns[index] - robot_ns) < 1_000_000          # 1 ms
    joints = robot[[f"panda_joint{i}_position[rad]" for i in range(1, 8)]].to_numpy(float)
    return {int(video["frame_index"].iloc[i]): joints[k]
            for k, i in enumerate(index) if exact[k] and np.isfinite(joints[k]).all()}


# --------------------------------------------------------------------------- human

class Skeleton:
    """A run's scaled human model, posed frame by frame into world segments."""

    def __init__(self, run_dir, meta):
        import example_robot_data as robex
        import pandas as pd
        import pinocchio as pin
        from rtcosmik.human_model.model_utils import scale_human_model

        markers = pd.read_csv(run_dir / "markers.csv", nrows=1)
        names = sorted({c[:-2] for c in markers.columns if c.endswith("_x")})
        first = {n: markers.loc[0, [f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(float) for n in names}
        gender, height = meta["gender"][0], float(meta["height"])
        model = robex.human.HumanLoader(height=height, weight=meta["weight"], gender=gender).robot.model
        self.model = scale_human_model(model, first, gender=gender, subject_height=height)
        self.data = self.model.createData()
        self.q = pd.read_csv(run_dir / "joint_angles.csv").to_numpy(float)
        self.ids = {name: self.model.getJointId(name) for name in self.model.names}

        # Extensions are fixed in their joint's frame; take them in the neutral pose.
        pin.forwardKinematics(self.model, self.data, pin.neutral(self.model))
        o = lambda j: self.data.oMi[self.ids[j]]
        self.hands = []
        for _, distal, proximal in HANDS:
            direction = o(distal).translation - o(proximal).translation
            direction /= np.linalg.norm(direction)
            self.hands.append((self.ids[distal],
                               o(distal).rotation.T @ direction * HAND_LENGTH_H * height))
        ankle = 0.5 * (o("left_ankle_Z").translation[2] + o("right_ankle_Z").translation[2])
        cervical = o("middle_cervical_Y")
        head = height - (cervical.translation[2] - ankle + ANKLE_HEIGHT_H * height)
        self.head = (self.ids["middle_cervical_Y"], cervical.rotation.T @ np.array([0.0, 0.0, head]))

    def segments(self, row):
        """(n, 2, 3) world endpoints of every segment at one row of joint_angles.csv."""
        import pinocchio as pin
        pin.forwardKinematics(self.model, self.data, self.q[row])
        at = lambda j: self.data.oMi[self.ids[j]].translation
        out = [(at(a), at(b)) for _, a, b in SEGMENTS]
        for joint, local in self.hands + [self.head]:
            m = self.data.oMi[joint]
            out.append((m.translation, m.translation + m.rotation @ local))
        return np.asarray(out)


def min_distance(segments, placements, robot, request, result, capsules, subset):
    """Minimum distance from the segments in ``subset`` to the robot, and the
    closest segment's index.

    Pairs are visited in order of a cheap lower bound (segment to the object's
    bounding sphere), and the search stops once no pair left can beat the best.
    """
    import coal
    import pinocchio as pin
    a, b = segments[subset, 0], segments[subset, 1]
    centres = np.array([R @ c + t for (R, t), c in zip(placements, robot.centres)])
    ab = b - a
    length2 = np.maximum((ab ** 2).sum(1), 1e-12)
    s = np.clip((((centres[None] - a[:, None]) * ab[:, None]).sum(2)) / length2[:, None], 0, 1)
    nearest = a[:, None] + s[..., None] * ab[:, None]
    bound = np.linalg.norm(centres[None] - nearest, axis=2) - robot.radii[None]

    best, best_segment = np.inf, -1
    for flat in np.argsort(bound, axis=None):
        i, k = divmod(int(flat), len(placements))
        if bound[i, k] >= best:
            break
        half = 0.5 * np.sqrt(length2[i])
        if half not in capsules:
            capsules[half] = coal.Capsule(0.0, half)
        axis = ab[i] / (2 * half)
        rotation = pin.Quaternion.FromTwoVectors(np.array([0.0, 0.0, 1.0]), axis).matrix()
        result.clear()
        d = coal.distance(capsules[half], coal.Transform3s(rotation, 0.5 * (a[i] + b[i])),
                          robot.shapes[k], coal.Transform3s(*placements[k]), request, result)
        if d < best:
            best, best_segment = d, int(subset[i])
    return max(best, 0.0), best_segment


def distances(skeleton, robot, rows, joints):
    """Every measure, per (row of the run, robot joints): name -> (distance m, segment)."""
    import coal
    request, result, capsules = coal.DistanceRequest(), coal.DistanceResult(), {}
    body = np.array([i for i, n in enumerate(SEGMENT_NAMES) if n not in DISTAL])
    distal = np.array([i for i, n in enumerate(SEGMENT_NAMES) if n in DISTAL])
    hands = [SEGMENT_NAMES.index("left_hand"), SEGMENT_NAMES.index("right_hand")]
    out = {m: (np.empty(len(rows)), np.full(len(rows), -1)) for m in MEASURES}
    for n, (row, q) in enumerate(zip(rows, joints)):
        segments = skeleton.segments(row)
        placements, flange = robot.place(q)
        args = (segments, placements, robot, request, result, capsules)
        d_body, s_body = min_distance(*args, body)
        d_distal, s_distal = min_distance(*args, distal)
        out["body"][0][n], out["body"][1][n] = d_body, s_body
        out["whole"][0][n], out["whole"][1][n] = ((d_body, s_body) if d_body <= d_distal
                                                  else (d_distal, s_distal))
        for measure, i in zip(("left_hand_ee", "right_hand_ee"), hands):
            centre = segments[i].mean(axis=0)
            out[measure][0][n] = np.linalg.norm(centre - flange)
    return out


# --------------------------------------------------------------------------- trials

def one_trial(job):
    """Every arm of one trial. Returns (rows, per-frame arrays, messages)."""
    import pandas as pd
    import yaml
    runs_root, participant, task, arms, frames_dir = job
    messages, rows = [], []
    by_frame = robot_frames(participant, task)
    if not by_frame:
        return rows, [f"  {participant}/{task}: no robot states"]
    ev = load_eval()
    meta = yaml.safe_load((DATASET / "metadata" / f"{participant}.yaml").read_text())
    robot = Robot(participant)
    ref_dir = runs_root / participant / task / REFERENCE_TAG
    reference = ev.load_run(str(ref_dir))
    ref_frames = pd.read_csv(ref_dir / "markers.csv", usecols=["Frame_0"])["Frame_0"].to_numpy(int)
    ref_skeleton = Skeleton(ref_dir, meta)

    ref_cache = {}
    for arm in arms:
        run_dir = runs_root / participant / task / arm
        try:
            lag, _ = ev.estimate_lag(ev.load_run(str(run_dir)), reference)
            skeleton = Skeleton(run_dir, meta)
            s0, r0 = max(0, lag), max(0, -lag)
            n = min(len(skeleton.q) - s0, len(ref_skeleton.q) - r0)
            pairs = [(s0 + k, r0 + k) for k in range(n) if int(ref_frames[r0 + k]) in by_frame]
            if not pairs:
                raise ValueError("no frame with robot states")
            arm_rows, ref_rows = map(list, zip(*pairs))
            joints = [by_frame[int(ref_frames[r])] for r in ref_rows]
            key = tuple(ref_rows)
            if key not in ref_cache:
                ref_cache[key] = distances(ref_skeleton, robot, ref_rows, joints)
            ref = ref_cache[key]
            est = distances(skeleton, robot, arm_rows, joints)
        except Exception as exc:
            messages.append(f"  {arm} {participant}/{task}: {type(exc).__name__}: {exc}")
            continue

        arrays = {"ref_frame": np.asarray(ref_rows), "arm_row": np.asarray(arm_rows),
                  "segment_names": np.asarray(SEGMENT_NAMES)}
        for measure in MEASURES:
            (d_ref, seg_ref), (d_arm, seg_arm) = ref[measure], est[measure]
            e = d_arm - d_ref
            segments = measure in ("whole", "body")
            row = {"arm": arm, "participant": participant, "task": task, "measure": measure,
                   "frames": len(e), "lag_frames": int(lag),
                   "ref_mean_mm": 1000 * d_ref.mean(), "ref_min_mm": 1000 * d_ref.min(),
                   "arm_mean_mm": 1000 * d_arm.mean(), "arm_min_mm": 1000 * d_arm.min(),
                   "bias_mm": 1000 * e.mean(), "mae_mm": 1000 * np.abs(e).mean(),
                   "rmse_mm": 1000 * np.sqrt((e ** 2).mean()), "sd_mm": 1000 * e.std(),
                   "r": (float(np.corrcoef(d_arm, d_ref)[0, 1])
                         if d_ref.std() > 0 and d_arm.std() > 0 else np.nan),
                   "closest_approach_err_mm": 1000 * (d_arm.min() - d_ref.min()),
                   "closest_segment_agree_pct": (100 * np.mean(seg_arm == seg_ref)
                                                 if segments else np.nan),
                   "contact_agree_pct": 100 * np.mean((d_arm < CONTACT_M) == (d_ref < CONTACT_M))}
            row.update({f"below_{int(t * 1000)}mm_agree_pct":
                        100 * np.mean((d_arm < t) == (d_ref < t)) for t in THRESHOLDS_M})
            rows.append(row)
            arrays.update({f"{measure}_ref": d_ref, f"{measure}_arm": d_arm})
            if segments:
                arrays.update({f"{measure}_segment_ref": seg_ref, f"{measure}_segment_arm": seg_arm})
        target = frames_dir / arm
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / f"{participant}_{task}.npz", **arrays)
    return rows, messages


def trials_for(results, arm):
    path = results / "vs_mocap" / f"{arm}.csv"
    if not path.exists():
        return set()
    return {(r["participant"], r["task"]) for r in csv.DictReader(open(path))
            if r.get("status") == "ok" and r["task"] in ROBOT_TASKS}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="where run folders live; defaults to settings.output_dir")
    ap.add_argument("--results", type=Path, default=REPO / "results",
                    help="folder whose vs_mocap/ lists the trials each arm completed")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "paper")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    from rtcosmik.config_loader import settings
    runs_root = args.output_dir or Path(settings.output_dir)
    wanted = {}
    for arm in args.arms:
        for trial in trials_for(args.results, arm):
            wanted.setdefault(trial, []).append(arm)
    jobs = [(runs_root, p, t, arms, args.out / "robot_distance" / "frames")
            for (p, t), arms in sorted(wanted.items())]

    results = {arm: [] for arm in args.arms}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows, messages in pool.map(one_trial, jobs):
            for m in messages:
                print(m, flush=True)
            for row in rows:
                results[row["arm"]].append(row)

    out = args.out / "robot_distance"
    out.mkdir(parents=True, exist_ok=True)
    for arm, rows in results.items():
        with open(out / f"{arm}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["participant"], r["task"],
                                                         MEASURES.index(r["measure"]))))
        trials = len({(r["participant"], r["task"]) for r in rows})
        print(f"{arm}: {trials} trials -> {out / f'{arm}.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
