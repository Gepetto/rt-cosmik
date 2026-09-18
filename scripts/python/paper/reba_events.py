#!/usr/bin/env python3
"""Posture REBA at the demanding postures of each task, per trial and arm.

``reba_agreement.py`` compares the REBA of every frame of a trial. Averaged over
a whole trial, errors of opposite sign cancel, and the risk that matters -- the
one at the demanding postures -- is diluted by long stretches of standing. This
script scores those postures only, two ways, always with frames chosen on the
**reference** kinematics (the mocap-driven run), so every arm is scored at the
same instants after the usual knee-flexion lag alignment (reference frame ``f``
is arm row ``f + lag``):

**Event.** One representative frame per trial, from the reference markers
(metres, world z up; a hand is the midpoint of its two wrist markers, the pelvis
the centroid of its four markers):

``Screwing``, ``Polishing``
    maximum trunk flexion: the inclination of the pelvis-to-C7 axis that REBA
    scores, relative to the trial's neutral.
``SideOverhead``
    maximum hand height, either hand.
``RobotPolishing``, ``RobotWelding``
    farthest reach: maximum horizontal distance from the pelvis to the hand
    holding the robot's tool. That hand is the one whose median distance to the
    Panda's hand frame over the trial is smaller, from ``robot_distance.py``'s
    per-frame reference distances; for the three trials without robot states it
    is taken from the same participant's other robot task.
``Lifting``
    first instant the load is off the ground. The toolbox starts on the floor
    and is carried with both hands, so the hand height is the mean of the two.
    The first visit to floor level (below the trial's lowest hand height + 0.10
    m) is the first grasp; its lowest frame is the grasp, and the event is the
    first frame after it where the hands have risen ``LIFT_RISE_M`` above it.
    The brief's suggested rule (hands back above their median height) is
    recorded alongside as ``alt_frame``: it fires once the participant is
    almost upright, not at lift-off.

**Peak (top decile).** No event definition: the ``k = ceil(0.1 n)`` frames with
the highest reference REBA. REBA is an integer, so many frames usually tie at
the threshold score -- keeping them all selected up to 40 % of a trial. Instead
the frames above the threshold count fully and the tied ones share the remaining
weight equally, so the selection is exactly ``k`` frames' worth; every mean over
it (reference, estimate, absolute error, risk agreement) is then the average
over all the ways of breaking the ties, without choosing one.

Both use the trial's own start as neutral for each series (``own``, what a
deployed system does, and what the paper's Table II reports) and the
reference's neutral for the arm as well (``reference``), as in
``reba_agreement.py``.

Writes, under ``--out``:

    reba_events/events.csv      one row per trial: the rule, the selected
                                reference frame, the quantity that selected it,
                                the reference REBA there, the top-decile set
    reba_events/<arm>.csv       one row per trial and neutral mode: event score
                                of reference and arm, error, risk agreement;
                                top-decile means, MAE, risk agreement
    tables/reba_events.*        mean (SD) across participants per task
    tables/reba_peak.*          the same for the top decile
    tables/stats_reba_*.*       planned comparisons on the all-task MAE

    python3 scripts/python/paper/reba_events.py --output-dir output/campaign \\
        --results results/campaign --out results/campaign/paper
"""
import argparse
import csv
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

REFERENCE_TAG = "mocap_reference"
TASKS = ("Screwing", "Polishing", "SideOverhead", "RobotPolishing", "RobotWelding", "Lifting")
TASK_LABELS = {"Screwing": "Screwing", "Polishing": "Polishing",
               "SideOverhead": "Overhead manipulation", "RobotPolishing": "Robot polishing",
               "RobotWelding": "Robot welding", "Lifting": "Lifting"}
#: Arms in the order of the paper's Table I, with its names.
ARMS = ("mmpose_0-2-4-6", "mmpose_0-2", "mmpose_0-4",
        "nlf2d_0-2-4-6", "nlf2d_0-2", "nlf2d_0-4",
        "nlf_0-2-4-6", "nlf_0-2", "nlf_0-4", "nlf_0",
        "fastsam_0-2-4-6", "fastsam_0-2", "fastsam_0-4", "fastsam_0")
FRONT_END = {"mmpose": "RTMPose+LSTM", "nlf2d": "NLF-2D", "nlf": "NLF-3D", "fastsam": "FastSAM-3D"}
CAMERAS = {"0-2-4-6": "4", "0-2": "2S", "0-4": "2F", "0": "1"}
COMPONENTS = ("neck", "trunk", "legs", "upper_arm", "lower_arm")
MODES = ("own", "reference")

FLOOR_BAND_M = 0.10     # floor level: within this of the trial's lowest hand height
LIFT_RISE_M = 0.05      # hands this far above the grasp: the load is off the ground
PEAK_SHARE = 0.10
ROBOT_FRAMES_ARM = "nlf_0-2-4-6"   # any arm: the reference distances are the same in all

EVENT_FIELDS = ["participant", "task", "rule", "ref_frame", "time_s", "quantity", "value", "unit",
                "grasp_frame", "grasp_height_m", "alt_frame", "tool_hand", "tool_hand_source",
                "reba_ref", "risk_ref"] + [f"{c}_ref" for c in COMPONENTS] + \
               ["peak_threshold", "peak_frames", "peak_above", "peak_tied", "peak_pct", "trial_frames"]
ARM_FIELDS = (["arm", "front_end", "cameras", "participant", "task", "neutral", "lag_frames",
               "ref_frame", "arm_frame", "reba_ref", "reba_arm", "reba_err", "reba_abs_err",
               "risk_ref", "risk_arm", "risk_agree"]
              + [f"{c}_{s}" for c in COMPONENTS for s in ("ref", "arm")]
              + ["peak_frames", "peak_ref_mean", "peak_arm_mean", "peak_bias", "peak_mae",
                 "peak_risk_agree_pct"])


def label(arm):
    family, cams = arm.split("_", 1)
    return FRONT_END[family], CAMERAS[cams]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def xyz(markers, name):
    return np.column_stack([markers[f"{name}_{a}"] for a in "xyz"]).astype(float)


def hand(markers, side):
    return 0.5 * (xyz(markers, f"{side}WRI") + xyz(markers, f"{side}MWRI"))


def pelvis(markers):
    return np.mean([xyz(markers, n) for n in ("RASI", "LASI", "RPSI", "LPSI")], axis=0)


# --------------------------------------------------------------------------- events

def tool_hand(frames_dir, participant, task):
    """'R' or 'L' and where the choice came from; None if nothing is known."""
    def from_trial(t):
        path = frames_dir / f"{participant}_{t}.npz"
        if not path.exists():
            return None
        d = np.load(path)
        return "R" if np.median(d["right_hand_ee_ref"]) <= np.median(d["left_hand_ee_ref"]) else "L"
    side = from_trial(task)
    if side:
        return side, "closer to end effector"
    other = "RobotWelding" if task == "RobotPolishing" else "RobotPolishing"
    side = from_trial(other)
    return (side, f"no robot states; from {other}") if side else (None, "no robot states")


def select_event(task, joints, markers, neutral, frames_dir, participant):
    """The representative frame of one reference trial, and how it was chosen."""
    from rtcosmik.ergonomics import reba_posture as rp
    out = {"grasp_frame": "", "grasp_height_m": "", "alt_frame": "", "tool_hand": "",
           "tool_hand_source": ""}
    if task in ("Screwing", "Polishing"):
        trunk = rp.raw_inputs(joints, markers)["trunk_incl"] - neutral["trunk_incl"]
        f = int(np.nanargmax(trunk))
        out.update(rule="max trunk flexion", quantity="trunk inclination from neutral",
                   value=float(trunk[f]), unit="deg")
    elif task == "SideOverhead":
        height = np.fmax(hand(markers, "R")[:, 2], hand(markers, "L")[:, 2])
        f = int(np.nanargmax(height))
        out.update(rule="max hand height", quantity="highest hand height", value=float(height[f]),
                   unit="m")
    elif task in ("RobotPolishing", "RobotWelding"):
        side, source = tool_hand(frames_dir, participant, task)
        if side is None:
            raise ValueError("tool hand unknown")
        reach = np.linalg.norm((hand(markers, side) - pelvis(markers))[:, :2], axis=1)
        f = int(np.nanargmax(reach))
        out.update(rule="max reach", quantity=f"horizontal pelvis to {side} hand distance",
                   value=float(reach[f]), unit="m", tool_hand=side, tool_hand_source=source)
    elif task == "Lifting":
        height = 0.5 * (hand(markers, "R")[:, 2] + hand(markers, "L")[:, 2])
        low = np.flatnonzero(height < np.nanmin(height) + FLOOR_BAND_M)
        breaks = np.flatnonzero(np.diff(low) > 1)
        visit = low[:breaks[0] + 1] if breaks.size else low      # first visit to floor level
        grasp = int(visit[np.nanargmin(height[visit])])
        after = np.flatnonzero(height[grasp:] >= height[grasp] + LIFT_RISE_M)
        if not after.size:
            raise ValueError("hands never rise after the first grasp")
        f = grasp + int(after[0])
        median = np.nanmedian(height)
        alt = np.flatnonzero(height[int(np.nanargmin(height)):] > median)
        out.update(rule="lift-off", quantity="hand height (mean of both)", value=float(height[f]),
                   unit="m", grasp_frame=grasp, grasp_height_m=float(height[grasp]),
                   alt_frame=int(np.nanargmin(height)) + int(alt[0]) if alt.size else "")
    else:
        raise ValueError(f"no event rule for {task}")
    out["ref_frame"] = f
    return out


def peak_set(reba):
    """Top-decile reference frames and their weights (see the module docstring).

    Returns (frames, weights, threshold, frames above it, frames tied at it).
    """
    valid = np.flatnonzero(np.isfinite(reba))
    k = max(1, math.ceil(PEAK_SHARE * valid.size))
    threshold = float(np.sort(reba[valid])[::-1][k - 1])
    above, tied = valid[reba[valid] > threshold], valid[reba[valid] == threshold]
    frames = np.r_[above, tied]
    weights = np.r_[np.ones(above.size), np.full(tied.size, (k - above.size) / tied.size)]
    return frames, weights, threshold, above.size, tied.size


# --------------------------------------------------------------------------- scoring

def score_arm(ev, run_dir, ref_dir, event_frame, peak, ref_joints, ref_markers, ref_neutral):
    """Rows for both neutral modes of one arm on one trial."""
    from rtcosmik.ergonomics import reba_posture as rp
    aj, am = pd.read_csv(run_dir / "joint_angles.csv"), pd.read_csv(run_dir / "markers.csv")
    own = rp.neutral_from(aj.iloc[rp.NEUTRAL_FRAMES], am.iloc[rp.NEUTRAL_FRAMES])
    lag, _ = ev.estimate_lag(ev.load_run(str(run_dir)), ev.load_run(str(ref_dir)))
    sr = rp.scores(ref_joints, ref_markers, ref_neutral)
    rows = []
    for mode, neutral in (("own", own), ("reference", ref_neutral)):
        sa = rp.scores(aj, am, neutral)
        row = {"neutral": mode, "lag_frames": int(lag), "ref_frame": event_frame}
        a = event_frame + lag
        if 0 <= a < len(aj) and sa["valid"][a] and sr["valid"][event_frame]:
            ra, rr = float(sa["reba"][a]), float(sr["reba"][event_frame])
            row.update(arm_frame=a, reba_ref=rr, reba_arm=ra, reba_err=ra - rr,
                       reba_abs_err=abs(ra - rr), risk_ref=int(sr["risk"][event_frame]),
                       risk_arm=int(sa["risk"][a]),
                       risk_agree=int(sa["risk"][a] == sr["risk"][event_frame]))
            for c in COMPONENTS:
                row[f"{c}_ref"], row[f"{c}_arm"] = int(sr[c][event_frame]), int(sa[c][a])
        frames, weights = peak
        rows_arm = frames + lag
        keep = (rows_arm >= 0) & (rows_arm < len(aj))
        ref_f, arm_f, w = frames[keep], rows_arm[keep], weights[keep]
        ok = sa["valid"][arm_f] & sr["valid"][ref_f]
        ref_f, arm_f, w = ref_f[ok], arm_f[ok], w[ok]
        if ref_f.size:
            ea, er = sa["reba"][arm_f], sr["reba"][ref_f]
            mean = lambda x: float(np.average(x, weights=w))
            row.update(peak_frames=float(w.sum()), peak_ref_mean=mean(er), peak_arm_mean=mean(ea),
                       peak_bias=mean(ea - er), peak_mae=mean(np.abs(ea - er)),
                       peak_risk_agree_pct=100 * mean(sa["risk"][arm_f] == sr["risk"][ref_f]))
        rows.append(row)
    return rows


def trials(results, output_dir, participants):
    rows = list(csv.DictReader(open(results / f"{REFERENCE_TAG}.csv")))     # the sweep summary
    out = sorted({(r["participant"], r["task"]) for r in rows
                  if r.get("status") == "ok" and r["task"] in TASKS})
    if participants:
        out = [t for t in out if t[0] in participants]
    return [t for t in out if (output_dir / t[0] / t[1] / REFERENCE_TAG).is_dir()]


def completed(results, arm):
    path = results / "vs_mocap" / f"{arm}.csv"
    return {(r["participant"], r["task"]) for r in csv.DictReader(open(path))
            if r.get("status") == "ok"} if path.exists() else set()


def compute(args, ev):
    from rtcosmik.ergonomics import reba_posture as rp
    frames_dir = args.out / "robot_distance" / "frames" / ROBOT_FRAMES_ARM
    if not frames_dir.is_dir():
        frames_dir = args.results / "paper" / "robot_distance" / "frames" / ROBOT_FRAMES_ARM
    done = {arm: completed(args.results, arm) for arm in ARMS}
    events, per_arm = [], defaultdict(list)
    for participant, task in trials(args.results, args.output_dir, args.participants):
        ref_dir = args.output_dir / participant / task / REFERENCE_TAG
        rj, rm = pd.read_csv(ref_dir / "joint_angles.csv"), pd.read_csv(ref_dir / "markers.csv")
        neutral = rp.neutral_from(rj.iloc[rp.NEUTRAL_FRAMES], rm.iloc[rp.NEUTRAL_FRAMES])
        sr = rp.scores(rj, rm, neutral)
        try:
            event = select_event(task, rj, rm, neutral, frames_dir, participant)
        except Exception as exc:
            print(f"  {participant}/{task}: no event ({type(exc).__name__}: {exc})", flush=True)
            continue
        f = event["ref_frame"]
        frames, weights, threshold, n_above, n_tied = peak_set(sr["reba"])
        events.append({"participant": participant, "task": task, **event, "time_s": f / 40.0,
                       "reba_ref": float(sr["reba"][f]), "risk_ref": int(sr["risk"][f]),
                       **{f"{c}_ref": int(sr[c][f]) for c in COMPONENTS},
                       "peak_threshold": threshold, "peak_frames": float(weights.sum()),
                       "peak_above": n_above, "peak_tied": n_tied,
                       "peak_pct": 100.0 * weights.sum() / np.isfinite(sr["reba"]).sum(),
                       "trial_frames": len(rj)})
        for arm in ARMS:
            if (participant, task) not in done[arm]:
                continue
            run_dir = args.output_dir / participant / task / arm
            try:
                rows = score_arm(ev, run_dir, ref_dir, f, (frames, weights), rj, rm, neutral)
            except Exception as exc:
                print(f"  {arm} {participant}/{task}: {type(exc).__name__}: {exc}", flush=True)
                continue
            front_end, cameras = label(arm)
            for row in rows:
                per_arm[arm].append({"arm": arm, "front_end": front_end, "cameras": cameras,
                                     "participant": participant, "task": task, **row})
        print(f"  {participant}/{task}: {event['rule']} at frame {f} "
              f"({event['quantity']} = {event['value']:.3f} {event['unit']}), "
              f"reference REBA {sr['reba'][f]:.0f}; top decile {weights.sum():.0f} frames "
              f"({n_above} above REBA {threshold:.0f}, {n_tied} tied)", flush=True)

    out = args.out / "reba_events"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "events.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    for arm, rows in per_arm.items():
        with open(out / f"{arm}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ARM_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(events)} trials, {len(per_arm)} arms -> {out}", flush=True)


# --------------------------------------------------------------------------- tables

def number(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def per_participant(rows, field, task=None):
    """Participant -> mean of ``field`` over their trials (of ``task``)."""
    buckets = defaultdict(list)
    for r in rows:
        if task is None or r["task"] == task:
            v = number(r[field])
            if np.isfinite(v):
                buckets[r["participant"]].append(v)
    return {p: float(np.mean(v)) for p, v in buckets.items()}


def tex_table(path, caption, lead, header_groups, header, body):
    """Booktabs table: ``lead`` plain columns, then one column group per task."""
    esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
    ncol = len(header)
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{" + esc(caption) + "}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + "l" * lead + "c" * (ncol - lead) + "}",
             r"\toprule"]
    top, rules, col = [""] * lead, [], lead + 1
    for name, width in header_groups:
        top.append(rf"\multicolumn{{{width}}}{{c}}{{{esc(name)}}}")
        rules.append(rf"\cmidrule(lr){{{col}-{col + width - 1}}}")
        col += width
    lines += [" & ".join(top) + r" \\", " ".join(rules),
              " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    previous = None
    for row in body:
        cells = list(row)
        if cells[0] == previous:
            cells[0] = ""
        else:
            if previous is not None:
                lines.append(r"\addlinespace")
            previous = cells[0]
        lines.append(" & ".join(esc(c) for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def aggregate(args, agg):
    """tables/reba_events.* and tables/reba_peak.*, own neutral; stats on all-task MAE."""
    rng = np.random.default_rng(0)
    data = args.out / "reba_events"
    rows = {arm: [r for r in csv.DictReader(open(data / f"{arm}.csv")) if r["neutral"] == "own"]
            for arm in ARMS if (data / f"{arm}.csv").exists()}
    tables = args.out / "tables"
    specs = {
        "reba_events": ("reba_ref", "reba_arm", "reba_abs_err", "risk_agree", 100.0,
                        "Posture REBA at the representative posture of each task (frame chosen on "
                        "the reference), mean (SD) across participants; risk agreement in % of "
                        "trials (all tasks: mean (SD) of each participant's %)."),
        "reba_peak": ("peak_ref_mean", "peak_arm_mean", "peak_mae", "peak_risk_agree_pct", 1.0,
                      "Posture REBA over the top decile of reference REBA frames of each trial, "
                      "mean (SD) across participants."),
    }
    summary = {}
    for name, (f_ref, f_arm, f_mae, f_agree, scale, caption) in specs.items():
        lead = ["Front end", "Cameras", "Trials"]
        short = ["Ref.", "Est.", "MAE", "Risk agr. %"]
        groups = [(TASK_LABELS.get(t, "All tasks"), len(short)) for t in TASKS + ("all",)]
        header = lead + [f"{g} {c}" for g, _ in groups for c in short]      # csv / md
        tex_header = lead + short * len(groups)
        body, by_arm = [], {}
        for arm, sel in rows.items():
            front_end, cameras = label(arm)
            line = [front_end, cameras, str(len({(r['participant'], r['task']) for r in sel}))]
            summary.setdefault(name, {})[arm] = {}
            for t in TASKS + (None,):
                stats = {}
                for key, field in (("ref", f_ref), ("est", f_arm), ("mae", f_mae)):
                    stats[key] = agg.mean_sd(per_participant(sel, field, t).values())
                stats["agree"] = agg.mean_sd([v * scale for v in
                                              per_participant(sel, f_agree, t).values()])
                # One event per participant and task: its agreement is 0 or 100 %, so
                # per task only the share of participants in agreement is meaningful.
                if name == "reba_events" and t is not None:
                    cell = (f"{stats['agree']['mean']:.0f}"
                            if stats["agree"]["mean"] is not None else "--")
                else:
                    cell = agg.fmt(stats["agree"], 1)
                if t is None:
                    by_arm[arm] = per_participant(sel, f_mae)
                summary[name][arm][t or "all"] = stats
                line += [agg.fmt(stats["ref"]), agg.fmt(stats["est"]), agg.fmt(stats["mae"]), cell]
            body.append(line)
        tables.mkdir(parents=True, exist_ok=True)
        agg.write_table(tables / name, header, body)
        tex_table(tables / f"{name}.tex", caption, len(lead), groups, tex_header, body)

        # Planned comparisons on each participant's all-task MAE, as for the other tables.
        realtime = [a for a in by_arm if not a.startswith("fastsam")]
        stat = {"friedman": agg.friedman(by_arm, realtime),
                "planned": agg.compare(by_arm, agg.PLANNED, rng, correct=True),
                "offline": agg.compare(by_arm, agg.OFFLINE, rng, correct=False)}
        summary[name]["stats"] = stat
        srows = []
        for family in ("planned", "offline"):
            for c in stat[family]:
                fa, fb = label(c["a"]), label(c["b"])
                srows.append([family, f"{fa[0]}, {fa[1]}", f"{fb[0]}, {fb[1]}", c["n"],
                              f"{c['mean_diff']:+.2f}", f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]",
                              f"{c['p']:.4f}", f"{c['p_adjusted']:.4f}", c["correction"],
                              f"{c['rank_biserial']:+.2f}", "yes" if c["significant_0.05"] else "no"])
        f = stat["friedman"]
        srows.append(["friedman", f"{len(f['arms'])} real-time arms", "", f["n"], "", "",
                      f"{f['p']:.2e}" if f["p"] is not None else "--", "", "",
                      f"chi2({f['df']}) = {f['chi2']:.1f}" if f["chi2"] is not None else "--", ""])
        agg.write_table(tables / f"stats_{name}_mae",
                        ["family", "a", "b", "n", "mean diff (a - b) REBA points", "95% CI", "p",
                         "p adjusted", "correction", "rank-biserial", "sig"], srows)
    import json
    (args.out / "reba_events" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"tables written to {tables}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True, help="campaign run folders")
    ap.add_argument("--results", type=Path, required=True,
                    help="campaign results folder (vs_mocap/, paper/robot_distance/frames/)")
    ap.add_argument("--out", type=Path, required=True, help="where reba_events/ and tables/ go")
    ap.add_argument("--participants", nargs="*", default=None, help="restrict to these")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    ev = load_module(REPO / "scripts" / "python" / "eval" / "compare_to_mocap.py", "compare_to_mocap")
    agg = load_module(here / "aggregate_results.py", "aggregate_results")
    compute(args, ev)
    aggregate(args, agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
