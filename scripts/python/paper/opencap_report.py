"""Scores and tables of the OpenCap evaluation (the ``report`` stage of opencap_eval.py).

Written under ``<out>/results``: one row per trial and arm (``trials.csv``), per
DoF (``per_dof.csv``), OpenCap's released video results against its mocap ones
(``released.csv``), our reference against OpenCap's mocap angles
(``reference_agreement.csv``) and the tables (``tables.md``).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtcosmik.paper import opencap as oc
from rtcosmik.paper import scoring

RELEASED = [f"Video/{model}/{n}-cameras" for model in ("HRNet", "OpenPose_default", "OpenPose_highAccuracy")
            for n in (2, 3, 5)]
EARLY_S = 0.25      # the start of a trial, looked at apart for a warm-start transient


def _settings():
    from rtcosmik.config_loader import settings
    return oc.dataset_settings(settings)


def _run(out, subject, trial, tag):
    return out / "runs" / subject / trial / tag


def _complete(run_dir):
    path = run_dir / "run_info.json"
    return path.exists() and "frames" in json.loads(path.read_text())


def session_frames(dataset, out, subject, tag="nlf_5cam", references=None):
    """Per session, the rigid transform from the checkerboard frame to the mocap frame.

    Fitted once on the five-camera arm's markers against the reference's, all
    trials of the session pooled, and used for every arm. ``references`` is the
    folder holding the reference runs when it is not ``out``.
    """
    pairs = {}
    for trial in oc.trials(dataset / subject):
        run = _run(out, subject, trial.name, tag)
        reference = _run(references or out, subject, trial.name, scoring.REFERENCE_TAG)
        if not (_complete(run) and _complete(reference)):
            continue
        markers, ref_markers = scoring.by_frame(run)[2], scoring.by_frame(reference)[2]
        for name in sorted(set(markers) & set(ref_markers)):
            a, b = scoring.shift(markers[name], ref_markers[name], 0)
            source, target = pairs.setdefault(trial.session, ([], []))
            source.append(a)
            target.append(b)
    return {session: oc.rigid_fit(np.concatenate(source), np.concatenate(target))
            for session, (source, target) in pairs.items()}


def _early_error(run, reference, columns, frames, early):
    """Mean absolute joint error (deg) in the first ``early`` frames of the window and after."""
    names, q, _ = scoring.by_frame(run)
    q_ref = scoring.by_frame(reference)[1]
    q, q_ref = scoring.shift(q[frames[0]:frames[1] + 1], q_ref[frames[0]:frames[1] + 1], 0)
    q = scoring.match_shoulder_branch(names, q, q_ref)
    d = np.abs((np.degrees(q - q_ref)[:, columns] + 180) % 360 - 180)
    return float(np.nanmean(d[:early])), float(np.nanmean(d[early:]))


def score_subject(job):
    """Per-trial and per-DoF scores of one subject's arms, in the released window and on the whole trial."""
    dataset, out, subject = job
    settings = _settings()
    max_lag = round(settings.opencap["max_lag_s"] * settings.fs)
    early = round(EARLY_S * settings.fs)
    trials, dofs = [], []
    transforms = session_frames(dataset, out, subject)
    frames_used = {str(k): {"R": R.tolist(), "t": t.tolist()} for k, (R, t) in transforms.items()}
    for trial in oc.trials(dataset / subject):
        reference = _run(out, subject, trial.name, scoring.REFERENCE_TAG)
        if not _complete(reference):
            continue
        times = np.load(reference / "frame_times.npy")
        t_released = oc.read_mot(oc.released_ik(dataset / subject, trial.name))[0]
        window = oc.window_frames(times, (t_released[0], t_released[-1]))
        for tag in oc.ARMS:
            run = _run(out, subject, trial.name, tag)
            if not _complete(run):
                continue
            info = json.loads((run / "run_info.json").read_text())
            sync = json.loads((out / "synced" / subject / f"{trial.name}.json").read_text())
            for scope, frames in (("window", window), ("clip", None)):
                row, per_dof = scoring.score_trial(
                    run, reference, settings.paper["clock_offset"], unscored=oc.UNSCORED,
                    max_lag=max_lag, frames=frames, marker_frame=transforms[trial.session])
                key = dict(arm=tag, subject=subject, trial=trial.name, activity=trial.activity,
                           scope=scope)
                if scope == "window":
                    names = scoring.by_frame(run)[0]
                    columns = [names.index(d["dof"] + "[rad]") for d in per_dof]
                    first, later = _early_error(run, reference, columns, window, early)
                    row.update(early_mae_deg=first, later_mae_deg=later,
                               start_residual_mm=info.get("start_residual_mm"),
                               camera_offset_span=int(np.ptp(sync["camera_offsets_frames"])),
                               clock_offset_s=sync["clock_offset_from_nominal_s"],
                               sync_residual_mm=sync["residual_mm"],
                               far_views_pct=100 * float(np.mean(info["arm"]["far_views_share"])),
                               window_first=window[0], window_frames=window[1] - window[0] + 1,
                               clip_frames=len(times))
                trials.append({**key, **row})
                dofs += [{**key, **d} for d in per_dof]
    return subject, trials, dofs, frames_used


def score(dataset, out, subjects, workers):
    """Scores of every subject, one process each."""
    import multiprocessing
    trials, dofs, frames_used = [], [], {}
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for subject, t, d, f in pool.map(score_subject, [(dataset, out, s) for s in subjects]):
            trials += t
            dofs += d
            frames_used[subject] = f
    return pd.DataFrame(trials), pd.DataFrame(dofs), frames_used


def released(dataset, subjects):
    """OpenCap's released video angles against its mocap angles, per trial and coordinate."""
    rows = []
    for subject in subjects:
        for trial in oc.trials(dataset / subject):
            mocap = oc.released_ik(dataset / subject, trial.name)
            for source in RELEASED:
                video = oc.released_ik(dataset / subject, trial.name, source)
                if not video.is_file():
                    continue
                _, model, cameras = source.split("/")
                for name, e in oc.released_errors(video, mocap).items():
                    rows.append(dict(model=model, cameras=int(cameras[0]), subject=subject,
                                     trial=trial.name, activity=trial.activity, coordinate=name, **e))
    return pd.DataFrame(rows)


def reference_agreement(dataset, out, subjects):
    """Our reference angles against OpenCap's mocap angles on the matched DoF."""
    rows = []
    for subject in subjects:
        for trial in oc.trials(dataset / subject):
            run = _run(out, subject, trial.name, scoring.REFERENCE_TAG)
            if not _complete(run):
                continue
            names, q, _ = scoring.by_frame(run)
            times = np.load(run / "frame_times.npy")[:len(q)]
            t_released, theirs = oc.read_mot(oc.released_ik(dataset / subject, trial.name))
            for label, coordinates, columns in oc.matched_pairs():
                for coordinate, column in zip(coordinates, columns):
                    ours = np.degrees(q[:, names.index(column + "[rad]")])
                    r, rmse = oc.agreement(ours, oc.resample(t_released, theirs[coordinate], times))
                    rows.append(dict(subject=subject, trial=trial.name, dof=label, r=r,
                                     rmse_offset_free_deg=rmse))
    return pd.DataFrame(rows)


def over_subjects(frame, value, by=()):
    """Mean and SD over subjects of each subject's mean over its trials."""
    by = list(by)
    per_subject = frame.groupby(by + ["subject"])[value].mean()
    if not by:
        return pd.Series({"mean": per_subject.mean(), "sd": per_subject.std(ddof=1),
                          "n": per_subject.size})
    grouped = per_subject.groupby(level=list(range(len(by))))
    return pd.DataFrame({"mean": grouped.mean(), "sd": grouped.std(ddof=1), "n": grouped.size()})


def _pm(stat, digits=1):
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def _markdown(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def _matched(dofs, rel):
    """Per matched joint: ours per arm against our reference, OpenCap's against its own."""
    ours = dofs[dofs.scope == "window"].copy()
    ours["dof"] = ours["dof"].str.replace("Right_", "").str.replace("Left_", "")
    header = (["DoF"] + [f"ours {tag[4:]} causal" for tag in oc.ARMS] + [f"ours {tag[4:]} aligned" for tag in oc.ARMS]
              + [f"OpenCap {m} {c} cam" for m in ("HRNet", "OpenPose_default") for c in (2, 5)])
    rows, means = [], []
    for label, coordinates, columns in oc.matched_pairs():
        column = columns[0].replace("Right_", "")
        cells = []
        for kind in ("causal", "aligned"):
            for tag in oc.ARMS:
                sel = ours[(ours.arm == tag) & (ours.dof == column)]
                cells.append(over_subjects(sel, f"{kind}_rmse_deg")["mean"])
        for model in ("HRNet", "OpenPose_default"):
            for cameras in (2, 5):
                sel = rel[(rel.model == model) & (rel.cameras == cameras) & rel.coordinate.isin(coordinates)]
                cells.append(over_subjects(sel, "rmse")["mean"])
        means.append(cells)
        rows.append([label] + [f"{c:.1f}" for c in cells])
    rows.append(["mean"] + [f"{c:.1f}" for c in np.mean(means, axis=0)])
    lower = [i for i, (label, _, _) in enumerate(oc.matched_pairs())
             if not label.startswith(("arm", "elbow"))]
    rows.append(["mean without shoulder and elbow"]
                + [f"{c:.1f}" for c in np.mean(np.array(means)[lower], axis=0)])
    return _markdown(rows, header)


def tables(trials, dofs, rel, agreement):
    """The report's tables as Markdown."""
    out = []
    window = trials[trials.scope == "window"]
    for scope, frame in (("released analysis window", window), ("full clip", trials[trials.scope == "clip"])):
        rows = []
        for tag in oc.ARMS:
            a = frame[frame.arm == tag]
            if a.empty:
                continue
            stat = lambda v: _pm(over_subjects(a, v))
            rows.append([tag, a.subject.nunique(), len(a), stat("causal_rmse_mean"),
                         stat("aligned_rmse_mean"), stat("causal_marker_mm"), stat("aligned_marker_mm"),
                         f"{a.lag.mean():.2f}", f"{a.marker_lag.mean():.2f}",
                         int(a.solver_failures.sum()), int(a.dropped_frames.sum()),
                         f"{a.violation_pct.mean():.2f}", f"{a.jitter_deg.mean():.3f}"])
        out += [f"### Arms, {scope} (mean ± SD over subjects)", "", _markdown(rows, [
            "arm", "subjects", "trials", "causal deg", "aligned deg", "causal marker mm",
            "aligned marker mm", "joint lag frames", "marker lag frames", "solver failures",
            "dropped frames", "limit violation %", "jitter deg/frame"]), ""]

    rows = []
    for tag in oc.ARMS:
        a = window[window.arm == tag]
        for kind in ("causal", "aligned"):
            by = over_subjects(a, f"{kind}_rmse_mean", ["activity"])
            rows.append([tag, kind] + [_pm(by.loc[name]) if name in by.index else "" for name in oc.ACTIVITIES])
        by = over_subjects(a, "causal_marker_mm", ["activity"])
        rows.append([tag, "marker mm"] + [_pm(by.loc[name]) if name in by.index else "" for name in oc.ACTIVITIES])
    counts = window[window.arm == next(iter(oc.ARMS))].activity.value_counts()
    out += ["### Per activity (deg, mean ± SD over subjects; released analysis window)", "",
            _markdown(rows, ["arm", "error"] + [f"{n} ({counts.get(n, 0)} trials)" for n in oc.ACTIVITIES]), ""]

    d = dofs[dofs.scope == "window"]
    names = list(dict.fromkeys(d.dof))
    rows = []
    for name in names:
        row = [name]
        for tag in oc.ARMS:
            sel = d[(d.arm == tag) & (d.dof == name)]
            row += [_pm(over_subjects(sel, "causal_rmse_deg")), _pm(over_subjects(sel, "aligned_rmse_deg")),
                    f"{sel.causal_bias_deg.mean():.1f}"]
        rows.append(row)
    out += [f"### Per DoF, ours against our reference ({len(names)} scored DoF)", "",
            _markdown(rows, ["DoF"] + [f"{tag[4:]} {k}" for tag in oc.ARMS
                                       for k in ("causal", "aligned", "bias")]), ""]

    configs = [(m, c) for m in ("HRNet", "OpenPose_default", "OpenPose_highAccuracy") for c in (2, 3, 5)]
    coordinates = [c for c in oc.RELEASED_LOWER_BODY + oc.RELEASED_UPPER_BODY]
    rows = []
    for name in coordinates:
        rows.append([name] + [f"{over_subjects(rel[(rel.model == m) & (rel.cameras == c) & (rel.coordinate == name)], 'rmse')['mean']:.1f}"
                              for m, c in configs])
    for label, group, value in (("mean RMSE, 18 lower-body and trunk rotations", oc.RELEASED_LOWER_BODY, "rmse"),
                                ("mean MAE, 18 lower-body and trunk rotations", oc.RELEASED_LOWER_BODY, "mae"),
                                ("mean RMSE, 10 arm rotations", oc.RELEASED_UPPER_BODY, "rmse"),
                                ("mean RMSE, all 28", coordinates, "rmse")):
        row = [label]
        for m, c in configs:
            sel = rel[(rel.model == m) & (rel.cameras == c) & rel.coordinate.isin(group)]
            per_trial = sel.groupby(["subject", "trial"])[value].mean().reset_index()
            row.append(_pm(over_subjects(per_trial, value)))
        rows.append(row)
    n = rel.groupby(["model", "cameras"]).apply(lambda g: g[["subject", "trial"]].drop_duplicates().shape[0])
    out += ["### OpenCap's released video angles against its mocap angles (RMSE deg)", "",
            _markdown(rows, ["coordinate"] + [f"{m} {c} cam ({n.get((m, c), 0)} trials)" for m, c in configs]), ""]

    out += ["### Matched DoF (RMSE deg, mean over subjects; both sides pooled)", "", _matched(dofs, rel), ""]

    for scope, frame in (("released analysis window", rel),):
        rows = []
        for name in oc.ACTIVITIES:
            row = [name]
            for m, c in configs:
                sel = frame[(frame.model == m) & (frame.cameras == c) & (frame.activity == name)
                            & frame.coordinate.isin(oc.RELEASED_LOWER_BODY)]
                per_trial = sel.groupby(["subject", "trial"])["rmse"].mean().reset_index()
                row.append(_pm(over_subjects(per_trial, "rmse")))
            rows.append(row)
        out += ["### OpenCap's released error per activity (mean RMSE over the 18 rotations, deg)", "",
                _markdown(rows, ["activity"] + [f"{m} {c} cam" for m, c in configs]), ""]

    g = agreement.groupby("dof")
    rows = [[name, f"{g.r.median()[name]:.3f}", f"{g.rmse_offset_free_deg.mean()[name]:.1f}"]
            for name in dict.fromkeys(agreement.dof)]
    out += ["### Our reference against OpenCap's mocap angles", "",
            _markdown(rows, ["DoF", "median r", "RMSE without sign and offset, deg"]), ""]

    rows = []
    for tag in oc.ARMS:
        a = window[window.arm == tag]
        rows.append([tag, f"{a.early_mae_deg.mean():.2f}", f"{a.later_mae_deg.mean():.2f}",
                     f"{a.start_residual_mm.mean():.1f}", f"{a.start_residual_mm.max():.1f}",
                     f"{a.marker_lag.mean():.2f} ({a.marker_lag.min()} to {a.marker_lag.max()})",
                     f"{a.far_views_pct.mean():.2f}", int((a.far_views_pct > 0).sum()),
                     int((a.far_views_pct > 99 / len(oc.ARMS[tag])).sum())])
    sync = window[window.arm == next(iter(oc.ARMS))]
    sync_rows = [[name, len(g), f"{g.camera_offset_span.mean():.1f}", int(g.camera_offset_span.max()),
                  f"{g.clock_offset_s.mean():+.3f}", f"{g.clock_offset_s.min():+.3f} to {g.clock_offset_s.max():+.3f}",
                  f"{g.sync_residual_mm.mean():.0f}", f"{(g.window_frames / g.clip_frames).mean():.2f}"]
                 for name, g in sync.groupby("activity")]
    out += [f"### Start of a trial (first {EARLY_S} s of the window against the rest)", "",
            _markdown(rows, ["arm", "MAE first frames deg", "MAE later deg", "start residual mm, mean",
                             "start residual mm, max", "marker lag frames, mean (range)",
                             "views dropped as far %", "trials with any", "trials with a camera dropped throughout"]), "",
            "### Synchronisation of the released clips", "",
            _markdown(sync_rows, ["activity", "trials", "camera offset span frames, mean", "max",
                                  "clock offset from the clip's nominal start s, mean", "range",
                                  "residual after rigid fit mm", "scored share of the synchronised frames"]), ""]
    return "\n".join(out)


def write(dataset, out, subjects, workers=8):
    results = Path(out) / "results"
    results.mkdir(parents=True, exist_ok=True)
    trials, dofs, frames_used = score(dataset, out, subjects, workers)
    rel = released(dataset, subjects)
    agreement = reference_agreement(dataset, out, subjects)
    trials.to_csv(results / "trials.csv", index=False)
    dofs.to_csv(results / "per_dof.csv", index=False)
    rel.to_csv(results / "released.csv", index=False)
    agreement.to_csv(results / "reference_agreement.csv", index=False)
    (results / "session_frames.json").write_text(json.dumps(frames_used, indent=1))
    text = tables(trials, dofs, rel, agreement)
    (results / "tables.md").write_text(text + "\n")
    print(text)
