"""Locating the pieces of one recorded trial on disk.

A recording is laid out the same way regardless of which study produced it::

    <dataset>/cam_params/<participant>/                      calibration
    <dataset>/videos/<participant>/<task>/camera_<i>.mp4     synchronised video
    <dataset>/metadata/<participant>.yaml                    height, weight, gender

so anything in this layout works, not one dataset in particular.
"""

import logging
from pathlib import Path

from rtcosmik.config_loader import settings
from rtcosmik.utils.read_write_utils import read_subject_yaml

LOGGER = logging.getLogger(__name__)


# -----------------------
# Offline trial resolution
# -----------------------

# A recording is laid out the same way whether or not it comes from COMFI:
#   <dataset>/cam_params/<participant>/          calibration, see cam_utils
#   <dataset>/videos/<participant>/<task>/camera_<i>.mp4
#   <dataset>/metadata/<participant>.yaml        id, height, weight, gender
# --dataset/--participant/--task is shorthand for the three explicit paths;
# any dataset in this layout works, and the explicit flags take precedence.

def run_variant(cameras):
    """Short tag naming the settings that distinguish one run from another.

    Offline results are indexed by it, so changing the IK method or the camera
    count writes to a new directory instead of overwriting the previous run and
    the variants stay directly comparable. The full configuration is recorded in
    each run's run_info.json; only the discriminating knobs go in the name.
    """
    parts = [f"{len(cameras)}cam", settings.ik_type]
    if settings.ik_type == "mhe":
        parts.append(settings.mhe_backend)
    return "_".join(parts)


def resolve_trial(args):
    """Work out calibration, video and subject paths for one offline trial."""
    cam_params = args.cam_params
    trial_dir = args.trial_dir
    subject = args.subject
    out_dir = args.out

    if args.dataset:
        if not (args.participant and args.task):
            raise ValueError("--dataset requires both --participant and --task")
        root = Path(args.dataset)
        cam_params = cam_params or root / "cam_params" / args.participant
        trial_dir = trial_dir or root / "videos" / args.participant / args.task
        subject = subject or root / "metadata" / f"{args.participant}.yaml"
        out_dir = out_dir or (Path(settings.output_dir) / args.participant
                              / args.task / run_variant(args.cameras))
    elif not args.videos:
        if not trial_dir:
            raise ValueError("Offline mode needs --dataset, --trial-dir or --videos")
        out_dir = out_dir or (Path(settings.output_dir) / Path(trial_dir).name
                              / run_variant(args.cameras))

    if args.videos:
        video_paths = [Path(v) for v in args.videos]
        out_dir = out_dir or (Path(settings.output_dir) / settings.no_trial
                              / run_variant(args.cameras))
    else:
        trial_dir = Path(trial_dir)
        if not trial_dir.is_dir():
            raise FileNotFoundError(f"Trial directory does not exist: {trial_dir}")
        video_paths = [trial_dir / f"camera_{cam}.mp4" for cam in args.cameras]
        missing = [str(p) for p in video_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing videos in {trial_dir}: {', '.join(missing)}")

    if cam_params is None:
        raise ValueError("Offline mode needs --cam-params (or --dataset)")

    return Path(cam_params), video_paths, (Path(subject) if subject else None), Path(out_dir)


def load_subject(subject_path):
    """Read subject anthropometry, falling back to the configured defaults."""
    if subject_path is None:
        LOGGER.warning(
            "No --subject given; using settings defaults (h=%.2f m, m=%.1f kg, %s)",
            settings.human_height, settings.human_weight, settings.human_gender,
        )
        return settings.human_height, settings.human_weight, settings.human_gender

    _, height, weight, gender = read_subject_yaml(subject_path)

    # The model's anthropometric regressions test `gender == 'm'`, so a spelled
    # out "male" would silently select the female scaling for every subject.
    normalized = str(gender).strip().lower()[:1]
    if normalized not in ("m", "f"):
        raise ValueError(f"Unrecognised gender {gender!r} in {subject_path}")

    LOGGER.info(
        "Subject %s: height=%.2f m, weight=%.1f kg, gender=%s",
        Path(subject_path).stem, height, weight, normalized,
    )
    return height, weight, normalized

def add_trial_arguments(parser, cameras_default=None):
    """Add the flags that select one offline trial.

    Shared by every offline entry point so they accept the same arguments.
    """
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset root; shorthand for --cam-params/--trial-dir/--subject")
    parser.add_argument("--participant", type=str, default=None,
                        help="Participant id within --dataset")
    parser.add_argument("--task", type=str, default=None, help="Task name within --dataset")

    parser.add_argument("--cam-params", type=str, default=None,
                        help="Calibration directory (intrinsics/ and extrinsics/)")
    parser.add_argument("--trial-dir", type=str, default=None,
                        help="Directory holding camera_<i>.mp4 for one trial")
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject YAML with height, weight and gender")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Explicit video list, ordered to match --cameras")

    if cameras_default is None:
        cameras_default = list(settings.cameras)
    parser.add_argument("--cameras", type=int, nargs="+", default=cameras_default,
                        help="Camera ids to use; the first is the triangulation reference frame")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for the CSV files")
    return parser


TRIAL_CLI_EPILOG = (
    "Offline example (dataset in the standard layout):\n"
    "  %(prog)s --dataset /path/to/COMFI --participant 1012 --task Lifting\n"
    "Explicit paths work with any dataset in the same format:\n"
    "  %(prog)s --cam-params CAL/S03 --trial-dir VID/S03/Lifting --subject META/S03.yaml"
)
