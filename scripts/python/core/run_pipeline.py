#!/usr/bin/env python3
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
import json
import os

import time
from pathlib import Path
import meshcat
import meshcat.geometry as g

import numpy as np
import torch
import pinocchio as pin 
from pinocchio.visualize import MeshcatVisualizer

from rtcosmik.config_loader import settings
from rtcosmik.nlf.nlf import NLFEstimator, extract_views
from rtcosmik.triangulation.triangulation import reconstruct_3d
from rtcosmik.filtering.iir import IIR
from rtcosmik.pipeline.solver import HumanSolver
from rtcosmik.camera.cam_utils import list_cameras, load_camera_parameters, load_world_transformation
from rtcosmik.camera.camera import Camera
from rtcosmik.utils.mp_utils import create_camera_shared_ressources
from rtcosmik.utils.VideoReader import OfflineVideoSource
from rtcosmik.saver.csv_saver import CSVSaver
from rtcosmik.saver.hotkeys import TerminalHotkeys
from rtcosmik.viewer.async_display import AsyncDisplay
from rtcosmik.utils.dataset import (
    TRIAL_CLI_EPILOG, add_trial_arguments, load_subject, resolve_trial, run_variant)
from rtcosmik.model_weights import resolve_detector_engine
from rtcosmik.pipeline.pipeline import PipelineProcess

from multiprocessing import set_start_method, Value, Event as MPEvent
from collections import deque, OrderedDict

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)

LOGGER = logging.getLogger(__name__)



def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine size
    W = settings.width
    H = settings.height

    if args.online:
        # --cam-params still applies online: a replay uses the recordings' own
        # calibration, not whatever happens to sit in config/cam_params.
        cam_params_path = args.cam_params or settings.cam_calib_path
        # --subject too: replaying a recorded subject with default anthropometry
        # would calibrate a different body than the one in the video.
        subject_path = args.subject
        video_paths = out_dir = None
    else:
        cam_params_path, video_paths, subject_path, out_dir = resolve_trial(args)
        if len(video_paths) != len(args.cameras):
            raise ValueError(
                f"{len(video_paths)} videos but {len(args.cameras)} cameras requested; "
                "pass --cameras matching the videos, in the same order"
            )

    # Cameras are loaded in the requested order and the first is the reference
    # frame triangulation outputs into, so the world transform uses that one.
    mtxs, dists, projections, rotations, translations = load_camera_parameters(
        cam_params_path, args.cameras
    )
    world_R1_cam, world_T1_cam = load_world_transformation(cam_params_path, args.cameras[0])

    if args.online:
        # --replay feeds recordings through the *online* path: the same camera
        # processes, barrier, shared buffers and pipeline, with files standing in
        # for devices. Paced at the recording's own frame rate, so it shows
        # whether the pipeline keeps up rather than just how fast it can chew
        # through a file. It exercises the software path, not the capture
        # hardware: every file source is always ready, so the barrier never
        # actually waits and real inter-camera skew stays invisible.
        if args.replay:
            replay_dir = Path(args.replay)
            sources = [str(replay_dir / f"camera_{c}.mp4") for c in args.cameras]
            missing = [s for s in sources if not Path(s).is_file()]
            if missing:
                raise FileNotFoundError(f"missing recordings for replay: {missing}")
            camera_ids = list(args.cameras)
            LOGGER.info("[CAP] replaying %d recordings from %s as live cameras",
                        len(sources), replay_dir)
        else:
            cameras = list_cameras()
            camera_ids = list(cameras.keys())
            sources = [None] * len(camera_ids)

        NUM_CAMERAS = len(camera_ids)
        FRAME_SHAPE = (H, W, 3)
        camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
        # Shared with the video writers so one toggle drives every recorder.
        saving_flag = Value('b', settings.record_on_start)
        # Replay sources hold their first frame until this is set.
        calibrated_event = MPEvent()

        # Recording is a stream copy alongside capture, so it costs no decode
        # and no re-encode -- which is why the online path can now save video at
        # all.
        record_paths = [None] * NUM_CAMERAS
        if settings.SAVE_VID:
            os.makedirs(settings.SAVE_DIR, exist_ok=True)
            record_paths = [os.path.join(settings.SAVE_DIR, f"camera_{c}.mkv")
                            for c in camera_ids]
            LOGGER.info("[CAP] recording video to %s", settings.SAVE_DIR)

        camera_processes = [
            Camera(camera_ids[i],
                camera_buffers[i],
                camera_timestamps[i],
                camera_locks[i],
                frame_counters[i],
                camera_barrier,
                stop_event,
                FRAME_SHAPE,
                settings.fs,
                settings.fourcc,
                source=sources[i],
                record_path=record_paths[i],
                realtime=bool(args.replay),
                calibrated_event=calibrated_event if args.replay else None)
            for i in range(NUM_CAMERAS)
        ]

        online_height, online_weight, online_gender = load_subject(subject_path)
        pipeline = PipelineProcess(
            settings=settings,
            subject=(online_height, online_weight, online_gender),
            frame_counters=frame_counters,
            camera_buffers=camera_buffers,
            camera_locks=camera_locks,
            timestamp_buffers=camera_timestamps,
            stop_event=stop_event,
            saving_flag=saving_flag,
            calibrated_event=calibrated_event,
            mtxs=mtxs,
            dists=dists,
            projections=projections,
            world_R1_cam=world_R1_cam,
            world_T1_cam=world_T1_cam,
            frame_shape=FRAME_SHAPE,
            num_cameras=NUM_CAMERAS,
        )

        # Display and recording now live inside the pipeline process, where the
        # calibrated model and the results already are.
        processes = camera_processes + [pipeline]

        # Start processes
        for p in processes:
            p.start()

        # The hotkey listener lives here, in the parent: children are started
        # with 'spawn' and get /dev/null for stdin, and reading the terminal is
        # what works over SSH where pynput's X hook does not.
        def _set_recording(on):
            saving_flag.value = on
            LOGGER.info("[KEY] recording %s", "started" if on else "stopped")

        hotkeys = TerminalHotkeys(
            {"s": lambda: _set_recording(True),
             "q": lambda: _set_recording(False)}, logger=LOGGER)
        with hotkeys:
            if hotkeys.active:
                LOGGER.info("[KEY] press 's' to start recording, 'q' to stop, "
                            "Ctrl-C to quit")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                stop_event.set()
                # Stop processes
                for process in processes:
                    process.stop() if hasattr(process, 'stop') else None
                    process.join(timeout=2)
    
    else: # offline mode

        # --- 1. INITIALISATION MESHCAT ---
        vis = meshcat.Visualizer()
        LOGGER.info(f"[INFO] Meshcat visualizer available here: {vis.url()}")

        vis_markers = vis["markers"]

        paths = video_paths
        NUM_CAMERAS = len(paths)
        LOGGER.info(
            "Processing %d camera(s): %s", NUM_CAMERAS, ", ".join(str(p) for p in paths)
        )

        subject_height, subject_weight, subject_gender = load_subject(subject_path)

        # loop=False so the run ends at the end of the videos instead of
        # restarting them, which is what makes sweeping over trials possible.
        src = OfflineVideoSource(paths=paths, size_wh=(W, H), loop=False)
        try:

            # joint_angles.csv uses the reference mocap's column names so a trial's
            # estimate lines up with its ground truth without renaming anything.
            saver = None
            if not args.no_save:
                saver = CSVSaver(
                    str(out_dir),
                    markers_header=['Frame'] + list(settings.marker_names),
                    joint_angles_header=list(settings.joint_angles_names),
                )
                LOGGER.info("Writing markers.csv and joint_angles.csv to %s", out_dir)
            frames_read = 0
            frames_written = 0

            est = NLFEstimator(
                yolo_path=resolve_detector_engine(settings.yolo_path, NUM_CAMERAS),
                nlf_path=settings.nlf_path,
                cano_path=settings.cano_path,
                image_size=(W, H),
                cam_Ks=mtxs,
                indices=settings.nlf_indices,
                conf=settings.yolo_conf,
                imgsz=settings.yolo_imgsz,
                device=settings.device,
            )

            # Init for the rest
            solver = HumanSolver(settings, gender=subject_gender, height=subject_height,
                                 weight=subject_weight, logger=LOGGER)
            first_sample = True
            p3d_buffer = deque(maxlen=settings.N)

            # Filter
            num_channel = 3*len(settings.marker_names)
            iir_filter = IIR(
                num_channel=num_channel,
                sampling_frequency=settings.fs
            )
            iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq, filter_type=settings.filter_type)

            display = AsyncDisplay(logger=LOGGER)
            display.__enter__()

            stage_ms = {"read": [], "pose": [], "reconstruct": [], "ik": [],
                        "display": [], "save": [], "frame": []}
            # NLFEstimator already reports its own split; it was being discarded.
            pose_parts = {"yolo": [], "h2d+pre": [], "nlf": []}
            calibration_ms = None
            viewer_setup_ms = 0.0   # subtracted from the frame it occurs in
            viewer_total_ms = 0.0   # kept for the summary
            first_frame_calibration_ms = 0.0  # subtracted from frame 0

            while True:
                t0=time.perf_counter()
                frames = src.read()
                t_read=time.perf_counter()
                if frames is None:
                    break
                frames_read += 1

                nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)
                t_pose=time.perf_counter()
                pose_parts["yolo"].append(infer_ms.get("yolo_ms", float("nan")))
                pose_parts["h2d+pre"].append(infer_ms.get("h2d+pre_ms", float("nan")))
                pose_parts["nlf"].append(infer_ms.get("nlf_ms", float("nan")))

                views = extract_views(nlf_out, NUM_CAMERAS)
                p3d = reconstruct_3d(views, projections)
                t_rec=time.perf_counter()
                if len(p3d) == 0:
                    continue

                p3d_in_world=np.array([np.dot(world_R1_cam,point) + world_T1_cam for point in p3d])

                if first_sample:
                    for k in range(settings.N):
                        p3d_buffer.append(p3d_in_world)  # add the 1st frame 30 times
                else:
                    p3d_buffer.append(p3d_in_world) # add the keypoints to the buffer normally
            
                if len(p3d_buffer) == settings.N:
                    p3d_buffer_array = np.array(p3d_buffer)

                    # Filter keypoints in world to remove noisy artefacts 
                    filtered_p3d_buffer = iir_filter.filter(np.reshape(p3d_buffer_array,(settings.N, 3*len(settings.marker_names))))
                    filtered_p3d_buffer = np.reshape(filtered_p3d_buffer,(settings.N, len(settings.marker_names), 3))

                    augmented_markers=filtered_p3d_buffer[-1]

                    # VISUALISATION OF AUGMENTED MARKERS in RED
                    colors = np.zeros_like(augmented_markers.T)
                    colors[0, :] = 1.0  # R
                    colors[1, :] = 0.0  # G
                    colors[2, :] = 0.0  # B

                    t_disp0=time.perf_counter()
                    display.submit(
                        lambda pts=augmented_markers.T.copy(), col=colors.copy():
                        vis_markers.set_object(g.PointCloud(position=pts, color=col,
                                                            size=0.02)))
                    display_ms = (time.perf_counter()-t_disp0)*1e3

                    mks_dict = dict(zip(settings.marker_names, augmented_markers))

                    if first_sample:
                        # Kept OUT of the per-frame IK statistics: this call builds
                        # and scales the model, registers the markers, runs an IPOPT
                        # solve and loads the OCP. It is seconds, happens once, and
                        # would otherwise sit in the same distribution as the
                        # millisecond steady-state solves.
                        t_ik0=time.perf_counter()
                        q = solver.calibrate(mks_dict)
                        calibration_ms = (time.perf_counter()-t_ik0)*1e3
                        first_frame_calibration_ms = calibration_ms
                        human_model, human_data = solver.model, solver.data

                        # Also one-off, and also excluded: loadViewerModel uploads
                        # the whole human mesh to the meshcat server over a
                        # websocket, which is ~1 s. Left in, it lands in the frame
                        # statistics as a single ~1000 ms outlier that looks like a
                        # solver stall.
                        t_viz0=time.perf_counter()
                        # Init meshcat viewer for human
                        viz_human = MeshcatVisualizer(
                            human_model, solver.collision_model, solver.visual_model)
                        viz_human.initViewer(vis, open=True)

                        # Don't delete the whole Meshcat tree: keep '/markers' etc.
                        try:
                            vis["ref"].delete()
                        except Exception:
                            pass
                        viz_human.loadViewerModel("ref")

                        viz_human.viewer["/Background"].set_property("top_color", [1, 1, 1])
                        viz_human.viewer["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])
                        viewer_setup_ms = (time.perf_counter()-t_viz0)*1e3
                        viewer_total_ms = viewer_setup_ms

                        first_sample = False
                    else:
                        t_ik0=time.perf_counter()
                        q = solver.step(mks_dict)
                        stage_ms["ik"].append((time.perf_counter()-t_ik0)*1e3)

                    t_disp1=time.perf_counter()
                    display.submit(lambda qq=np.array(q, copy=True): viz_human.display(qq))
                    display_ms += (time.perf_counter()-t_disp1)*1e3
                    stage_ms["display"].append(display_ms)
                    t_save0=time.perf_counter()

                    if saver is not None:
                        marker_row = OrderedDict(Frame=frames_read)
                        for name, position in mks_dict.items():
                            marker_row[name + '_x'] = float(position[0])
                            marker_row[name + '_y'] = float(position[1])
                            marker_row[name + '_z'] = float(position[2])
                        saver.save_markers(marker_row)

                        if len(q) != len(settings.joint_angles_names):
                            raise ValueError(
                                f"Model has {len(q)} configuration variables but "
                                f"{len(settings.joint_angles_names)} joint angle names are defined"
                            )
                        saver.save_joint_angles(
                            OrderedDict(zip(settings.joint_angles_names, (float(v) for v in q)))
                        )
                        frames_written += 1
                    stage_ms["save"].append((time.perf_counter()-t_save0)*1e3)
                t1=time.perf_counter()
                # perf_counter is in SECONDS; this used to be printed as "ms",
                # understating every timing by a factor of 1000.
                stage_ms["read"].append((t_read-t0)*1e3)
                stage_ms["pose"].append((t_pose-t_read)*1e3)
                stage_ms["reconstruct"].append((t_rec-t_pose)*1e3)
                # One-off setup is charged to its own line, not to this frame. Both
                # of them: the model build and the viewer upload happen on frame 0
                # and together were showing up as a ~1000 ms "frame" outlier.
                one_off = viewer_setup_ms + first_frame_calibration_ms
                stage_ms["frame"].append((t1-t0)*1e3 - one_off)
                viewer_setup_ms = 0.0
                first_frame_calibration_ms = 0.0
                if frames_read % 100 == 0:
                    print(f"  {frames_read} frames, last {(t1-t0)*1e3:.1f} ms", flush=True)
        finally:
            # Explicit teardown: an unreleased decoder never exits on
            # its own, it blocks on a full pipe holding GPU memory.
            src.release()
        display.close()

        # Timing summary. Medians, because the first frames include model
        # calibration and solver warm-up and would drag a mean.
        if stage_ms["frame"]:
            if calibration_ms is not None:
                print(f"\nOne-off setup, excluded below: model calibration "
                      f"{calibration_ms/1000:.1f} s, viewer {viewer_total_ms/1000:.1f} s")
            print(f"Timing over {len(stage_ms['frame'])} frames "
                  f"(median / p95 / max, ms; @ = frame of the max):")
            for name in ("read", "pose", "reconstruct", "ik", "display",
                         "save", "frame"):
                vals = np.asarray(stage_ms[name], dtype=float)
                if vals.size:
                    # Where the max happened separates a one-off from a
                    # recurring stall; without it a single outlier is
                    # indistinguishable from a periodic one.
                    print(f"  {name:<12} {np.median(vals):7.1f} / "
                          f"{np.percentile(vals, 95):7.1f} / {vals.max():7.1f}"
                          f"   @ {int(np.argmax(vals))}")
            print("  pose breaks down as:")
            for name in ("yolo", "h2d+pre", "nlf"):
                vals = np.asarray(pose_parts[name], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    print(f"    {name:<10} {np.median(vals):7.1f} / "
                          f"{np.percentile(vals, 95):7.1f} / {vals.max():7.1f}")
            fps = 1000.0 / max(np.median(stage_ms["frame"]), 1e-9)
            print(f"  -> {fps:.1f} fps sustained (dataset is {settings.fs} fps)")
            # What the same pipeline would sustain with the viewer detached.
            if stage_ms["display"]:
                headless = np.median(stage_ms["frame"]) - np.median(stage_ms["display"])
                print(f"  -> {1000.0/max(headless,1e-9):.1f} fps without display "
                      f"({headless:.1f} ms/frame)")
            print("  ('ik' is the solver alone; 'pose' is YOLO + preprocess + NLF)")

        if saver is not None:
            saver.close()
            # Provenance so an evaluation can tell runs apart and align frames.
            # The model's root placement is recorded because the free-flyer pose
            # in joint_angles.csv is expressed in that frame.
            run_info = {
                "participant": args.participant,
                "task": args.task,
                "cameras": list(args.cameras),
                "num_cameras": NUM_CAMERAS,
                "videos": [str(v) for v in paths],
                "fps": settings.fs,
                "ik": {
                    "type": settings.ik_type,
                    "mhe_backend": settings.mhe_backend if settings.ik_type == "mhe" else None,
                    "horizon_N": settings.N if settings.ik_type == "mhe" else None,
                    "cost_weights": list(settings.cost_weights) if settings.ik_type == "mhe" else None,
                    "mhe_profile": settings.mhe_profile if settings.ik_type == "mhe" else None,
                },
                "filter": {
                    "order": settings.order,
                    "cutoff_hz": settings.cutoff_freq,
                    "type": settings.filter_type,
                },
                "variant": run_variant(args.cameras),
                "subject": {"height": subject_height,
                            "weight": subject_weight,
                            "gender": subject_gender},
                "frames_read": frames_read,
                "frames_written": frames_written,
                "joint_angles_names": list(settings.joint_angles_names),
                "marker_names": list(settings.marker_names),
                "root_placement_rotation":
                    np.asarray(human_model.jointPlacements[1].rotation).tolist(),
                # The model is rescaled from the measured markers during
                # calibration, so record the resulting kinematics: that lets a
                # viewer rebuild exactly the model the IK ran on rather than a
                # nominal one built from height and weight alone.
                "joint_names": [human_model.names[i] for i in range(human_model.njoints)],
                "joint_placements": [
                    np.asarray(human_model.jointPlacements[i].translation).tolist()
                    for i in range(human_model.njoints)
                ],
            }
            with open(Path(out_dir) / "run_info.json", "w") as handle:
                json.dump(run_info, handle, indent=2)
        LOGGER.info(
            "Finished: %d frames read, %d rows written to %s",
            frames_read, frames_written, out_dir,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run the RT-COSMIK pipeline live, or offline over one recorded trial.",
        epilog=TRIAL_CLI_EPILOG + "\nSweep trials with a shell loop; there is no separate batch script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--replay", default=None, metavar="DIR",
                   help="run the ONLINE path against recordings in DIR "
                        "(camera_<id>.mp4), paced as if live")
    p.add_argument("--online", action="store_true",
                   help="Capture from live cameras instead of video files")
    add_trial_arguments(p)
    p.add_argument("--no-save", action="store_true", help="Visualise only, write no CSV files")
    args = p.parse_args()

    if args.online:
        set_start_method('spawn')

    main(args)