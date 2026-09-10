#!/usr/bin/env python3
import argparse
from collections import OrderedDict, deque
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multiprocessing import Event as MPEvent, Value, set_start_method
import cv2
import meshcat
import meshcat.geometry as g
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from rtcosmik.camera.cam_utils import (
    list_cameras,
    load_camera_parameters,
    load_world_transformation,
)
from rtcosmik.camera.camera import Camera
from rtcosmik.config_loader import settings
from rtcosmik.filtering.iir import IIR
from rtcosmik.model_weights import resolve_detector_engine
from rtcosmik.nlf.nlf import NLFEstimator, extract_views
from rtcosmik.pipeline.pipeline import PipelineProcess
from rtcosmik.pipeline.solver import HumanSolver
from rtcosmik.saver.csv_saver import CSVSaver
from rtcosmik.saver.hotkeys import TerminalHotkeys
from rtcosmik.triangulation.triangulation import reconstruct_3d
from rtcosmik.utils.dataset import (
    TRIAL_CLI_EPILOG,
    add_trial_arguments,
    load_subject,
    resolve_trial,
    run_variant,
)
from rtcosmik.utils.mp_utils import create_camera_shared_ressources
from rtcosmik.utils.VideoReader import OfflineVideoSource
from rtcosmik.viewer.async_display import AsyncDisplay
from rtcosmik.viewer.viewer import ViserRobotVisualizer
import torch
import viser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

LOGGER = logging.getLogger(__name__)

SENTINEL = None


def _timing_summary(values):
    """mean/median/p95/max in ms for a list of floats, or None if empty."""
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "n": int(arr.size),
    }


def run_pipelined(
    src,
    est,
    args,
    settings,
    projections,
    world_R1_cam,
    world_T1_cam,
    solver,
    saver,
    display,
    vis,
    vis_markers,
    NUM_CAMERAS,
):
    """Three-stage producer/consumer pipeline (reader / gpu_worker / ik_worker),
    mirroring the threading model used in the benchmark script. Runs entirely
    on background threads so read, pose-estimation and IK+save overlap instead
    of running back-to-back in a single loop.
    """
    frame_q = queue.Queue(maxsize=3)
    infer_q = queue.Queue(maxsize=3)
    stop_event = threading.Event()

    # --- Timing collections (filled in by the three threads) ---
    read_ms, pose_ms, reconstruct_ms, ik_ms, display_ms_list, save_ms_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    pose_parts = {"yolo": [], "h2d+pre": [], "nlf": []}
    frame_latencies = []
    frame_start_ts = {}
    frame_start_lock = threading.Lock()

    span_lock = threading.Lock()
    first_read_ts = [None]
    last_done_ts = [None]
    calib_done_ts = [None]  # first frame that reaches full IK (calibration frame)
    warm_done_ts = [None]  # second completed frame (first-call/JIT spike)

    # Mutable state shared out of ik_worker once it's known/finished.
    state = {
        "frames_read": 0,
        "frames_written": 0,
        "calibration_ms": None,
        "viewer_setup_ms": 0.0,
        "human_model": None,
        "error": None,
    }

    def reader():
        idx = 0
        while not stop_event.is_set():
            t0 = time.perf_counter()
            frames = src.read()
            if frames is None:
                break
            read_ms.append((time.perf_counter() - t0) * 1e3)
            with frame_start_lock:
                frame_start_ts[idx] = t0
            with span_lock:
                if first_read_ts[0] is None:
                    first_read_ts[0] = t0
            state["frames_read"] += 1
            frame_q.put((idx, frames))
            idx += 1
        frame_q.put(SENTINEL)

    def gpu_worker():
        while True:
            item = frame_q.get()
            if item is SENTINEL:
                infer_q.put(SENTINEL)
                break
            idx, frames = item
            t0 = time.perf_counter()
            nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)
            pose_ms.append((time.perf_counter() - t0) * 1e3)
            pose_parts["yolo"].append(infer_ms.get("yolo_ms", float("nan")))
            pose_parts["h2d+pre"].append(
                infer_ms.get("h2d+pre_ms", float("nan"))
            )
            pose_parts["nlf"].append(infer_ms.get("nlf_ms", float("nan")))

            if args.show_nlf:
                vis_frames = est.visualize_frames(
                    frames,
                    nlf_out,
                    boxes=boxes,
                    draw_boxes=True,
                    put_text=True,
                    text_prefix="cam",
                )
                cv2.imshow("NLF Output", np.hstack(vis_frames))
                cv2.waitKey(1)

            infer_q.put((idx, nlf_out))

    def ik_worker():
        markers_handle = None
        viz_human = None
        first_sample = True
        p3d_buffer = deque(maxlen=settings.N)
        num_channel = 3 * len(settings.marker_names)
        iir_filter = IIR(
            num_channel=num_channel, sampling_frequency=settings.fs
        )
        iir_filter.add_filter(
            order=settings.order,
            cutoff=settings.cutoff_freq,
            filter_type=settings.filter_type,
        )

        while True:
            item = infer_q.get()
            if item is SENTINEL:
                break
            idx, nlf_out = item
            try:
                t0 = time.perf_counter()

                views = extract_views(nlf_out, NUM_CAMERAS)
                p3d = reconstruct_3d(views, projections)
                t_rec = time.perf_counter()
                reconstruct_ms.append((t_rec - t0) * 1e3)
                if len(p3d) == 0:
                    continue

                p3d_in_world = np.array(
                    [
                        np.dot(world_R1_cam, point) + world_T1_cam
                        for point in p3d
                    ]
                )

                if first_sample:
                    for _ in range(settings.N):
                        p3d_buffer.append(p3d_in_world)
                else:
                    p3d_buffer.append(p3d_in_world)

                if len(p3d_buffer) != settings.N:
                    continue

                p3d_buffer_array = np.array(p3d_buffer)
                filtered_p3d_buffer = iir_filter.filter(
                    np.reshape(
                        p3d_buffer_array,
                        (settings.N, 3 * len(settings.marker_names)),
                    )
                )
                filtered_p3d_buffer = np.reshape(
                    filtered_p3d_buffer,
                    (settings.N, len(settings.marker_names), 3),
                )
                augmented_markers = filtered_p3d_buffer[-1]

                # --- Marker point cloud (RED), offloaded to AsyncDisplay ---
                t_disp0 = time.perf_counter()
                if args.visualizer == "meshcat":
                    colors = np.zeros_like(augmented_markers.T)
                    colors[0, :] = 1.0  # R
                    display.submit(
                        lambda pts=augmented_markers.T.copy(), col=colors.copy(): (
                            vis_markers.set_object(
                                g.PointCloud(position=pts, color=col, size=0.02)
                            )
                        )
                    )
                elif args.visualizer == "viser":
                    pts = augmented_markers.astype(np.float32)
                    cols = np.zeros((pts.shape[0], 3), dtype=np.uint8)
                    cols[:, 0] = 255  # R
                    if markers_handle is None:
                        markers_handle = vis.scene.add_point_cloud(
                            "/markers",
                            points=pts,
                            colors=cols,
                            point_size=0.02,
                        )
                    else:
                        display.submit(
                            lambda h=markers_handle, p=pts.copy(), c=cols.copy(): (
                                setattr(h, "points", p),
                                setattr(h, "colors", c),
                            )
                        )
                display_ms = (time.perf_counter() - t_disp0) * 1e3

                mks_dict = dict(zip(settings.marker_names, augmented_markers))

                if first_sample:
                    t_ik0 = time.perf_counter()
                    q = solver.calibrate(mks_dict)
                    calibration_ms = (time.perf_counter() - t_ik0) * 1e3
                    state["calibration_ms"] = calibration_ms
                    human_model, human_data = solver.model, solver.data
                    state["human_model"] = human_model

                    t_viz0 = time.perf_counter()
                    if args.visualizer == "meshcat":
                        viz_human = MeshcatVisualizer(
                            human_model,
                            solver.collision_model,
                            solver.visual_model,
                        )
                        viz_human.initViewer(vis, open=False)
                        try:
                            vis["ref"].delete()
                        except Exception:
                            pass
                        viz_human.loadViewerModel("ref")
                        viz_human.viewer["/Background"].set_property(
                            "top_color", [1, 1, 1]
                        )
                        viz_human.viewer["/Background"].set_property(
                            "bottom_color", [0.65, 0.65, 0.65]
                        )
                    elif args.visualizer == "viser":
                        viz_human = ViserRobotVisualizer(
                            human_model,
                            solver.collision_model,
                            solver.visual_model,
                        )
                        viz_human.initViewer(viewer=vis)
                        viz_human.loadViewerModel(rootNodeName="ref")
                        viz_human.displayCollisions(False)
                        viz_human.displayVisuals(True)
                    else:
                        viz_human = None
                    viewer_setup_ms = (time.perf_counter() - t_viz0) * 1e3
                    state["viewer_setup_ms"] = viewer_setup_ms

                    first_sample = False
                else:
                    t_ik0 = time.perf_counter()
                    q = solver.step(mks_dict)
                    ik_ms.append((time.perf_counter() - t_ik0) * 1e3)

                t_disp1 = time.perf_counter()
                if viz_human is not None:
                    display.submit(
                        lambda qq=np.array(q, copy=True): viz_human.display(qq)
                    )
                display_ms += (time.perf_counter() - t_disp1) * 1e3
                display_ms_list.append(display_ms)

                t_save0 = time.perf_counter()
                if saver is not None:
                    marker_row = OrderedDict(Frame=idx)
                    for name, position in mks_dict.items():
                        marker_row[name + "_x"] = float(position[0])
                        marker_row[name + "_y"] = float(position[1])
                        marker_row[name + "_z"] = float(position[2])
                    saver.save_markers(marker_row)

                    if len(q) != len(settings.joint_angles_names):
                        raise ValueError(
                            f"Model has {len(q)} configuration variables but "
                            f"{len(settings.joint_angles_names)} joint angle names are defined"
                        )
                    saver.save_joint_angles(
                        OrderedDict(
                            zip(
                                settings.joint_angles_names,
                                (float(v) for v in q),
                            )
                        )
                    )
                    state["frames_written"] += 1
                save_ms_list.append((time.perf_counter() - t_save0) * 1e3)

                with frame_start_lock:
                    frame_t0 = frame_start_ts.pop(idx, None)
                done_ts = time.perf_counter()
                if frame_t0 is not None:
                    frame_latencies.append((done_ts - frame_t0) * 1e3)
                with span_lock:
                    if calib_done_ts[0] is None:
                        calib_done_ts[0] = done_ts
                    elif warm_done_ts[0] is None:
                        warm_done_ts[0] = done_ts
                    last_done_ts[0] = done_ts

                if state["frames_read"] % 100 == 0:
                    print(
                        f"  {state['frames_read']} frames read, "
                        f"{state['frames_written']} written",
                        flush=True,
                    )
            except Exception:
                LOGGER.exception(
                    "[ERROR] ik_worker failed on frame %s, skipping", idx
                )
                continue

    threads = [
        threading.Thread(target=fn, daemon=True)
        for fn in (reader, gpu_worker, ik_worker)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_first_to_last_s = None
    if first_read_ts[0] is not None and last_done_ts[0] is not None:
        total_first_to_last_s = last_done_ts[0] - first_read_ts[0]

    warm_steady_state_s = None
    if warm_done_ts[0] is not None and last_done_ts[0] is not None:
        warm_steady_state_s = last_done_ts[0] - warm_done_ts[0]

    timings = {
        "read": read_ms,
        "pose": pose_ms,
        "pose_parts": pose_parts,
        "reconstruct": reconstruct_ms,
        "ik": ik_ms,
        "display": display_ms_list,
        "save": save_ms_list,
        "frame_latency": frame_latencies,
        "total_first_to_last_s": total_first_to_last_s,
        "warm_steady_state_s": warm_steady_state_s,
    }
    return state, timings


def build_benchmark_stats(state, timings, fs):
    """Same shape/naming as the benchmark script's benchmark_stats, so both
    scripts' run_info.json / console output line up."""
    stats = {
        "frames_read": state["frames_read"],
        "frames_written": state["frames_written"],
    }
    if state["calibration_ms"] is not None:
        stats["calibration_s"] = state["calibration_ms"] / 1000.0
    if timings["total_first_to_last_s"] is not None:
        stats["post_calibration_span_s"] = timings["total_first_to_last_s"]
    if timings["warm_steady_state_s"] is not None:
        stats["steady_state_time_s"] = timings["warm_steady_state_s"]
        # First two completed frames are the one-off calibration frame and the
        # first-call/JIT-warmup frame; exclude them from the throughput count,
        # same convention as warm_done_ts marking the start of this window.
        n_processed = max(len(timings["frame_latency"]) - 2, 0)
        if timings["warm_steady_state_s"] > 0 and n_processed:
            stats["steady_state_fps"] = n_processed / timings["warm_steady_state_s"]

    # The first entry in ik_ms is the first real solver.step() call after
    # calibration -- for the acados MHE backend this is where the one-off
    # JIT/codegen cost of the first solve lands (calibration_ms already
    # isolates solver.calibrate() itself, so this is a separate spike).
    # Pull it out of the ik stats so median/p95/max reflect steady-state
    # solves; report it on its own instead of dropping it.
    ik_vals = list(timings["ik"])
    if ik_vals:
        stats["ik_first_call_compile_ms"] = ik_vals.pop(0)

    for key in ("read", "pose", "reconstruct", "display", "save"):
        summary = _timing_summary(timings[key])
        if summary is not None:
            stats[key] = summary
    ik_summary = _timing_summary(ik_vals)
    if ik_summary is not None:
        stats["ik"] = ik_summary

    pose_breakdown = {}
    for part_name in ("yolo", "h2d+pre", "nlf"):
        summary = _timing_summary(timings["pose_parts"][part_name])
        if summary is not None:
            pose_breakdown[part_name] = summary
    if pose_breakdown:
        stats["pose_breakdown"] = pose_breakdown

    frame_summary = _timing_summary(timings["frame_latency"])
    if frame_summary is not None:
        # NOTE: this is end-to-end latency (read-start -> ik-done) per frame,
        # not throughput -- in a pipelined run frames overlap, so 1/latency
        # is NOT fps and is intentionally not reported as one. Use
        # steady_state_fps (wall-clock frames-processed rate) for throughput.
        stats["frame_latency"] = frame_summary
    stats["dataset_fps"] = fs
    return stats


def print_benchmark_stats(stats):
    print(
        f"Timing summary (mean / median / p95 / max, ms; n = sample count):"
    )
    for name in ("read", "pose", "reconstruct", "ik", "display", "save"):
        if name in stats:
            s = stats[name]
            print(
                f"  {name:<12} {s['mean_ms']:7.1f} / {s['median_ms']:7.1f} / "
                f"{s['p95_ms']:7.1f} / {s['max_ms']:7.1f}    (n={s['n']})"
            )
    if "pose_breakdown" in stats:
        print("  pose breaks down as:")
        for name in ("yolo", "h2d+pre", "nlf"):
            if name in stats["pose_breakdown"]:
                s = stats["pose_breakdown"][name]
                print(
                    f"    {name:<10} {s['mean_ms']:7.1f} / {s['median_ms']:7.1f} / "
                    f"{s['p95_ms']:7.1f} / {s['max_ms']:7.1f}"
                )
    if "frame_latency" in stats:
        s = stats["frame_latency"]
        print(
            f"  {'frame_latency':<12} {s['mean_ms']:7.1f} / {s['median_ms']:7.1f} / "
            f"{s['p95_ms']:7.1f} / {s['max_ms']:7.1f}    (n={s['n']})"
        )
    if "calibration_s" in stats:
        print(f"\nOne-off model calibration: {stats['calibration_s']:.1f} s")
    if "ik_first_call_compile_ms" in stats:
        print(
            f"One-off ik first-call compile (e.g. acados codegen): "
            f"{stats['ik_first_call_compile_ms']:.1f} ms"
        )
    if "steady_state_fps" in stats:
        print(
            f"  -> steady-state throughput: {stats['steady_state_fps']:.2f} fps "
            f"(dataset is {stats['dataset_fps']} fps)"
        )
    print(
        "  ('ik' is the solver alone; 'pose' is YOLO + preprocess + NLF; "
        "'frame_latency' is end-to-end read -> ik-done per frame, NOT the "
        "inverse of throughput once stages overlap -- see steady-state "
        "throughput above for real fps)"
    )


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
        cam_params_path, video_paths, subject_path, out_dir = resolve_trial(
            args
        )
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
    world_R1_cam, world_T1_cam = load_world_transformation(
        cam_params_path, args.cameras[0]
    )

    if args.online:
        # --replay feeds recordings through the *online* path
        if args.replay:
            replay_dir = Path(args.replay)
            sources = [
                str(replay_dir / f"camera_{c}.mp4") for c in args.cameras
            ]
            missing = [s for s in sources if not Path(s).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"missing recordings for replay: {missing}"
                )
            camera_ids = list(args.cameras)
            LOGGER.info(
                "[CAP] replaying %d recordings from %s as live cameras",
                len(sources),
                replay_dir,
            )
        else:
            cameras = list_cameras()
            camera_ids = list(cameras.keys())
            sources = [None] * len(camera_ids)

        NUM_CAMERAS = len(camera_ids)
        FRAME_SHAPE = (H, W, 3)
        (
            camera_buffers,
            camera_timestamps,
            camera_locks,
            frame_counters,
            camera_barrier,
            stop_event,
        ) = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
        # Shared with the video writers so one toggle drives every recorder.
        saving_flag = Value("b", settings.record_on_start)
        # Replay sources hold their first frame until this is set.
        calibrated_event = MPEvent()

        record_paths = [None] * NUM_CAMERAS
        if settings.SAVE_VID:
            os.makedirs(settings.SAVE_DIR, exist_ok=True)
            record_paths = [
                os.path.join(settings.SAVE_DIR, f"camera_{c}.mkv")
                for c in camera_ids
            ]
            LOGGER.info("[CAP] recording video to %s", settings.SAVE_DIR)

        camera_processes = [
            Camera(
                camera_ids[i],
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
                calibrated_event=calibrated_event if args.replay else None,
            )
            for i in range(NUM_CAMERAS)
        ]

        online_height, online_weight, online_gender = load_subject(
            subject_path
        )
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

        processes = camera_processes + [pipeline]

        for p in processes:
            p.start()

        def _set_recording(on):
            saving_flag.value = on
            LOGGER.info("[KEY] recording %s", "started" if on else "stopped")

        hotkeys = TerminalHotkeys(
            {
                "s": lambda: _set_recording(True),
                "q": lambda: _set_recording(False),
            },
            logger=LOGGER,
        )
        with hotkeys:
            if hotkeys.active:
                LOGGER.info(
                    "[KEY] press 's' to start recording, 'q' to stop, "
                    "Ctrl-C to quit"
                )
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                stop_event.set()
                for process in processes:
                    process.stop() if hasattr(process, "stop") else None
                    process.join(timeout=2)

    else:  # offline mode

        # --- 1. INITIALISATION VISUALIZER ---
        if args.visualizer == "meshcat":
            vis = meshcat.Visualizer()
            LOGGER.info(
                f"[INFO] Meshcat visualizer available here: {vis.url()}"
            )
            vis_markers = vis["markers"]
        elif args.visualizer == "viser":
            vis = viser.ViserServer()
            LOGGER.info(
                "[INFO] Viser visualizer available here:"
                f" http://{vis.get_host()}:{vis.get_port()}"
            )
            vis.scene.add_grid(
                "/grid",
                width=10.0,
                height=10.0,
                position=(0.0, 0.0, 0.0),
            )
            vis_markers = None
        else:  # args.visualizer == "none"
            vis = None
            vis_markers = None

        paths = video_paths
        NUM_CAMERAS = len(paths)
        LOGGER.info(
            "Processing %d camera(s): %s",
            NUM_CAMERAS,
            ", ".join(str(p) for p in paths),
        )

        subject_height, subject_weight, subject_gender = load_subject(
            subject_path
        )

        src = OfflineVideoSource(paths=paths, size_wh=(W, H), loop=False)

        saver = None
        if not args.no_save:
            saver = CSVSaver(
                str(out_dir),
                markers_header=["Frame"] + list(settings.marker_names),
                joint_angles_header=list(settings.joint_angles_names),
            )
            LOGGER.info(
                "Writing markers.csv and joint_angles.csv to %s", out_dir
            )

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

        solver = HumanSolver(
            settings,
            gender=subject_gender,
            height=subject_height,
            weight=subject_weight,
            logger=LOGGER,
        )

        display = AsyncDisplay(logger=LOGGER)
        display.__enter__()

        try:
            state, timings = run_pipelined(
                src=src,
                est=est,
                args=args,
                settings=settings,
                projections=projections,
                world_R1_cam=world_R1_cam,
                world_T1_cam=world_T1_cam,
                solver=solver,
                saver=saver,
                display=display,
                vis=vis,
                vis_markers=vis_markers,
                NUM_CAMERAS=NUM_CAMERAS,
            )
        finally:
            src.release()
            display.close()
            if args.show_nlf:
                cv2.destroyAllWindows()

        human_model = state["human_model"]
        frames_read = state["frames_read"]
        frames_written = state["frames_written"]

        benchmark_stats = None
        if timings["frame_latency"] or timings["read"]:
            benchmark_stats = build_benchmark_stats(state, timings, settings.fs)
            print_benchmark_stats(benchmark_stats)

        if saver is not None:
            saver.close()
            run_info = {
                "participant": args.participant,
                "task": args.task,
                "cameras": list(args.cameras),
                "num_cameras": NUM_CAMERAS,
                "videos": [str(v) for v in paths],
                "fps": settings.fs,
                "ik": {
                    "type": settings.ik_type,
                    "mhe_backend": (
                        settings.mhe_backend
                        if settings.ik_type == "mhe"
                        else None
                    ),
                    "horizon_N": (
                        settings.N if settings.ik_type == "mhe" else None
                    ),
                    "cost_weights": (
                        list(settings.cost_weights)
                        if settings.ik_type == "mhe"
                        else None
                    ),
                    "mhe_profile": (
                        settings.mhe_profile
                        if settings.ik_type == "mhe"
                        else None
                    ),
                },
                "filter": {
                    "order": settings.order,
                    "cutoff_hz": settings.cutoff_freq,
                    "type": settings.filter_type,
                },
                "variant": run_variant(args.cameras),
                "subject": {
                    "height": subject_height,
                    "weight": subject_weight,
                    "gender": subject_gender,
                },
                "frames_read": frames_read,
                "frames_written": frames_written,
                "joint_angles_names": list(settings.joint_angles_names),
                "marker_names": list(settings.marker_names),
                "root_placement_rotation": np.asarray(
                    human_model.jointPlacements[1].rotation
                ).tolist(),
                "joint_names": [
                    human_model.names[i] for i in range(human_model.njoints)
                ],
                "joint_placements": [
                    np.asarray(
                        human_model.jointPlacements[i].translation
                    ).tolist()
                    for i in range(human_model.njoints)
                ],
                # NEW: full benchmark timing breakdown, so run_info.json alone
                # is enough to compare runs without re-parsing console logs.
                "benchmark": benchmark_stats,
            }
            with open(Path(out_dir) / "run_info.json", "w") as handle:
                json.dump(run_info, handle, indent=2)
        LOGGER.info(
            "Finished: %d frames read, %d rows written to %s",
            frames_read,
            frames_written,
            out_dir,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Run the RT-COSMIK pipeline live, or offline over one recorded"
            " trial."
        ),
        epilog=(
            TRIAL_CLI_EPILOG
            + "\nSweep trials with a shell loop; there is no separate batch"
            " script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--replay",
        default=None,
        metavar="DIR",
        help=(
            "run the ONLINE path against recordings in DIR "
            "(camera_<id>.mp4), paced as if live"
        ),
    )
    p.add_argument(
        "--online",
        action="store_true",
        help="Capture from live cameras instead of video files",
    )
    add_trial_arguments(p)
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Visualise only, write no CSV files",
    )
    p.add_argument(
        "--show-nlf",
        action="store_true",
        help=(
            "Show a live cv2 window with YOLO boxes + NLF 2D keypoints "
            "overlaid per camera, in both online and offline modes."
        ),
    )
    p.add_argument(
        "--visualizer",
        type=str,
        choices=["viser", "meshcat", "none"],
        default="viser",
        help=(
            "3D display backend for the human model + marker point cloud, "
            "in both online (ViewerProcess) and offline (ik_worker) modes. "
            "Defaults to viser."
        ),
    )
    args = p.parse_args()

    if args.online:
        set_start_method("spawn")

    main(args)