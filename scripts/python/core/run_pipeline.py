#!/usr/bin/env python3
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
import json

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
from rtcosmik.utils.mp_utils import create_camera_shared_ressources, create_pipeline_shared_ressources
from rtcosmik.utils.VideoReader import OfflineVideoSource
from rtcosmik.saver.csv_saver import CSVSaver
from rtcosmik.utils.dataset import (
    TRIAL_CLI_EPILOG, add_trial_arguments, load_subject, resolve_trial, run_variant)
from rtcosmik.model_weights import resolve_detector_engine
from rtcosmik.pipeline.pipeline import PipelineProcess

from multiprocessing import set_start_method
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
        cam_params_path = settings.cam_calib_path
        video_paths = subject_path = out_dir = None
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
        cameras = list_cameras()
        NUM_CAMERAS = len(cameras)
        FRAME_SHAPE = (H, W, 3)
        camera_buffers, camera_timestamps, camera_locks, frame_counters, camera_barrier, stop_event = create_camera_shared_ressources(NUM_CAMERAS, FRAME_SHAPE)
        results_queues = create_pipeline_shared_ressources()

        # Create camera processes
        camera_processes = [
            Camera(list(cameras.keys())[i], 
                camera_buffers[i], 
                camera_timestamps[i], 
                camera_locks[i], 
                frame_counters[i], 
                camera_barrier, 
                stop_event, 
                FRAME_SHAPE, 
                settings.fs, 
                settings.fourcc,)
            for i in range(NUM_CAMERAS)
        ]

        pipeline = PipelineProcess(
            settings=settings,
            frame_counters=frame_counters,
            camera_buffers=camera_buffers,
            camera_locks=camera_locks,
            timestamp_buffers=camera_timestamps,
            results_queues=results_queues,
            stop_event=stop_event,
            mtxs=mtxs,
            dists=dists,
            projections=projections,
            world_R1_cam=world_R1_cam,
            world_T1_cam=world_T1_cam,
            frame_shape=FRAME_SHAPE,
            num_cameras=NUM_CAMERAS,
        )

        # Imported here rather than at module scope: it depends on pynput, which
        # requires an X display, and offline runs must work headless.
        from rtcosmik.viewer.viewer import ViewerProcess

        viewer= ViewerProcess(
            settings=settings,
            results_queues=results_queues,
            stop_event=stop_event,
            num_cameras=NUM_CAMERAS,
        )

        processes = camera_processes + [pipeline, viewer]

        # Start processes
        for p in processes:
            p.start()

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

        while True:
            t0=time.perf_counter()
            frames = src.read()
            if frames is None:
                break
            frames_read += 1

            nlf_out, infer_ms, yres, boxes = est.estimate_from_frames(frames)

            views = extract_views(nlf_out, NUM_CAMERAS)
            p3d = reconstruct_3d(views, projections)
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

                vis_markers.set_object(
                    g.PointCloud(position=augmented_markers.T, color=colors, size=0.02)
                )

                mks_dict = dict(zip(settings.marker_names, augmented_markers))

                if first_sample:
                    q = solver.calibrate(mks_dict)
                    human_model, human_data = solver.model, solver.data

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

                    first_sample = False
                else:
                    q = solver.step(mks_dict)

                viz_human.display(q)

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
            t1=time.perf_counter()
            print(f"Time elapsed for treating one frame = {t1-t0} ms")

        src.release()
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
                    "mhe_max_iter": settings.mhe_max_iter if settings.ik_type == "mhe" else None,
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
    p.add_argument("--online", action="store_true",
                   help="Capture from live cameras instead of video files")
    add_trial_arguments(p)
    p.add_argument("--no-save", action="store_true", help="Visualise only, write no CSV files")
    args = p.parse_args()

    if args.online:
        set_start_method('spawn')

    main(args)