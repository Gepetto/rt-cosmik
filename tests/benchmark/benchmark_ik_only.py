#!/usr/bin/env python3
import sys
import time
import json
import argparse
import numpy as np
import torch
import pinocchio as pin 
import example_robot_data as robex
from pathlib import Path
from collections import deque

# Ensure internal source roots are visible
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtcosmik.config_loader import settings
from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
from rtcosmik.triangulation.triangulation import triangulate_points
from rtcosmik.filtering.iir import IIR
from rtcosmik.human_model.model_utils import (
    scale_human_model, 
    mks_registration, 
    recalibrate_marker_frames_in_joint_space
)
from rtcosmik.ik.ik import RT_IK, RT_SWIKA_FATROP, RT_SWIKA_ACADOS


def load_2d_trajectory(filepath: str):
    """Loads recorded 2D keypoints and reconstructs camera array matrices."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Could not find '{filepath}'. Run the main script first to generate it.")
        
    with open(filepath, "r") as f:
        raw_data = json.load(f)
    
    # Reconstruct numpy arrays per camera frame
    trajectory = [
        [np.array(cam_kp, dtype=np.float32) if cam_kp is not None else None for cam_kp in frame]
        for frame in raw_data
    ]
    return trajectory


def main():
    parser = argparse.ArgumentParser(description="Geometric Backend Pipeline Benchmark")
    parser.add_argument("--input-data", type=str, default="nlf_2d_keypoints.json", help="Path to saved 2D keypoints JSON")
    args = parser.parse_args()

    # ----------------------------------------------------
    # 1. Load Calibration & Data Streams
    # ----------------------------------------------------
    print(f"Loading recorded 2D keypoints from: {args.input_data}...")
    trajectory = load_2d_trajectory(args.input_data)
    num_frames = len(trajectory)
    
    if num_frames == 0:
        raise ValueError("The provided data payload contains 0 frames.")

    mtxs, dists, projections, _, _ = load_camera_parameters(settings.cam_calib_path)
    world_R1_cam, world_T1_cam = load_world_transformation(settings.cam_calib_path)

    print(f"\n--- BENCHMARK CONFIGURATION ---")
    print(f"IK Type      : {settings.ik_type}")
    if settings.ik_type == 'mhe':
        print(f"MHE Backend  : {settings.mhe_backend}")
    print(f"Total Frames : {num_frames}\n")

    # ----------------------------------------------------
    # 2. Filter & Model Initialization
    # ----------------------------------------------------
    p3d_buffer = deque(maxlen=settings.N)
    num_channel = 3 * len(settings.marker_names)
    iir_filter = IIR(num_channel=num_channel, sampling_frequency=settings.fs)
    iir_filter.add_filter(order=settings.order, cutoff=settings.cutoff_freq, filter_type=settings.filter_type)

    human = robex.human.HumanLoader(
        height=settings.human_height, 
        weight=settings.human_weight, 
        gender=settings.human_gender
    ).robot
    human_model = human.model

    # Run a quick initial triangulation using frame 0 to perform model calibration
    init_p3d = triangulate_points(trajectory[0], mtxs, dists, projections)
    init_p3d_np = torch.from_numpy(init_p3d).to(dtype=torch.float32)
    init_p3d_world = np.array([np.dot(world_R1_cam, pt) + world_T1_cam for pt in init_p3d_np])
    base_mks_dict = dict(zip(settings.marker_names, init_p3d_world))

    # Scale & Register
    human_model = scale_human_model(human_model, base_mks_dict, gender=settings.human_gender, subject_height=settings.human_height)
    human_model = mks_registration(human_model, base_mks_dict, gender=settings.human_gender, subject_height=settings.human_height)

    # ----------------------------------------------------
    # 3. Solver Setup / Cold Calibration
    # ----------------------------------------------------
    q = pin.neutral(human_model)
    
    if settings.ik_type == 'sbs':
        omega = {key: 1 for key in settings.keys_to_track_list}
        ik_class = RT_IK(human_model, base_mks_dict, q, settings.keys_to_track_list, settings.dt, omega)
        q = ik_class.solve_ik_sample_casadi()
        ik_class._q0 = q
        human_model = recalibrate_marker_frames_in_joint_space(human_model, q, base_mks_dict, settings.marker_names)
        ik_class = RT_IK(human_model, base_mks_dict, q, settings.keys_to_track_list, settings.dt, omega)

    elif settings.ik_type == 'mhe':
        x_array = np.zeros((human_model.nq + human_model.nv, settings.N))
        x_array[6, :] = 1.0
        u_array = np.zeros((human_model.nv, settings.N))
        
        deque_lstm_dict = deque(maxlen=settings.N)
        for _ in range(settings.N):
            deque_lstm_dict.append(base_mks_dict)

        omega = {key: 1 for key in settings.keys_to_track_list}
        ik_class = RT_IK(human_model, base_mks_dict, q, settings.keys_to_track_list, settings.dt, omega)
        q = ik_class.solve_ik_sample_casadi()
        ik_class._q0 = q
        human_model = recalibrate_marker_frames_in_joint_space(human_model, q, base_mks_dict, settings.marker_names)

        if settings.mhe_backend == 'acados':
            ik_class = RT_SWIKA_ACADOS(human_model, settings.keys_to_track_list, settings.N, settings.dt, export_dir=settings.acados_export_dir, acados_source_dir=settings.acados_source_dir)
        else:
            ik_class = RT_SWIKA_FATROP(human_model, settings.keys_to_track_list, settings.N, code=settings.ik_code)

    print("Calibration finished. Running geometric pipeline loop...")

    # ----------------------------------------------------
    # 4. Monitored Execution Loop
    # ----------------------------------------------------
    tri_history = []
    fil_history = []
    ik_history = []
    total_history = []

    first_sample = True

    for f in range(num_frames):
        keypoints_list = trajectory[f]
        t_frame_start = time.perf_counter()

        # --- STEP 4.1: Triangulation & World Frame Transform ---
        t_tri_start = time.perf_counter()
        p3d = triangulate_points(keypoints_list, mtxs, dists, projections)
        p3d_np = torch.from_numpy(p3d).to(dtype=torch.float32)
        p3d_in_world = np.array([np.dot(world_R1_cam, point) + world_T1_cam for point in p3d_np])
        
        if first_sample:
            for _ in range(settings.N):
                p3d_buffer.append(p3d_in_world)
        else:
            p3d_buffer.append(p3d_in_world)
        t_tri_end = time.perf_counter()

        # --- STEP 4.2: Signal Filtering ---
        t_fil_start = time.perf_counter()
        if len(p3d_buffer) == settings.N:
            p3d_buffer_array = np.array(p3d_buffer)
            filtered_p3d_buffer = iir_filter.filter(np.reshape(p3d_buffer_array, (settings.N, 3 * len(settings.marker_names))))
            filtered_p3d_buffer = np.reshape(filtered_p3d_buffer, (settings.N, len(settings.marker_names), 3))
            augmented_markers = filtered_p3d_buffer[-1]
        else:
            augmented_markers = p3d_in_world  # Fallback if buffer sizing mismatch occurs
        t_fil_end = time.perf_counter()

        mks_dict = dict(zip(settings.marker_names, augmented_markers))

        # --- STEP 4.3: Kinematic Tracking Optimizations (IK) ---
       
        if first_sample:
            # First sample is already handled during the script setup phase above
            first_sample = False
            t_ik_end = time.perf_counter()
            continue 

        t_ik_start = time.perf_counter()

        if settings.ik_type == 'sbs':
            ik_class._dict_m = mks_dict
            t_ik_start = time.perf_counter()
            q = ik_class.solve_ik_sample_quadprog()
            t_ik_end = time.perf_counter()
            ik_class._q0 = q
        elif settings.ik_type == 'mhe':
            deque_lstm_dict.append(mks_dict)
            array_data = np.array([np.hstack([d[marker] for marker in settings.keys_to_track_list]) for d in deque_lstm_dict]).T
            #t_ik_start = time.perf_counter()
            x_array, u_array = ik_class.solve(x_array, u_array, array_data, x_array[:, -1], settings.cost_weights, settings.dt)
            t_ik_end = time.perf_counter()
            q = pin.neutral(human_model)
            q[:] = np.array(x_array[:human_model.nq, -1]).flatten()
        

        # Calculate durations in milliseconds
        t_frame_end = time.perf_counter()
        
        tri_history.append((t_tri_end - t_tri_start) * 1000)
        fil_history.append((t_fil_end - t_fil_start) * 1000)
        ik_history.append((t_ik_end - t_ik_start) * 1000)
        total_history.append((t_frame_end - t_frame_start) * 1000)

    # ----------------------------------------------------
    # 5. Granular Performance Reporting
    # ----------------------------------------------------
    print("\n" + "="*20 + " PIPELINE BREAKDOWN " + "="*20)
    print(f"Total Processed Frames: {len(total_history)}")
    print(f"Triangulation Time    : mean {np.mean(tri_history):.2f} ms | median {np.median(tri_history):.2f} ms | max {np.max(tri_history):.2f} ms")
    print(f"IIR Filter Time       : mean {np.mean(fil_history):.2f} ms | median {np.median(fil_history):.2f} ms | max {np.max(fil_history):.2f} ms")
    print(f"IK Solver Time        : mean {np.mean(ik_history):.2f} ms | median {np.median(ik_history):.2f} ms | max {np.max(ik_history):.2f} ms")
    print("-" * 59)
    print(f"TOTAL BACKEND TIME    : mean {np.mean(total_history):.2f} ms | median {np.median(total_history):.2f} ms | max {np.max(total_history):.2f} ms")
    print(f"Estimated Throughput  : {1000.0 / np.mean(total_history):.1f} Hz")
    print("="*59)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json",  type=str, default="nlf_2d_keypoints.json")
    args = p.parse_args()
    main()