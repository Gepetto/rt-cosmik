import os
import time
import csv
import traceback
from collections import defaultdict
from multiprocessing import Process, Queue, Event, set_start_method

import cv2
import numpy as np

from rtcosmik.pose_estimator.pose_estimator import BatchPoseTrackerEstimator
from settings import Settings

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

settings = Settings()

# Default COMFI path (adapt if needed or override with env var COMFI_VIDEOS_ROOT)
COMFI_VIDEOS_ROOT = os.environ.get(
    "COMFI_VIDEOS_ROOT",
    "/home/msabbah/Desktop/comfi-examples/COMFI/videos"
)

# Tasks to benchmark
TASKS_OF_INTEREST = [
    "Lifting",
    "Screwing",
    "SideOverhead",
    "RobotPolishing",
    "RobotWelding",
    "Polishing",
]

# Cameras we expect per task
CAMERA_IDS = [0, 2]

DET_MODEL = settings.det_model_path
POSE_MODEL = settings.pose_model_path


# --------------------------------------------------------------------------- #
# Worker: processes one (participant, task) set of 4 videos in a separate proc
# --------------------------------------------------------------------------- #

def worker_fn(video_paths, det_model, pose_model, result_queue, stop_event):
    """
    Worker process:
    - Opens the given video paths (one per camera).
    - Warms up the pose estimator.
    - Loops over frames until one of the videos ends or stop_event is set.
    - For each step: read frames, run HPE, measure time.
    - Sends timing stats back through result_queue.
    """
    caps = []
    try:
        # Open videos
        caps = [cv2.VideoCapture(p) for p in video_paths]
        for p, cap in zip(video_paths, caps):
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {p}")

        num_cams = len(video_paths)

        # Warmup (size doesn't matter much, this is just to load models)
        tracker = BatchPoseTrackerEstimator(num_cams, det_model, pose_model)
        _ = tracker.estimate(
            [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(num_cams)]
        )

        num_batches = 0
        total_time = 0.0

        while not stop_event.is_set():
            frames = []
            # Read one frame per camera
            for cap in caps:
                ok, frame = cap.read()
                if not ok or frame is None:
                    # End of at least one video: stop benchmark for this task
                    stop_event.set()
                    break
                frames.append(frame)

            if stop_event.is_set():
                break

            # Measure HPE time for this multi-camera batch
            start = time.perf_counter()
            _ = tracker.estimate(frames)
            end = time.perf_counter()

            total_time += (end - start)
            num_batches += 1

        result_queue.put({
            "error": None,
            "num_batches": num_batches,
            "num_cams": num_cams,
            "total_time": total_time,
        })

    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put({
            "error": f"{e}",
            "traceback": tb,
            "num_batches": 0,
            "num_cams": len(video_paths),
            "total_time": 0.0,
        })
    finally:
        for cap in caps:
            cap.release()


class VideoBenchmarker:
    """
    Runs a benchmark for a specific (participant, task) combination:
    - Spawns one worker process with the given video paths.
    - Waits for the timing result.
    """
    def __init__(self, video_paths, det_model, pose_model):
        self.video_paths = video_paths
        self.det_model = det_model
        self.pose_model = pose_model
        self.result_queue = Queue()
        self.stop_event = Event()
        self.worker = Process(
            target=worker_fn,
            args=(video_paths, det_model, pose_model, self.result_queue, self.stop_event),
        )

    def run_benchmark(self):
        self.worker.start()
        result = self.result_queue.get()  # blocks until worker finishes
        self.stop_event.set()
        self.worker.join()
        return result


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def find_participants(comfi_videos_root):
    """Return sorted list of participant IDs (directory names)."""
    participants = []
    for name in os.listdir(comfi_videos_root):
        p_dir = os.path.join(comfi_videos_root, name)
        if os.path.isdir(p_dir):
            participants.append(name)
    return sorted(participants)


def build_video_paths_for_task(participant_id, task_name):
    """
    Build the list of camera video paths for a given participant and task.
    Returns a list of existing paths or an empty list if something is missing.
    """
    task_dir = os.path.join(COMFI_VIDEOS_ROOT, participant_id, task_name)
    if not os.path.isdir(task_dir):
        print(f"[SKIP] Participant {participant_id}, task {task_name}: missing directory {task_dir}")
        return []

    video_paths = []
    for cam_id in CAMERA_IDS:
        vp = os.path.join(task_dir, f"camera_{cam_id}.mp4")
        if not os.path.exists(vp):
            print(f"[SKIP] Participant {participant_id}, task {task_name}: missing {vp}")
            return []
        video_paths.append(vp)

    return video_paths


def main():
    print(f"Using COMFI videos root: {COMFI_VIDEOS_ROOT}")
    if not os.path.isdir(COMFI_VIDEOS_ROOT):
        raise RuntimeError(f"COMFI_VIDEOS_ROOT does not exist: {COMFI_VIDEOS_ROOT}")

    participants = find_participants(COMFI_VIDEOS_ROOT)
    print(f"Found {len(participants)} participants.")

    all_results = []
    per_task_results = defaultdict(list)

    for pid in participants:
        print(f"\n=== Participant {pid} ===")
        for task in TASKS_OF_INTEREST:
            video_paths = build_video_paths_for_task(pid, task)
            if not video_paths:
                continue  # missing data, already logged

            print(f"  -> Benchmarking task {task} with {len(video_paths)} cameras")

            benchmarker = VideoBenchmarker(video_paths, DET_MODEL, POSE_MODEL)
            result = benchmarker.run_benchmark()

            if result["error"] is not None:
                print(f"    [ERROR] {result['error']}")
                if "traceback" in result:
                    print(result["traceback"])
                continue

            num_batches = result["num_batches"]
            num_cams = result["num_cams"]
            total_time = result["total_time"]

            if num_batches == 0 or total_time <= 0:
                print(f"    [WARN] No frames processed for {pid} - {task}")
                continue

            total_frames = num_batches * num_cams
            avg_time_per_batch = total_time / num_batches
            avg_time_per_frame = total_time / total_frames
            fps_total = total_frames / total_time
            fps_per_camera = num_batches / total_time

            # Log per participant / task
            print(
                f"    {pid} | {task}: "
                f"batches={num_batches}, cams={num_cams}, "
                f"total_time={total_time:.3f}s, "
                f"avg_batch={avg_time_per_batch*1000:.3f} ms, "
                f"avg_frame={avg_time_per_frame*1000:.3f} ms, "
                f"FPS_total={fps_total:.2f}, "
                f"FPS_per_cam={fps_per_camera:.2f}"
            )

            entry = {
                "participant": pid,
                "task": task,
                "num_cams": num_cams,
                "num_batches": num_batches,
                "total_frames": total_frames,
                "total_time": total_time,
                "avg_time_per_batch": avg_time_per_batch,
                "avg_time_per_frame": avg_time_per_frame,
                "fps_total": fps_total,
                "fps_per_camera": fps_per_camera,
            }
            all_results.append(entry)
            per_task_results[task].append(entry)

    if not all_results:
        print("No successful benchmark results.")
        return

    # ------------------------------------------------------------------ #
    # Global statistics
    # ------------------------------------------------------------------ #
    global_total_time = sum(e["total_time"] for e in all_results)
    global_total_frames = sum(e["total_frames"] for e in all_results)
    global_fps = global_total_frames / global_total_time
    global_avg_time_per_frame = global_total_time / global_total_frames

    print("\n=== Global statistics across all participants and tasks ===")
    print(f"Total frames (all cams, all tasks): {global_total_frames}")
    print(f"Total time: {global_total_time:.3f} s")
    print(f"Average time per frame: {global_avg_time_per_frame*1000:.3f} ms")
    print(f"Overall FPS (all cams combined): {global_fps:.2f}")

    # ------------------------------------------------------------------ #
    # Per-task statistics
    # ------------------------------------------------------------------ #
    print("\n=== Per-task statistics ===")
    for task, entries in per_task_results.items():
        t_total_time = sum(e["total_time"] for e in entries)
        t_total_frames = sum(e["total_frames"] for e in entries)
        if t_total_frames == 0 or t_total_time <= 0:
            continue
        t_fps = t_total_frames / t_total_time
        t_avg_time_frame = t_total_time / t_total_frames
        print(
            f"Task {task}: "
            f"participants={len(entries)}, "
            f"frames={t_total_frames}, "
            f"time={t_total_time:.3f}s, "
            f"avg_frame={t_avg_time_frame*1000:.3f} ms, "
            f"FPS={t_fps:.2f}"
        )

    # ------------------------------------------------------------------ #
    # Save detailed CSV
    # ------------------------------------------------------------------ #
    csv_path = f"benchmark_mmpose_batched_all_results_{len(CAMERA_IDS)}cams.csv"

    fieldnames = [
        "participant",
        "task",
        "num_cams",
        "num_batches",
        "total_frames",
        "total_time",
        "avg_time_per_batch",
        "avg_time_per_frame",
        "fps_total",
        "fps_per_camera",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in all_results:
            writer.writerow(e)

    print(f"\nDetailed results written to: {csv_path}")


if __name__ == "__main__":
    # Spawn is generally safer with CUDA / large objects
    set_start_method("spawn", force=True)
    main()
