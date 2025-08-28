#!/usr/bin/env python3
import os
import csv
import cv2
from typing import Dict, List

# =========================
# USER SETTINGS
# =========================
ROOT_DIR   = "output"  # <- change me (this folder contains the subject folders)
MOUV_DIR   = "mouv"                    # fixed subfolder name between subject and task
VIDEO_NAME = "camera_0.mp4"            # video file name

SUBJECTS = ["Alessandro"]  # or set your own list

TASKS = ["lifting","overhead","crouch_object","robot_sanding", "robot_welding"]  # add more if needed

OUTPUT_DIR = os.path.join(ROOT_DIR, "_annotations_frames")  # where per-subject CSVs will be saved
OVERWRITE_EXISTING = False  # if True, you can overwrite an already-saved frame number for a task

# =========================
# UTILITIES
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def csv_path_for_subject(subject: str) -> str:
    ensure_dir(OUTPUT_DIR)
    return os.path.join(OUTPUT_DIR, f"{subject}_frames.csv")

def load_subject_record(subject: str, tasks: List[str]) -> Dict[str, str]:
    """
    Return a dict {task: frame_str_or_empty}. If CSV exists, load it; else initialize blanks.
    """
    path = csv_path_for_subject(subject)
    record = {t: "" for t in tasks}
    if not os.path.exists(path):
        return record

    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                row = rows[0]
                for t in tasks:
                    if t in row and row[t] is not None:
                        record[t] = row[t]
    except Exception as e:
        print(f"[WARN] Could not read existing CSV for {subject}: {e}")
    return record

def save_subject_record(subject: str, record: Dict[str, str], tasks: List[str]) -> None:
    """
    Write a one-row CSV with header = tasks, values = frame numbers (or empty).
    """
    path = csv_path_for_subject(subject)
    try:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tasks)
            writer.writeheader()
            writer.writerow({t: record.get(t, "") for t in tasks})
        print(f"[INFO] Saved CSV: {path}")
    except Exception as e:
        print(f"[ERROR] Could not save CSV for {subject}: {e}")

def video_path(subject: str, task: str) -> str:
    return os.path.join(ROOT_DIR, subject, MOUV_DIR, task, VIDEO_NAME)

# =========================
# INTERACTIVE REVIEW
# =========================
INSTRUCTIONS = (
    "Controls: [any key]=next frame | b=back one frame | y=save frame & next video | "
    "q=skip video | ESC=quit"
)

def draw_overlay(frame, text_lines, origin=(12, 28), line_height=24):
    """Overlay multiple lines of text on the frame."""
    x, y = origin
    for line in text_lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += line_height

def review_video_for_task(subject: str, task: str, record: Dict[str, str]) -> bool:
    """
    Show the video frame-by-frame.
    Press 'y' to save current frame number for this task and move on.
    Returns False if user pressed ESC to quit everything; True otherwise.
    """
    path = video_path(subject, task)
    if not os.path.exists(path):
        print(f"[WARN] Missing video: {path}")
        return True  # continue

    if record.get(task, "") and not OVERWRITE_EXISTING:
        print(f"[INFO] {subject}/{task} already has a saved frame ({record[task]}). Skipping (OVERWRITE_EXISTING=False).")
        return True

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {path}")
        return True

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    win = "Video Review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"\n[REVIEW] {subject} — {task}")
    print(INSTRUCTIONS)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Reached end of video without saving. Skipping.")
            break

        # current frame index is POS_FRAMES after read, so subtract 1
        frame_idx = int(max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))

        overlay_lines = [
            f"Subject: {subject} | Task: {task}",
            f"Frame: {frame_idx}" + (f"/{total_frames - 1}" if total_frames > 0 else ""),
            INSTRUCTIONS,
            "Press 'y' anytime to SAVE this frame number for this video."
        ]
        draw_overlay(frame, overlay_lines)

        cv2.imshow(win, frame)
        key = cv2.waitKey(0) & 0xFF  # wait for a key at each frame

        if key == 27:  # ESC
            print("[INFO] ESC pressed. Quitting.")
            cap.release()
            cv2.destroyWindow(win)
            return False  # signal to quit everything

        if key in (ord('q'), ord('Q')):  # skip this video
            print("[INFO] Skipping this video.")
            break

        if key in (ord('b'), ord('B')):  # go back one frame
            back_to = max(0, frame_idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, back_to)
            continue

        if key in (ord('y'), ord('Y')):  # save current frame
            record[task] = str(frame_idx)
            print(f"[SAVED] {subject} — {task}: frame {frame_idx}")
            break

        # Any other key => step to next frame
        # (nothing else to do; next loop iteration reads next frame)

    cap.release()
    cv2.destroyWindow(win)
    return True

# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    for subject in SUBJECTS:
        print(f"\n========== {subject} ==========")
        record = load_subject_record(subject, TASKS)

        for task in TASKS:
            cont = review_video_for_task(subject, task, record)
            # Save after each task so progress isn't lost
            save_subject_record(subject, record, TASKS)
            if not cont:  # user pressed ESC
                print("[INFO] Exiting by user request.")
                return

    print("\n[DONE] All subjects/tasks processed.")

if __name__ == "__main__":
    main()
