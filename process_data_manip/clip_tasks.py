#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clip long task videos into short web-friendly previews + posters.

Layout assumed:
  <root>/<subject>/<task>/camera_0.mp4        (configurable via --camera)

Outputs (saved NEXT TO the source):
  <root>/<subject>/<task>/camera_0_preview.mp4
  <root>/<subject>/<task>/camera_0_poster.jpg

Two-step workflow:

1) SELECT (interactive):
   - If OpenCV GUI is available: scrub, I/O, save.
   - If GUI is NOT available: headless mode using ffplay preview + terminal input.

   python clip_tasks.py select \
     --root /path/to/root \
     --subject 1847 \
     --camera camera_0.mp4 \
     --tasks bolting bolting_sat crouch crouch_object hitting hitting_sat jump lifting \
             lifting_fast lower overhead overhead_front robot_sanding robot_welding \
             sanding sanding_sat sit_to_stand squat static upper walk walk_front \
             welding welding_sat \
     --map 1847_clips.json

   Headless tips:
     - The script can open ffplay for you (press 'q' in the player to quit),
       then you type start/end times in the terminal.
     - Times accepted: '12.3', '00:00:12.300', '2:05.2', or relative: '+0.5' (from last start).

2) RENDER (ffmpeg cuts clip + poster):
   python clip_tasks.py render \
     --map 1847_clips.json \
     --height 720 --fps 24 --crf 23 --preset veryfast \
     --poster-offset 0.3 --mute \
     --clip-suffix _preview --poster-suffix _poster

Requires: ffmpeg in PATH. For GUI mode only: opencv-python with GUI backend.
"""
import argparse, json, os, subprocess, sys, re
from typing import Dict, Any, List, Optional

# ----------------- utilities -----------------
def hhmmss(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600); s -= 3600 * h
    m = int(s // 60);   s -= 60 * m
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def parse_time(s: str, base: float = 0.0) -> float:
    """Parse time strings:
       - seconds: '12.3'
       - mm:ss(.ms): '2:05.2'
       - hh:mm:ss(.ms): '00:00:12.300'
       - relative: '+0.5' (adds to base)
    """
    s = s.strip()
    if s.startswith("+"):
        return max(0.0, base + float(s[1:]))
    if re.match(r"^\d+(\.\d+)?$", s):
        return float(s)
    parts = s.split(":")
    parts = [float(x) for x in parts]
    if len(parts) == 2:
        mm, ss = parts
        return mm*60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh*3600 + mm*60 + ss
    raise ValueError(f"Unrecognized time format: {s}")

def run(cmd: List[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_ffmpeg(cmd: List[str]) -> None:
    print("ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def load_map(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_map(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved selections -> {path}")

def has_cv2_gui() -> bool:
    """Return True if cv2 with GUI is available, else False."""
    try:
        import cv2  # noqa
        # Some headless builds import but raise on window creation;
        # we'll attempt a tiny create/destroy test.
        try:
            cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("__test__")
            return True
        except Exception:
            return False
    except Exception:
        return False

# ----------------- SELECT: GUI -----------------
def select_gui(args, mapping: Dict[str, Any]) -> None:
    import cv2

    def draw_hud(frame, cur_t, cur_s, cur_e, t_now, t_total, playing):
        font = cv2.FONT_HERSHEY_SIMPLEX
        hud = [
            f"Task: {cur_t}",
            f"Time: {hhmmss(t_now)} / {hhmmss(t_total)}  [{'PLAY' if playing else 'PAUSE'}]",
            f"IN: {('--' if cur_s is None else hhmmss(cur_s))}   OUT: {('--' if cur_e is None else hhmmss(cur_e))}",
            "Space=Play/Pause  I=IN  O=OUT  C=Clear  A/D=±1s  J/L=±1f  ,/.=±5f",
            "S=Save&Next  N=Skip  Q=Quit"
        ]
        y = 24
        for line in hud:
            cv2.putText(frame, line, (10, y), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
            y += 20
        return frame

    for task in args.tasks:
        info = mapping[task]
        src = info["input"]
        if not os.path.exists(src):
            print(f"[WARN] Missing file: {src}")
            continue

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERR] Could not open {src}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames > 0 else 0.0

        playing = False
        cur_frame = 0
        start_s = info.get("start", None)
        end_s   = info.get("end", None)

        win = "Select clip (Q=quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)

        def seek(fidx: int) -> int:
            fidx = max(0, min(total_frames-1, int(fidx)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            return fidx

        cur_frame = seek(0)

        while True:
            if playing:
                ret, frame = cap.read()
                if not ret:
                    playing = False
                    cur_frame = total_frames - 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    cur_frame += 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
                ret, frame = cap.read()
                if not ret:
                    break

            t_now = cur_frame / fps
            frame = draw_hud(frame, task, start_s, end_s, t_now, duration, playing)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if key in (ord('q'), ord('Q')):
                cv2.destroyWindow(win)
                save_map(args.map, mapping)
                return
            if key == 32:                      # space
                playing = not playing
            elif key in (ord('i'), ord('I')):  # IN
                start_s = t_now
                if end_s is not None and end_s <= start_s:
                    end_s = None
            elif key in (ord('o'), ord('O')):  # OUT
                end_s = t_now
                if start_s is not None and end_s <= start_s:
                    end_s = start_s + max(0.1, 1.0 / fps)
            elif key in (ord('c'), ord('C')):  # clear
                start_s = None; end_s = None
            elif key in (ord('s'), ord('S')):  # save & next
                if start_s is None or end_s is None:
                    print("Set both IN and OUT before saving.")
                else:
                    info["start"] = float(start_s)
                    info["end"]   = float(end_s)
                    info["poster_offset"] = float(args.poster_offset)
                    print(f"[OK] {task}: {hhmmss(start_s)} -> {hhmmss(end_s)}")
                    break
            elif key in (ord('n'), ord('N')):  # skip
                print(f"[SKIP] {task}")
                break
            elif key in (ord('a'), ord('A')):  # -1s
                cur_frame = seek(cur_frame - int(1 * fps)); playing = False
            elif key in (ord('d'), ord('D')):  # +1s
                cur_frame = seek(cur_frame + int(1 * fps)); playing = False
            elif key in (ord('j'), ord('J')):  # -1f
                cur_frame = seek(cur_frame - 1); playing = False
            elif key in (ord('l'), ord('L')):  # +1f
                cur_frame = seek(cur_frame + 1); playing = False
            elif key == ord(','):              # -5f
                cur_frame = seek(cur_frame - 5); playing = False
            elif key == ord('.'):              # +5f
                cur_frame = seek(cur_frame + 5); playing = False

        cap.release()
        cv2.destroyWindow(win)

    save_map(args.map, mapping)
    print("Selection finished (GUI).")

# ----------------- SELECT: HEADLESS (ffplay + terminal) -----------------
def select_headless(args, mapping: Dict[str, Any]) -> None:
    print("\nOpenCV GUI not available. Using headless selector with ffplay.")
    print("Instructions:")
    print("  • I will open the video in ffplay with a time overlay.")
    print("  • Watch, press ‘q’ to close the player, then type the START and END times here.")
    print("  • Examples: 12.5  |  00:00:12.500  |  2:05.2  |  +0.6 (relative from start)\n")

    drawtext = "drawtext=text='%{pts\\:hms}':x=10:y=h-th-10:fontsize=24:fontcolor=white:box=1:boxcolor=0x00000088"

    for task in args.tasks:
        info = mapping[task]
        src = info["input"]
        if not os.path.exists(src):
            print(f"[WARN] Missing file: {src}")
            continue

        # Preview full video
        print(f"\n=== {task} ===")
        print(f"Source: {src}")
        try:
            run(["ffplay", "-autoexit", "-hide_banner", "-loglevel", "warning",
                 "-vf", drawtext, src])
        except subprocess.CalledProcessError:
            pass  # user may close with q; ffplay returns non-zero sometimes

        # Enter times
        last_start = info.get("start", 0.0) or 0.0
        while True:
            start_in = input("Start time (e.g., 12.5 or 00:00:12.5): ").strip()
            try:
                start = parse_time(start_in, base=last_start)
                break
            except Exception as e:
                print("  Invalid time:", e)

        while True:
            end_in = input("End time (e.g., 16.3, 00:00:16.300, or +3.0): ").strip()
            try:
                end = parse_time(end_in, base=start)
                if end <= start:
                    print("  End must be > start.")
                    continue
                break
            except Exception as e:
                print("  Invalid time:", e)

        # Optional preview of the selected range
        ans = input("Preview this range? [y/N]: ").strip().lower()
        if ans == "y":
            dur = end - start
            try:
                run(["ffplay", "-autoexit", "-hide_banner", "-loglevel", "warning",
                     "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-vf", drawtext, src])
            except subprocess.CalledProcessError:
                pass

        info["start"] = float(start)
        info["end"] = float(end)
        info["poster_offset"] = float(args.poster_offset)
        save_map(args.map, mapping)
        print(f"[OK] {task}: {hhmmss(start)} -> {hhmmss(end)}")

    print("Selection finished (headless).")

# ----------------- RENDER -----------------
def render_outputs(args):
    mapping = load_map(args.map)

    # optional CSV merge
    if args.csv and os.path.exists(args.csv):
        import csv
        with open(args.csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row["task"].strip()
                mapping.setdefault(t, {})
                mapping[t]["input"] = row.get("input", mapping[t].get("input"))
                mapping[t]["start"] = float(row["start"])
                mapping[t]["end"]   = float(row["end"])
                if row.get("poster_offset"):
                    mapping[t]["poster_offset"] = float(row["poster_offset"])

    for task, info in mapping.items():
        src   = info.get("input")
        start = info.get("start")
        end   = info.get("end")
        if not src or start is None or end is None:
            print(f"[SKIP] {task}: missing src/start/end")
            continue
        if not os.path.exists(src):
            print(f"[SKIP] {task}: file not found -> {src}")
            continue

        dur = max(0.1, float(end) - float(start))
        poster_offset = float(info.get("poster_offset", args.poster_offset))

        out_dir = os.path.dirname(src)
        base, _ = os.path.splitext(os.path.basename(src))

        out_mp4 = os.path.join(out_dir, f"{base}{args.clip_suffix}.mp4")
        out_jpg = os.path.join(out_dir, f"{base}{args.poster_suffix}.jpg")

        vf = [f"scale=-2:{args.height}", "format=yuv420p"]
        if args.pad16by9:
            vf.append("pad=iw:trunc(iw/16*9):(ow-iw)/2:(oh-ih)/2")
        vf_str = ",".join(vf)

        # video
        cmd_vid = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", src,
            "-t", f"{dur:.3f}",
            "-r", str(args.fps),
            "-vf", vf_str,
            "-c:v", "libx264",
            "-preset", args.preset,
            "-crf", str(args.crf),
            "-movflags", "+faststart",
        ]
        if args.mute:
            cmd_vid += ["-an"]
        cmd_vid += [out_mp4]

        # poster
        cmd_img = [
            "ffmpeg", "-y",
            "-ss", f"{start + poster_offset:.3f}",
            "-i", src,
            "-vframes", "1",
            "-vf", f"scale=-2:{args.height}",
            "-q:v", "2",
            out_jpg
        ]

        try:
            run_ffmpeg(cmd_vid)
            run_ffmpeg(cmd_img)
            print(f"[OK] {task}: {out_mp4} | poster -> {out_jpg}")
        except subprocess.CalledProcessError as e:
            print(f"[ERR] {task}: ffmpeg failed ({e.returncode})")

# ----------------- main SELECT wrapper -----------------
def select_main(args):
    # Prepare mapping entries with computed input paths
    mapping = load_map(args.map)
    for task in args.tasks:
        default_src = os.path.join(args.root, str(args.subject), task, args.camera)
        if task not in mapping:
            mapping[task] = {
                "subject": str(args.subject),
                "task": task,
                "camera": args.camera,
                "input": default_src,
                "start": None,
                "end": None,
                "poster_offset": args.poster_offset
            }
        else:
            mapping[task].setdefault("input", default_src)
            mapping[task].setdefault("subject", str(args.subject))
            mapping[task].setdefault("camera", args.camera)
            mapping[task].setdefault("poster_offset", args.poster_offset)

    # Choose mode
    if args.headless or not has_cv2_gui():
        select_headless(args, mapping)
    else:
        try:
            select_gui(args, mapping)
        except Exception as e:
            print(f"[WARN] GUI selection failed ({e}). Falling back to headless mode.")
            select_headless(args, mapping)

# ----------------- CLI -----------------
def main():
    p = argparse.ArgumentParser(description="Create short task clips + poster images next to source videos.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("select", help="Pick IN/OUT for each task video (GUI if possible, else headless).")
    sp.add_argument("--root", required=True, help="Root folder with <subject>/<task>/camera_X.mp4")
    sp.add_argument("--subject", required=True, help="Subject id (e.g., 1847)")
    sp.add_argument("--camera", default="camera_0.mp4", help="Video filename inside each task folder (default: camera_0.mp4)")
    sp.add_argument("--tasks", nargs="+", required=True, help="Task names to process")
    sp.add_argument("--map", default="task_clips.json", help="JSON file to store selections")
    sp.add_argument("--poster-offset", type=float, default=0.3, help="Seconds after start for poster frame")
    sp.add_argument("--headless", action="store_true", help="Force headless (ffplay+terminal) even if GUI is available")
    sp.set_defaults(func=select_main)

    rp = sub.add_parser("render", help="Cut clips and posters using ffmpeg based on the saved JSON (or a CSV).")
    rp.add_argument("--map", default="task_clips.json", help="JSON produced by 'select'")
    rp.add_argument("--csv", default="", help="Optional CSV with columns: task,input,start,end,poster_offset")
    rp.add_argument("--height", type=int, default=720, help="Target height (keeps aspect with width=-2)")
    rp.add_argument("--fps", type=int, default=24, help="Output fps")
    rp.add_argument("--crf", type=int, default=23, help="x264 quality (lower = better/bigger)")
    rp.add_argument("--preset", default="veryfast", help="x264 speed preset")
    rp.add_argument("--poster-offset", type=float, default=0.3, help="Seconds after start for poster frame")
    rp.add_argument("--mute", action="store_true", help="Remove audio (smaller files; good for muted previews)")
    rp.add_argument("--pad16by9", action="store_true", help="Pad to a 16:9 canvas if needed")
    rp.add_argument("--clip-suffix", default="_preview", help="Suffix for video filename (before .mp4)")
    rp.add_argument("--poster-suffix", default="_poster", help="Suffix for poster filename (before .jpg)")
    rp.set_defaults(func=render_outputs)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
