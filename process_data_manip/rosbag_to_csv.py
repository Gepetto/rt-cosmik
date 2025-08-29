#!/usr/bin/env python3
"""
Convert /joint_states from ROS bag(s) to CSV.

Output CSV format (per bag):
  timestamp, position.<joint>..., velocity.<joint>..., effort.<joint>...

Rules:
- timestamp = Header stamp if present, else bag receipt time; formatted UTC "YYYY-MM-DD HH:MM:SS.ffffff"
- Drop any joint whose name contains "finger" (case-insensitive).
- No extra columns.

Modes:
1) Single bag:
   --bag /path/to/file.bag --out /path/to/outdir [--subject SUBJ] [--task TASK]
   -> writes outdir[/SUBJ/TASK]/joint_states.csv

2) Batch scan:
   --root-in /root/of/bags --root-out /out/root
   Finds .bag/.mcap files and ROS2 bag directories (with metadata.yaml) recursively.
   Subject/task are inferred from path using SUBJECTS and TASK_SYNONYMS below.
   -> writes /out/root/<subject>/<task>/<bag_stem>/joint_states.csv
"""

from __future__ import annotations
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Set, List, Tuple, Dict

# ======= Your subject & task catalogs =======
SUBJECTS = [
    "Alessandro","Anais","Anastasia","Batiste","Bilal","Claire_","Clement","Flavie","Guilhem",
    "Kahina","Marie_M","Mathis","Maxime_","Mohamed","Nicolas","Zoe","Herbert","Emmanuelle"
]
# Canonical task names and simple path match synonyms
TASK_SYNONYMS: Dict[str, List[str]] = {
    "robot_sanding": ["robot_sanding", "sanding"],
    "robot_welding": ["robot_welding", "welding"],
}

# ======= Dependencies =======
try:
    from rosbags.highlevel import AnyReader
except Exception as e:
    raise SystemExit(
        "Missing dependency 'rosbags'. Install it in THIS Python env:\n"
        "  python -m pip install rosbags\n\n"
        f"Import error was: {e}"
    )

# ======= Time helpers =======

def header_stamp_to_ns(header: Any) -> Optional[int]:
    """Return timestamp in nanoseconds from a ROS1/ROS2 Header if present."""
    if header is None:
        return None
    stamp = getattr(header, 'stamp', None)
    if stamp is None:
        return None
    # ROS 2
    if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
        return int(stamp.sec) * 10**9 + int(stamp.nanosec)
    # ROS 1
    if hasattr(stamp, 'secs') and hasattr(stamp, 'nsecs'):
        return int(stamp.secs) * 10**9 + int(stamp.nsecs)
    return None

def canonical_time_str(bag_t_ns: int, header_t_ns: Optional[int]) -> str:
    """Choose header time if present else bag time; return UTC 'YYYY-mm-dd HH:MM:SS.ffffff'."""
    t_ns = int(header_t_ns) if header_t_ns is not None else int(bag_t_ns)
    t_sec = t_ns / 1e9
    return datetime.utcfromtimestamp(t_sec).strftime('%Y-%m-%d %H:%M:%S.%f')

# ======= Misc helpers =======

def as_list(x):
    """Coerce ROS array/NumPy/list/tuple/None into a plain Python list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        from array import array as pyarray
        if isinstance(x, pyarray):
            return list(x)
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    try:
        return list(x)
    except Exception:
        return [x]

def topic_conns(reader: AnyReader, topic: str):
    return [c for c in reader.connections if c.topic == topic]

def is_finger_joint(name: str) -> bool:
    return "finger" in name.lower()

def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def infer_subject_and_task(path: Path) -> Tuple[str, str]:
    """Infer (subject, task) from a path using SUBJECTS and TASK_SYNONYMS."""
    pnorm = normalize(str(path))
    # subject
    subj_found = "unknown_subject"
    for s in SUBJECTS:
        if normalize(s) in pnorm:
            subj_found = s
            break
    # task
    task_found = "unknown_task"
    for canonical, syns in TASK_SYNONYMS.items():
        for syn in syns:
            if normalize(syn) in pnorm:
                task_found = canonical
                break
        if task_found != "unknown_task":
            break
    return subj_found, task_found

def find_candidate_bags(root_in: Path) -> List[Path]:
    """Find ROS bag candidates: .bag/.mcap files and ROS2 bag directories (with metadata.yaml)."""
    candidates: List[Path] = []
    for p in root_in.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.bag', '.mcap'}:
            candidates.append(p)
        elif p.is_dir():
            meta = p / 'metadata.yaml'
            if meta.exists():  # ros2 bag dir
                candidates.append(p)
    return candidates

# ======= Core export =======

def export_joint_states(bagpath: Path, outfile: Path, topic: str = '/joint_states'):
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # -------- Pass 1: collect all NON-FINGER joint names seen in the bag --------
    joint_names: Set[str] = set()
    with AnyReader([bagpath]) as reader:
        conns = topic_conns(reader, topic)
        if not conns:
            print(f"[joint_states] Topic '{topic}' not found in {bagpath}. No file written.")
            return
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            names = as_list(getattr(msg, 'name', None))
            for n in names:
                if not is_finger_joint(n):
                    joint_names.add(n)

    # Deterministic column order
    names_sorted: List[str] = sorted(joint_names)
    pos_cols  = [f'position.{n}' for n in names_sorted]
    vel_cols  = [f'velocity.{n}' for n in names_sorted]
    eff_cols  = [f'effort.{n}'   for n in names_sorted]
    header = ['timestamp'] + pos_cols + vel_cols + eff_cols

    # -------- Pass 2: stream rows to CSV --------
    count = 0
    with AnyReader([bagpath]) as reader, outfile.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        writer.writeheader()

        conns = topic_conns(reader, topic)
        for conn, bag_t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            hdr = getattr(msg, 'header', None)
            h_ns = header_stamp_to_ns(hdr)

            row = {'timestamp': canonical_time_str(bag_t_ns, h_ns)}

            names     = as_list(getattr(msg, 'name', None))
            positions = as_list(getattr(msg, 'position', None))
            velocities= as_list(getattr(msg, 'velocity', None))
            efforts   = as_list(getattr(msg, 'effort', None))

            for n, p in zip(names, positions):
                if is_finger_joint(n): continue
                key = f'position.{n}'
                if key not in header: continue
                row[key] = p
            for n, v in zip(names, velocities):
                if is_finger_joint(n): continue
                key = f'velocity.{n}'
                if key not in header: continue
                row[key] = v
            for n, e in zip(names, efforts):
                if is_finger_joint(n): continue
                key = f'effort.{n}'
                if key not in header: continue
                row[key] = e

            writer.writerow(row)
            count += 1

    print(f"[joint_states] Wrote {count} rows to {outfile}")

# ======= Main =======

def main():
    ap = argparse.ArgumentParser(description="Export /joint_states to CSV (timestamp + robot data, no finger joints).")
    ap.add_argument("--bag", help="Path to .bag/.mcap file or ROS2 bag directory")
    ap.add_argument("--out", help="Output directory (CSV will be 'joint_states.csv')")
    ap.add_argument("--subject", help="Optional subject name to nest under output dir")
    ap.add_argument("--task", help="Optional task name to nest under output dir")
    ap.add_argument("--root-in", help="Batch mode: root folder to scan for bags")
    ap.add_argument("--root-out", help="Batch mode: output root folder")
    args = ap.parse_args()

    if args.root_in and args.root_out:
        # Batch mode
        root_in = Path(args.root_in)
        root_out = Path(args.root_out)
        bags = find_candidate_bags(root_in)
        if not bags:
            print(f"[batch] No bag candidates found under {root_in}")
            return
        print(f"[batch] Found {len(bags)} candidates under {root_in}")
        for bag in bags:
            subj, task = infer_subject_and_task(bag)
            
            # --- NEW CODE ---
            output_folder = Path("robot_data_csv")  # folder where all CSVs go
            output_folder.mkdir(exist_ok=True)
            outfile = output_folder / f"{subj}_{task}.csv"
            export_joint_states(bag, outfile)
        print("[batch] Done.")
        return

    # Single-bag mode (default)
    if not args.bag or not args.out:
        raise SystemExit("Provide either (--bag and --out) for single mode, or (--root-in and --root-out) for batch mode.")

    bagpath = Path(args.bag)
    outdir = Path(args.out)
    if args.subject:
        outdir = outdir / args.subject
    if args.task:
        outdir = outdir / args.task
    export_joint_states(bagpath, outdir)

if __name__ == "__main__":
    main()
