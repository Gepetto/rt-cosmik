import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_log_summary(log_path):
    """Extract per-camera joint, marker, root orientation, and free-flyer RMSE
    from one eval log (as produced by compare_to_mocap.py)."""
    filename = os.path.basename(log_path)
    name_part = os.path.splitext(filename)[0]
    if "_" in name_part:
        subject, task = name_part.split("_", 1)
    else:
        subject, task = "Unknown", name_part

    summary_data = defaultdict(dict)
    try:
        with open(log_path, "r") as f:
            content = f.read()

        # 1. Joint and Marker Errors: "1cam          13.24 deg       124.4 mm"
        joint_marker_pattern = r"(\d+)cam\s+([\d.]+)\s*deg\s+([\d.]+)\s*mm"
        for cams, j_err, m_err in re.findall(joint_marker_pattern, content):
            summary_data[int(cams)]["joint_err_deg"] = float(j_err)
            summary_data[int(cams)]["marker_err_mm"] = float(m_err)

        # 2. Total Root Orientation Error: "4cam ... root orientation error: 47.96 deg"
        orient_pattern = (
            r"(\d+)cam\s+root frame rotation removed:.*?root orientation error:\s*([\d.]+)\s*deg"
        )
        for cams, rot_err in re.findall(orient_pattern, content):
            summary_data[int(cams)]["root_orient_err_deg"] = float(rot_err)

        # 3. Free-flyer Translation RMSE Table (X, Y, Z, and mean)
        ff_block = re.search(
            r"Free-flyer translation RMSE.*?\n(.*?)\n\s*-+\n(.*?)\n\s*-+\n\s*mean\s+(.*?)\n",
            content,
            re.DOTALL,
        )
        if ff_block:
            header_line = ff_block.group(1)
            body = ff_block.group(2)
            mean_line = ff_block.group(3)

            cams_list = [int(c) for c in re.findall(r"(\d+)cam", header_line)]

            x_match = re.search(r"Freeflyer_X\[m\]\s+([\d.\s]+)", body)
            y_match = re.search(r"Freeflyer_Y\[m\]\s+([\d.\s]+)", body)
            z_match = re.search(r"Freeflyer_Z\[m\]\s+([\d.\s]+)", body)
            mean_vals = [float(v) for v in mean_line.split() if re.match(r"^[\d.]+$", v)]

            if x_match:
                for cam, val in zip(cams_list, [float(v) for v in x_match.group(1).split()]):
                    summary_data[cam]["ff_x_mm"] = val
            if y_match:
                for cam, val in zip(cams_list, [float(v) for v in y_match.group(1).split()]):
                    summary_data[cam]["ff_y_mm"] = val
            if z_match:
                for cam, val in zip(cams_list, [float(v) for v in z_match.group(1).split()]):
                    summary_data[cam]["ff_z_mm"] = val
            for cam, val in zip(cams_list, mean_vals):
                summary_data[cam]["ff_mean_mm"] = val

    except Exception:
        pass

    return subject, task, summary_data


def print_table_header():
    print(
        f"{'Cams':<5} | {'Joint(deg)':<10} | {'Marker(mm)':<10} | {'Orient_Total':<12} | {'FF_X(mm)':<9} | {'FF_Y(mm)':<9} | {'FF_Z(mm)':<9} | {'FF_Mean':<9}"
    )
    print("-" * 90)


def print_table_row(cams, data_dict):
    j_err_str = f"{sum(data_dict['joint_errs'])/len(data_dict['joint_errs']):.2f}" if data_dict['joint_errs'] else "N/A"
    m_err_str = f"{sum(data_dict['marker_errs'])/len(data_dict['marker_errs']):.2f}" if data_dict['marker_errs'] else "N/A"
    o_err_str = f"{sum(data_dict['orient_errs'])/len(data_dict['orient_errs']):.2f}" if data_dict['orient_errs'] else "N/A"
    x_str = f"{sum(data_dict['ff_x'])/len(data_dict['ff_x']):.2f}" if data_dict['ff_x'] else "N/A"
    y_str = f"{sum(data_dict['ff_y'])/len(data_dict['ff_y']):.2f}" if data_dict['ff_y'] else "N/A"
    z_str = f"{sum(data_dict['ff_z'])/len(data_dict['ff_z']):.2f}" if data_dict['ff_z'] else "N/A"
    ff_mean_str = f"{sum(data_dict['ff_mean'])/len(data_dict['ff_mean']):.2f}" if data_dict['ff_mean'] else "N/A"

    print(
        f"{cams:<5} | {j_err_str:<10} | {m_err_str:<10} | {o_err_str:<12} | {x_str:<9} | {y_str:<9} | {z_str:<9} | {ff_mean_str:<9}"
    )


def plot_rmse_grouped_by_task(task_totals, metric_key, ylabel, title, output_dir):
    """Plots a single grouped bar chart with tasks on the X-axis 
    and side-by-side bars for each camera setup in different colors."""
    os.makedirs(output_dir, exist_ok=True)

    tasks = sorted(task_totals.keys())
    if not tasks:
        return

    all_cams = sorted({cams for task in tasks for cams in task_totals[task].keys()})
    if not all_cams:
        return

    x = np.arange(len(tasks))
    num_cams = len(all_cams)
    total_group_width = 0.8
    bar_width = total_group_width / num_cams

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 2.5), 6))

    all_heights = []

    for i, cams in enumerate(all_cams):
        heights = []
        for task in tasks:
            vals = task_totals[task].get(cams, {}).get(metric_key, [])
            mean_val = sum(vals) / len(vals) if vals else 0.0
            heights.append(mean_val)
            if mean_val > 0:
                all_heights.append(mean_val)

        # Offset each camera setup's bar within the task group
        offset = (i - (num_cams - 1) / 2) * bar_width
        rects = ax.bar(x + offset, heights, width=bar_width, label=f"{cams}cam")

        # Add data labels on top of each bar
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_xlabel("Task")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Camera Setup", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if all_heights:
        lo, hi = min(all_heights), max(all_heights)
        span = hi - lo
        pad = span * 0.25 if span > 0 else (hi * 0.1 if hi != 0 else 1.0)
        ax.set_ylim(max(0, lo - pad), hi + pad * 1.5)

    fig.tight_layout()
    outpath = os.path.join(output_dir, f"{metric_key}_by_task.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def aggregate_raw_metrics(root_dir=".", plot_dir="rmse_plots"):
    """Aggregate metrics directly from a run's logs/ directory
    (one log per subject/task, e.g. as written by run_eval.sh — no horizon-N sweep)."""
    data_store = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {
                    "joint_errs": [],
                    "marker_errs": [],
                    "orient_errs": [],
                    "ff_x": [],
                    "ff_y": [],
                    "ff_z": [],
                    "ff_mean": [],
                }
            )
        )
    )

    log_files = glob.glob(os.path.join(root_dir, "logs", "*.log"))
    if not log_files:
        log_files = glob.glob(os.path.join(root_dir, "**/logs/*.log"), recursive=True)

    for lf in log_files:
        subj, task, log_metrics = parse_log_summary(lf)
        for num_cams, metrics in log_metrics.items():
            for key, target in [
                ("joint_err_deg", "joint_errs"),
                ("marker_err_mm", "marker_errs"),
                ("root_orient_err_deg", "orient_errs"),
                ("ff_x_mm", "ff_x"),
                ("ff_y_mm", "ff_y"),
                ("ff_z_mm", "ff_z"),
                ("ff_mean_mm", "ff_mean"),
            ]:
                if key in metrics:
                    data_store[subj][task][num_cams][target].append(metrics[key])

    if not data_store:
        print(f"No logs found under {os.path.join(root_dir, 'logs')}.")
        return

    make_metrics_store = lambda: {
        "joint_errs": [], "marker_errs": [], "orient_errs": [],
        "ff_x": [], "ff_y": [], "ff_z": [], "ff_mean": []
    }

    subj_totals = defaultdict(lambda: defaultdict(make_metrics_store))
    task_totals = defaultdict(lambda: defaultdict(make_metrics_store))
    grand_totals = defaultdict(make_metrics_store)

    print("\n" + "=" * 90)
    print(" DETAILED RUN METRICS BY SUBJECT AND TASK")
    print("=" * 90)

    for subject in sorted(data_store.keys()):
        for task in sorted(data_store[subject].keys()):
            print("\n" + "-" * 90)
            print(f" SUBJECT: {subject}  |  TASK: {task}")
            print("-" * 90)
            print_table_header()

            cam_data = data_store[subject][task]
            for cams in sorted(cam_data.keys()):
                d = cam_data[cams]
                print_table_row(cams, d)

                for k in d.keys():
                    subj_totals[subject][cams][k].extend(d[k])
                    task_totals[task][cams][k].extend(d[k])
                    grand_totals[cams][k].extend(d[k])

    print("\n\n" + "=" * 90)
    print(" SUMMARY METRICS (MEANS ACROSS SUBJECTS & TASKS)")
    print("=" * 90)

    print("\n" + "#" * 90)
    print(" 1. SUBJECT MEANS (Averaged across all tasks for each subject)")
    print("#" * 90)
    for subject in sorted(subj_totals.keys()):
        print(f"\n>>> Subject {subject} Summary <<<")
        print_table_header()
        for cams in sorted(subj_totals[subject].keys()):
            print_table_row(cams, subj_totals[subject][cams])

    print("\n" + "#" * 90)
    print(" 2. TASK MEANS (Averaged across all subjects for each task)")
    print("#" * 90)
    for task in sorted(task_totals.keys()):
        print(f"\n>>> Task {task} Summary <<<")
        print_table_header()
        for cams in sorted(task_totals[task].keys()):
            print_table_row(cams, task_totals[task][cams])

    print("\n" + "#" * 90)
    print(" 3. OVERALL GRAND MEAN (Averaged across all subjects and tasks)")
    print("#" * 90)
    print_table_header()
    for cams in sorted(grand_totals.keys()):
        print_table_row(cams, grand_totals[cams])

    print("\n" + "#" * 90)
    print(" 4. GROUPED BAR CHARTS BY TASK")
    print("#" * 90)
    plot_rmse_grouped_by_task(task_totals, "joint_errs", "Joint angle RMSE (deg)", "Joint Angle RMSE by Task", plot_dir)
    plot_rmse_grouped_by_task(task_totals, "ff_mean", "Free-flyer translation RMSE (mm)", "Free-flyer Translation RMSE by Task", plot_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".", help="Run output directory (containing a logs/ subfolder), e.g. output/$run_dir")
    parser.add_argument("--plot-dir", default="rmse_plots", help="Directory to save RMSE bar charts")
    args = parser.parse_args()

    aggregate_raw_metrics(args.root_dir, plot_dir=args.plot_dir)