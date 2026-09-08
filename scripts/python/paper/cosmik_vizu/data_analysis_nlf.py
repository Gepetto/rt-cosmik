#!/usr/bin/env python3
"""Fusion, alignment and temporal filtering for the 43-point NLF dataset.

The two NLF triangulations (camera pairs 0-2 and 4-6) are fused using the
per-camera uncertainties.  A single rigid transform is then estimated from
the reliable NLF/MoCap marker correspondences and applied to all 43 NLF
points.  Finally, an independent constant-velocity Kalman filter and an
optional zero-phase Butterworth filter smooth every point trajectory.

This is an offline analysis script: MoCap is used for alignment and metrics,
but never blended into the NLF output positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


NLF_MARKERS = [
    "RASI", "LASI", "RPSI", "LPSI",
    "C7", "T11", "T6", "RSHO", "LSHO", "RELB", "LELB", "RMELB",
    "LMELB", "RWRI", "LWRI", "RMWRI", "LMWRI",
    "RTHU", "LTHU", "RMID", "LMID", "RPIN", "LPIN",
    "RKNE", "LKNE", "RMKNE", "LMKNE", "RANK", "LANK", "RMANK",
    "LMANK", "R5MHD", "L5MHD", "RTOE", "LTOE", "RHEE", "LHEE",
    "Nose", "Head", "REar", "LEar", "REye", "LEye",
]

# Deliberately excludes T6/T11, hands and face: their physical MoCap markers
# are not at exactly the same anatomical locations as the selected SMPL-X
# vertices.  Keep this list stable for comparable Kabsch results.
KABSCH_MARKERS = [
    "RASI", "LASI", "RPSI", "LPSI",
    "C7",
    "RSHO", "LSHO",
    "RELB", "LELB", "RMELB", "LMELB",
    "RWRI", "LWRI", "RMWRI", "LMWRI",
    "RKNE", "LKNE", "RMKNE", "LMKNE",
    "RANK", "LANK", "RMANK", "LMANK",
    "R5MHD", "L5MHD", "RTOE", "LTOE", "RHEE", "LHEE",
]

NLF_TO_MOCAP = {
    "RASI": "r.ASIS_study", "LASI": "L.ASIS_study",
    "RPSI": "r.PSIS_study", "LPSI": "L.PSIS_study",
    "C7": "C7_study",
    "RSHO": "r_shoulder_study", "LSHO": "L_shoulder_study",
    "RELB": "r_lelbow_study", "LELB": "L_lelbow_study",
    "RMELB": "r_melbow_study", "LMELB": "L_melbow_study",
    "RWRI": "r_lwrist_study", "LWRI": "L_lwrist_study",
    "RMWRI": "r_mwrist_study", "LMWRI": "L_mwrist_study",
    "RKNE": "r_knee_study", "LKNE": "L_knee_study",
    "RMKNE": "r_mknee_study", "LMKNE": "L_mknee_study",
    "RANK": "r_ankle_study", "LANK": "L_ankle_study",
    "RMANK": "r_mankle_study", "LMANK": "L_mankle_study",
    "R5MHD": "r_5meta_study", "L5MHD": "L_5meta_study",
    "RTOE": "r_toe_study", "LTOE": "L_toe_study",
    "RHEE": "r_calc_study", "LHEE": "L_calc_study",
}


def read_xyz(path: Path, markers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    frame_column = "frame" if "frame" in df.columns else "Frame"
    frames = df[frame_column].to_numpy()
    xyz = np.stack(
        [df[[f"{name}_X", f"{name}_Y", f"{name}_Z"]].to_numpy(float)
         for name in markers],
        axis=1,
    )
    return frames, xyz


def read_uncertainties(path: Path, markers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return (
        df["frame"].to_numpy(),
        df[[f"{name}_uncertainty" for name in markers]].to_numpy(float),
    )


def read_mocap(path: Path, marker_map: dict[str, str]) -> np.ndarray:
    df = pd.read_csv(path)
    return np.stack(
        [df[[f"{marker_map[name]}_x", f"{marker_map[name]}_y",
             f"{marker_map[name]}_z"]].to_numpy(float)
         for name in KABSCH_MARKERS],
        axis=1,
    ) / 1000.0


def read_all_mocap_markers(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read every complete XYZ triplet from the MoCap CSV, in metres."""
    df = pd.read_csv(path)
    markers: list[str] = []
    for column in df.columns:
        if "_" not in column:
            continue
        name, axis = column.rsplit("_", 1)
        if axis.lower() == "x" and all(
            f"{name}_{candidate}" in df.columns for candidate in ("x", "y", "z")
        ):
            markers.append(name)
    xyz = np.stack(
        [df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(float)
         for name in markers],
        axis=1,
    ) / 1000.0
    return xyz, markers


def uncertainties_to_variance(values: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(values, float)
    if np.any(values < 0):
        raise ValueError("Uncertainties must be non-negative")
    return values**2 if kind == "std" else values


def fuse_by_uncertainty(
    points_02: np.ndarray,
    points_46: np.ndarray,
    unc_0: np.ndarray,
    unc_2: np.ndarray,
    unc_4: np.ndarray,
    unc_6: np.ndarray,
    uncertainty_kind: str,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse two triangulations and return positions plus their variance.

    Each stereo-pair variance is conservatively represented by the mean of
    its two per-camera variances.  Independent estimates are combined using
    inverse-variance weighting.
    """
    var_02 = 0.5 * (
        uncertainties_to_variance(unc_0, uncertainty_kind)
        + uncertainties_to_variance(unc_2, uncertainty_kind)
    )
    var_46 = 0.5 * (
        uncertainties_to_variance(unc_4, uncertainty_kind)
        + uncertainties_to_variance(unc_6, uncertainty_kind)
    )

    valid_02 = np.isfinite(points_02).all(axis=2) & np.isfinite(var_02)
    valid_46 = np.isfinite(points_46).all(axis=2) & np.isfinite(var_46)
    weight_02 = np.where(valid_02, 1.0 / np.maximum(var_02, eps), 0.0)
    weight_46 = np.where(valid_46, 1.0 / np.maximum(var_46, eps), 0.0)
    weight_sum = weight_02 + weight_46

    fused = np.full_like(points_02, np.nan, dtype=float)
    valid = weight_sum > 0
    numerator = (
        weight_02[..., None] * np.nan_to_num(points_02)
        + weight_46[..., None] * np.nan_to_num(points_46)
    )
    fused[valid] = numerator[valid] / weight_sum[valid, None]

    fused_variance = np.full_like(weight_sum, np.nan, dtype=float)
    fused_variance[valid] = 1.0 / weight_sum[valid]
    return fused, fused_variance


def kabsch_global(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one rigid transform aligning source (T,K,3) to target."""
    valid = np.isfinite(source).all(axis=2) & np.isfinite(target).all(axis=2)
    x = source[valid]
    y = target[valid]
    if len(x) < 3:
        raise ValueError("Kabsch requires at least three valid correspondences")
    x_center = x.mean(axis=0)
    y_center = y.mean(axis=0)
    u, _, vt = np.linalg.svd((y - y_center).T @ (x - x_center))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = y_center - rotation @ x_center
    return rotation, translation


def apply_transform(points: np.ndarray, rotation: np.ndarray,
                    translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def kalman_filter_points(
    measurements: np.ndarray,
    measurement_variance: np.ndarray,
    dt: float,
    acceleration_std: float,
    minimum_measurement_std: float,
) -> np.ndarray:
    """Independent constant-velocity Kalman filter for arbitrary K points."""
    frames, marker_count, _ = measurements.shape
    identity3 = np.eye(3)
    transition = np.block([
        [identity3, dt * identity3],
        [np.zeros((3, 3)), identity3],
    ])
    observation = np.block([identity3, np.zeros((3, 3))])
    gain_shape = np.eye(6)
    g = np.vstack([0.5 * dt**2 * identity3, dt * identity3])
    process_covariance = acceleration_std**2 * (g @ g.T)

    state = np.zeros((marker_count, 6), dtype=float)
    covariance = np.tile(np.eye(6), (marker_count, 1, 1))
    initialized = np.zeros(marker_count, dtype=bool)
    output = np.full_like(measurements, np.nan, dtype=float)

    for frame in range(frames):
        for marker in range(marker_count):
            measurement = measurements[frame, marker]
            valid = np.isfinite(measurement).all()
            if not initialized[marker]:
                if not valid:
                    continue
                state[marker, :3] = measurement
                initialized[marker] = True
                output[frame, marker] = measurement
                continue

            state[marker] = transition @ state[marker]
            covariance[marker] = (
                transition @ covariance[marker] @ transition.T
                + process_covariance
            )
            if valid:
                variance = measurement_variance[frame, marker]
                variance = max(
                    float(variance) if np.isfinite(variance) else 0.0,
                    minimum_measurement_std**2,
                )
                measurement_covariance = variance * identity3
                innovation = measurement - observation @ state[marker]
                innovation_covariance = (
                    observation @ covariance[marker] @ observation.T
                    + measurement_covariance
                )
                kalman_gain = np.linalg.solve(
                    innovation_covariance,
                    observation @ covariance[marker],
                ).T
                state[marker] += kalman_gain @ innovation
                residual = gain_shape - kalman_gain @ observation
                covariance[marker] = (
                    residual @ covariance[marker] @ residual.T
                    + kalman_gain @ measurement_covariance @ kalman_gain.T
                )
            output[frame, marker] = state[marker, :3]
    return output


def lowpass(points: np.ndarray, frequency: float, cutoff: float,
            order: int) -> np.ndarray:
    if cutoff <= 0:
        return points.copy()
    if cutoff >= frequency / 2:
        raise ValueError("Low-pass cutoff must be below the Nyquist frequency")
    if not np.isfinite(points).all():
        raise ValueError("Butterworth filtering requires finite trajectories")
    sos = butter(order, cutoff / (frequency / 2), btype="low", output="sos")
    return sosfiltfilt(sos, points, axis=0)


def write_xyz(path: Path, frames: np.ndarray, points: np.ndarray,
              markers: list[str]) -> None:
    columns: dict[str, np.ndarray] = {"frame": frames}
    for index, name in enumerate(markers):
        columns[f"{name}_X"] = points[:, index, 0]
        columns[f"{name}_Y"] = points[:, index, 1]
        columns[f"{name}_Z"] = points[:, index, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(path, index=False)


def position_errors(estimate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    distances = np.linalg.norm(estimate - reference, axis=2)
    return float(np.nanmean(distances)), float(np.sqrt(np.nanmean(distances**2)))


def visualize_markers(
    frames: np.ndarray,
    points_02: np.ndarray,
    points_46: np.ndarray,
    corrected: np.ndarray,
    mocap: np.ndarray,
    step: int,
    interval_ms: int,
) -> None:
    """Animate raw triangulations, corrected NLF points and MoCap markers."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    step = max(int(step), 1)
    displayed_frames = np.arange(0, len(frames), step)
    datasets = (points_02, points_46, corrected, mocap)
    finite_points = np.concatenate(
        [values.reshape(-1, 3)[np.isfinite(values).all(axis=2).reshape(-1)]
         for values in datasets],
        axis=0,
    )
    lower = np.percentile(finite_points, 0.5, axis=0)
    upper = np.percentile(finite_points, 99.5, axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.55 * np.max(upper - lower)

    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.set_box_aspect((1, 1, 1))

    scatter_02 = axis.scatter([], [], [], s=15, c="tab:red", alpha=0.55,
                              label="Triangulation brute cams 0-2")
    scatter_46 = axis.scatter([], [], [], s=15, c="tab:orange", alpha=0.55,
                              label="Triangulation brute cams 4-6")
    scatter_corrected = axis.scatter([], [], [], s=22, c="tab:green", alpha=0.9,
                                     label="NLF fusionné/corrigé")
    scatter_mocap = axis.scatter([], [], [], s=18, c="black", alpha=0.75,
                                 label="MoCap")
    title = axis.set_title("")
    axis.legend(loc="upper right")

    def set_scatter(scatter, values: np.ndarray) -> None:
        valid = np.isfinite(values).all(axis=1)
        points = values[valid]
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    def update(animation_index: int):
        frame_index = int(displayed_frames[animation_index])
        set_scatter(scatter_02, points_02[frame_index])
        set_scatter(scatter_46, points_46[frame_index])
        set_scatter(scatter_corrected, corrected[frame_index])
        set_scatter(scatter_mocap, mocap[frame_index])
        title.set_text(
            f"Frame NLF {frames[frame_index]} ({frame_index + 1}/{len(frames)})"
        )
        return scatter_02, scatter_46, scatter_corrected, scatter_mocap, title

    animation = FuncAnimation(
        fig, update, frames=len(displayed_frames), interval=interval_ms,
        blit=False, repeat=True,
    )
    # Keep a live reference until the interactive window is closed.
    fig._nlf_animation = animation
    plt.tight_layout()
    plt.show()


def plot_marker_comparisons(
    frames: np.ndarray,
    points_02: np.ndarray,
    points_46: np.ndarray,
    corrected: np.ndarray,
    mocap_all: np.ndarray,
    mocap_marker_names: list[str],
    frequency: float,
    selected_markers: list[str],
    output_path: Path,
) -> None:
    """Write one XYZ comparison page per NLF marker to a PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    unknown = [name for name in selected_markers if name not in NLF_MARKERS]
    if unknown:
        raise ValueError(f"Unknown NLF markers requested for plots: {unknown}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(len(frames), dtype=float) / frequency
    colors = {
        "02": "tab:red", "46": "tab:orange", "corrected": "tab:green",
        "mocap": "black",
    }
    with PdfPages(output_path) as pdf:
        for marker_name in selected_markers:
            nlf_index = NLF_MARKERS.index(marker_name)
            mocap_name = NLF_TO_MOCAP.get(marker_name)
            mocap_index = (
                mocap_marker_names.index(mocap_name)
                if mocap_name in mocap_marker_names else None
            )
            figure, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
            figure.suptitle(f"NLF / MoCap — {marker_name}")
            for coordinate, axis in enumerate(axes):
                axis.plot(
                    time, points_02[:, nlf_index, coordinate],
                    color=colors["02"], linewidth=0.7, alpha=0.75,
                    label="Triangulation brute cams 0-2",
                )
                axis.plot(
                    time, points_46[:, nlf_index, coordinate],
                    color=colors["46"], linewidth=0.7, alpha=0.75,
                    label="Triangulation brute cams 4-6",
                )
                axis.plot(
                    time, corrected[:, nlf_index, coordinate],
                    color=colors["corrected"], linewidth=1.2,
                    label="NLF fusionné/corrigé/filtré",
                )
                if mocap_index is not None:
                    axis.plot(
                        time, mocap_all[:, mocap_index, coordinate],
                        color=colors["mocap"], linewidth=1.0, alpha=0.8,
                        label=f"MoCap ({mocap_name})",
                    )
                axis.set_ylabel(f"{'XYZ'[coordinate]} [m]")
                axis.grid(True, alpha=0.25)
            axes[-1].set_xlabel("Temps [s]")
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="upper center", ncol=2,
                          bbox_to_anchor=(0.5, 0.955), frameon=False)
            figure.tight_layout(rect=(0, 0, 1, 0.91))
            pdf.savefig(figure)
            plt.close(figure)


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "output" / "nlf"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--frequency", type=float, default=40.0)
    parser.add_argument(
        "--uncertainty-kind", choices=("std", "variance"), default="std",
        help="Interpretation of the NLF uncertainty values (default: std)",
    )
    parser.add_argument("--acceleration-std", type=float, default=3.0,
                        help="Kalman process acceleration standard deviation in m/s^2")
    parser.add_argument("--minimum-measurement-std", type=float, default=0.005)
    parser.add_argument("--cutoff", type=float, default=4.0,
                        help="Butterworth cutoff in Hz; use 0 to disable")
    parser.add_argument("--filter-order", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--visualize", action="store_true",
                        help="Animate raw triangulations, corrected NLF and MoCap")
    parser.add_argument("--visualization-step", type=int, default=1,
                        help="Display every Nth frame")
    parser.add_argument("--visualization-interval", type=int, default=25,
                        help="Animation interval in milliseconds")
    parser.add_argument("--plot", action="store_true",
                        help="Save temporal XYZ comparison plots as a PDF")
    parser.add_argument("--plot-output", type=Path, default=None,
                        help="Comparison PDF path (default: DATA_DIR/nlf_marker_plots.pdf)")
    parser.add_argument(
        "--plot-markers", default="all",
        help="Comma-separated NLF marker names, or 'all' (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data_dir
    paths = {
        "points_02": data / "triangulated_nofilter_1012_Lifting_cams0_2.csv",
        "points_46": data / "triangulated_nofilter_1012_Lifting_cams4_6.csv",
        "unc_0": data / "nlf_uncertainties_1012_Lifting_camera_0.csv",
        "unc_2": data / "nlf_uncertainties_1012_Lifting_camera_2.csv",
        "unc_4": data / "nlf_uncertainties_1012_Lifting_camera_4.csv",
        "unc_6": data / "nlf_uncertainties_1012_Lifting_camera_6.csv",
        "mocap": data / "mocap" / "mocap_downsampled_to_40hz.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    frames, points_02 = read_xyz(paths["points_02"], NLF_MARKERS)
    frames_46, points_46 = read_xyz(paths["points_46"], NLF_MARKERS)
    uncertainties = {}
    uncertainty_frames = {}
    for camera in (0, 2, 4, 6):
        uncertainty_frames[camera], uncertainties[camera] = read_uncertainties(
            paths[f"unc_{camera}"], NLF_MARKERS
        )
    if not np.array_equal(frames, frames_46) or any(
        not np.array_equal(frames, uncertainty_frames[camera])
        for camera in uncertainty_frames
    ):
        raise ValueError("NLF triangulations and uncertainty files are not frame-aligned")

    mocap = read_mocap(paths["mocap"], NLF_TO_MOCAP)
    mocap_all, mocap_marker_names = read_all_mocap_markers(paths["mocap"])
    if len(mocap) != len(frames):
        raise ValueError(
            f"NLF/MoCap length mismatch: {len(frames)} versus {len(mocap)} frames"
        )

    fused, fused_variance = fuse_by_uncertainty(
        points_02, points_46,
        uncertainties[0], uncertainties[2], uncertainties[4], uncertainties[6],
        args.uncertainty_kind,
    )
    marker_indices = [NLF_MARKERS.index(name) for name in KABSCH_MARKERS]
    fused_correspondences = fused[:, marker_indices]
    rotation, translation = kabsch_global(fused_correspondences, mocap)
    fused_aligned = apply_transform(fused, rotation, translation)

    filtered = kalman_filter_points(
        fused_aligned, fused_variance,
        dt=1.0 / args.frequency,
        acceleration_std=args.acceleration_std,
        minimum_measurement_std=args.minimum_measurement_std,
    )
    filtered = lowpass(filtered, args.frequency, args.cutoff, args.filter_order)

    aligned_error = position_errors(fused_aligned[:, marker_indices], mocap)
    filtered_error = position_errors(filtered[:, marker_indices], mocap)
    output = args.output or data / "nlf_fused_aligned_filtered.csv"
    write_xyz(output, frames, filtered, NLF_MARKERS)

    np.savez(
        output.with_suffix(".transform.npz"),
        rotation=rotation,
        translation=translation,
        kabsch_markers=np.asarray(KABSCH_MARKERS),
    )
    print(f"Frames: {len(frames)}; NLF markers: {len(NLF_MARKERS)}")
    print(f"MoCap markers displayed: {len(mocap_marker_names)}")
    print(f"Kabsch markers: {len(KABSCH_MARKERS)}")
    print(f"Kabsch rotation:\n{rotation}")
    print(f"Kabsch translation [m]: {translation}")
    print(f"Fused/aligned MPJPE: {aligned_error[0]:.4f} m; RMSE: {aligned_error[1]:.4f} m")
    print(f"Filtered MPJPE: {filtered_error[0]:.4f} m; RMSE: {filtered_error[1]:.4f} m")
    print(f"Saved: {output}")
    print(f"Saved transform: {output.with_suffix('.transform.npz')}")
    if args.plot:
        selected_markers = (
            NLF_MARKERS
            if args.plot_markers.strip().lower() == "all"
            else [name.strip() for name in args.plot_markers.split(",") if name.strip()]
        )
        plot_output = args.plot_output or data / "nlf_marker_plots.pdf"
        plot_marker_comparisons(
            frames, points_02, points_46, filtered, mocap_all,
            mocap_marker_names, args.frequency, selected_markers, plot_output,
        )
        print(f"Saved plots ({len(selected_markers)} markers): {plot_output}")
    if args.visualize:
        visualize_markers(
            frames, points_02, points_46, filtered, mocap_all,
            step=args.visualization_step,
            interval_ms=args.visualization_interval,
        )


if __name__ == "__main__":
    main()
