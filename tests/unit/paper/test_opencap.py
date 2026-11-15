"""OpenCap LabValidation: files, time base, marker mapping and coordinate comparison."""
import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rtcosmik.paper import opencap as oc


def _trc(path, names, times, xyz):
    """A TRC file as OpenCap writes them; NaN cells are left empty."""
    lines = ["PathFileType\t4\t(X/Y/Z)\tfile.trc",
             "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
             f"100.0\t100.0\t{len(times)}\t{len(names)}\tmm\t100.0\t1\t{len(times)}",
             "Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t",
             "\t\t" + "\t".join(f"{a}{i + 1}" for i in range(len(names)) for a in "XYZ"), ""]
    for f, (t, row) in enumerate(zip(times, xyz)):
        cells = ["" if np.isnan(v) else f"{v:.5f}" for v in np.asarray(row).ravel()]
        lines.append("\t".join([str(f + 1), f"{t:.5f}"] + cells))
    Path(path).write_text("\n".join(lines) + "\n")


def _mot(path, times, columns):
    names = list(columns)
    lines = ["Coordinates", "version=1", f"nRows={len(times)}", f"nColumns={len(names) + 1}",
             "inDegrees=yes", "", "endheader", "time\t" + "\t".join(names)]
    for i, t in enumerate(times):
        lines.append("\t".join([f"{t:.8f}"] + [f"{columns[n][i]:.8f}" for n in names]))
    Path(path).write_text("\n".join(lines) + "\n")


def test_trc_gives_named_markers_in_file_units_and_nan_for_empty_cells(tmp_path):
    xyz = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.5, 2.5, 3.5], [np.nan, np.nan, np.nan]]])
    _trc(tmp_path / "t.trc", ["r.ASIS", "C7"], [0.0, 0.01], xyz)

    times, markers = oc.read_trc(tmp_path / "t.trc")

    np.testing.assert_allclose(times, [0.0, 0.01])
    np.testing.assert_allclose(markers["r.ASIS"], [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    np.testing.assert_allclose(markers["C7"][0], [4.0, 5.0, 6.0])
    assert np.isnan(markers["C7"][1]).all()


def test_resampling_interpolates_linearly_and_never_extrapolates_or_bridges_gaps():
    t = np.arange(0, 0.05, 0.01)                       # 100 Hz
    values = np.stack([100 * t, 3 - 50 * t], 1)        # two linear channels
    values[3] = np.nan

    out = oc.resample(t, values, [0.005, 0.015, 0.025, 0.04, 0.045])

    np.testing.assert_allclose(out[0], [0.5, 2.75])
    np.testing.assert_allclose(out[1], [1.5, 2.25])
    assert np.isnan(out[2]).all()                      # next to the missing sample
    np.testing.assert_allclose(out[3], [4.0, 1.0])
    assert np.isnan(out[4]).all()                      # past the last sample


def _trajectory(frames, markers, seed=0, fs=60.0):
    """Smooth, non-periodic marker trajectories: a function of time in seconds."""
    rng = np.random.default_rng(seed)
    freq, phase = rng.uniform(0.3, 2.0, (4, markers, 3)), rng.uniform(0, 6.28, (4, markers, 3))
    amp = rng.uniform(0.05, 0.3, (4, markers, 3))
    at = lambda t: (amp * np.sin(2 * np.pi * freq * np.asarray(t, float)[:, None, None, None] + phase)).sum(1)
    return at, at(np.arange(frames) / fs)


def test_camera_offsets_are_recovered_despite_a_constant_bias_per_camera_and_gaps():
    _, track = _trajectory(260, 6)
    true = [4, 0, -7, 11]                                # camera c's row r shows instant r + true[c]
    rng = np.random.default_rng(1)
    views = np.stack([track[20 + o:220 + o] + rng.normal(0, 0.1, 3) + rng.normal(0, 0.005, (200, 6, 3))
                      for o in true], 1)
    views[:15, 2] = np.nan

    offsets = oc.camera_offsets(views, max_lag=20)

    assert list(offsets - offsets[1]) == true


def test_camera_without_landmarks_does_not_disturb_the_offsets_of_the_others():
    _, track = _trajectory(260, 6)
    true = [4, 0, -7]
    views = np.stack([track[20 + o:220 + o] for o in true] + [np.full((200, 6, 3), np.nan)], 1)

    offsets = oc.camera_offsets(views, max_lag=20)

    assert list(offsets[:3] - offsets[1]) == true
    assert offsets.min() <= offsets[3] <= offsets.max()


def test_aligned_cameras_show_the_same_instant_on_each_row():
    instants = np.arange(100.0)
    offsets = np.array([4, 0, -7])
    views = np.stack([instants + o for o in offsets], 1)[:, :, None]    # each cell holds its instant

    aligned, first = oc.align_cameras(views, offsets)

    assert aligned.shape == (100 - 11, 3, 1)
    assert (aligned[:, 0] == aligned[:, 1]).all() and (aligned[:, 1] == aligned[:, 2]).all()
    assert aligned[0, 1, 0] == first == 4               # in the rows of the camera with offset 0


def test_view_holding_someone_else_or_nobody_is_counted_as_far():
    views = np.zeros((10, 3, 4, 3))
    views[:4, 1] += [2.0, 0.0, 0.0]            # a bystander in camera 1 for 4 rows
    views[8:, 2] = np.nan                      # nobody in camera 2 for 2 rows
    np.testing.assert_allclose(oc.far_views(views, distance=1.0).mean(0), [0.0, 0.4, 0.2])


def test_clock_offset_places_the_video_rows_on_the_mocap_clock():
    from scipy.spatial.transform import Rotation
    at, _ = _trajectory(1, 8, seed=3)
    mocap_times = np.arange(0, 1.5, 0.01)
    R, t = Rotation.from_euler("zyx", [2.0, 0.1, -0.2]).as_matrix(), np.array([1.0, -3.0, 0.5])
    truth = -0.4625                                      # row 0 was filmed before the mocap started
    rows = at(np.arange(150) / 60 + truth) @ R.T + t

    offset, residual = oc.clock_offset(rows, 60, mocap_times, at(mocap_times), search=(-1.0, 0.5))

    assert offset == pytest.approx(truth, abs=1 / 240)
    assert residual < 0.005


def test_reference_markers_are_renamed_scaled_to_metres_and_put_on_the_frame_times(tmp_path):
    names = ["r.ASIS", "R_wrist_ulna", "r_thigh1"]
    times = np.arange(0, 0.04, 0.01)
    xyz = np.zeros((4, 3, 3))
    xyz[:, 0, 0] = 1000 * times                        # r.ASIS moves 1 m/s along x
    xyz[:, 1] = [10.0, 20.0, 30.0]
    _trc(tmp_path / "t.trc", names, times, xyz)

    markers = oc.reference_markers(tmp_path / "t.trc", at=[0.015, 0.02])

    assert set(markers) == {"RASI", "RMWRI"}           # clusters are not ours
    np.testing.assert_allclose(markers["RASI"][:, 0], [0.015, 0.02])
    np.testing.assert_allclose(markers["RMWRI"], [[0.01, -0.03, 0.02]] * 2)   # the lab's y is up, ours z


def test_opencap_mocap_set_is_what_dynamic_trials_carry():
    from rtcosmik.config_loader import settings
    from rtcosmik.marker_sets import apply_marker_set
    s = apply_marker_set(copy.deepcopy(settings), "opencap_mocap")

    assert len(s.marker_names) == 25
    assert set(s.marker_names) == set(oc.MARKERS) - set(oc.STATIC_ONLY)
    assert not {"Head", "REar", "LEar", "RMKNE", "LMANK"} & set(s.marker_names)
    assert {"middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y"} <= set(s.locked_joints)
    assert s.keys_to_track_list == s.marker_names


def test_sides_and_medial_lateral_are_not_crossed():
    assert oc.MARKERS["RWRI"] == "R_wrist_radius" and oc.MARKERS["RMWRI"] == "R_wrist_ulna"
    assert oc.MARKERS["LELB"] == "L_elbow_lat" and oc.MARKERS["LMELB"] == "L_elbow_med"
    assert oc.MARKERS["RHEE"] == "r_calc" and oc.MARKERS["LANK"] == "L_ankle"
    for ours, theirs in oc.MARKERS.items():
        assert ours[0] == theirs[0].upper() or ours == "C7"


def _camera_file(path, R, t):
    data = {"intrinsicMat": np.array([[900.0, 0, 360], [0, 900, 640], [0, 0, 1]]),
            "distortion": np.zeros((1, 5)), "rotation": R, "translation": t.reshape(3, 1),
            "imageSize": np.array([[1280.0], [720.0]])}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


def test_camera_file_loads_numpy_only(tmp_path):
    _camera_file(tmp_path / "ok.pickle", np.eye(3), np.array([1.0, 2.0, 3.0]))
    camera = oc.load_camera(tmp_path / "ok.pickle")
    np.testing.assert_allclose(camera["translation"].ravel(), [1.0, 2.0, 3.0])

    with open(tmp_path / "bad.pickle", "wb") as handle:
        pickle.dump({"f": os.getcwd}, handle)
    with pytest.raises(pickle.UnpicklingError):
        oc.load_camera(tmp_path / "bad.pickle")


def test_written_calibration_takes_a_board_point_into_every_camera_and_back(tmp_path):
    from scipy.spatial.transform import Rotation
    from rtcosmik.camera.cam_utils import load_camera_parameters, load_world_transformation
    poses = {0: (Rotation.from_euler("xyz", [0.1, -0.7, 0.05]).as_matrix(), np.array([100.0, -50.0, 3000.0])),
             3: (Rotation.from_euler("xyz", [-0.2, 0.6, 0.3]).as_matrix(), np.array([-400.0, 20.0, 3500.0]))}
    for camera, (R, t) in poses.items():
        _camera_file(tmp_path / "VideoData" / "Session1" / f"Cam{camera}" / oc.CAMERA_FILE, R, t)

    cam_dir = oc.write_calibration(tmp_path, 1, tmp_path / "calibration", cameras=(0, 3))
    mtxs, _, projections, _, _ = load_camera_parameters(cam_dir, [0, 3])
    world_R, world_T = load_world_transformation(cam_dir, 0)

    board_mm = np.array([250.0, -900.0, -400.0])
    in_camera_m = {c: (R @ board_mm + t) / 1000 for c, (R, t) in poses.items()}
    # the board's y points down and its z into the wall; the world is z up
    np.testing.assert_allclose(world_R @ in_camera_m[0] + world_T, [0.25, -0.4, 0.9], atol=1e-9)
    np.testing.assert_allclose(projections[1] @ np.append(in_camera_m[0], 1), in_camera_m[3], atol=1e-9)
    assert mtxs[0][0, 2] == 360 and mtxs[0][1, 2] == 640


def test_rigid_fit_recovers_a_rotation_and_translation_and_ignores_missing_points():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3))
    R, t = Rotation.from_euler("zyx", [1.0, -0.4, 2.0]).as_matrix(), np.array([0.3, -2.0, 1.0])
    target = source @ R.T + t
    source[7] = np.nan

    R_fit, t_fit = oc.rigid_fit(source, target)

    np.testing.assert_allclose(R_fit, R, atol=1e-9)
    np.testing.assert_allclose(t_fit, t, atol=1e-9)


def test_folded_calibration_posture_is_refused():
    names = ["Left_Hip_Flexion_Extension[rad]", "Left_Knee_Flexion_Extension[rad]", "Lumbar_Flexion_Extension[rad]"]
    oc.require_standing(names, np.radians([10.0, -8.0, 50.0]))
    with pytest.raises(RuntimeError, match="Left_Hip_Flexion"):
        oc.require_standing(names, np.radians([82.0, 48.0, 0.0]))


def test_activity_of_a_trial_name():
    assert [oc.activity(n) for n in ("squatsAsym1", "STSweakLegs1", "DJAsym4", "walkingTS2", "DJ1")] \
        == ["squat", "sit_to_stand", "drop_jump", "walking", "drop_jump"]
    with pytest.raises(ValueError):
        oc.activity("static1")


def test_analysis_window_is_the_frames_inside_the_released_time_range():
    times = np.array([0.98, 1.0, 1.0167, 1.0333, 1.05])
    assert oc.window_frames(times, (1.0, 1.04)) == (1, 3)


def test_released_error_compares_video_and_mocap_on_their_common_times(tmp_path):
    t = np.round(np.arange(0.01, 0.11, 0.01), 2)
    knee = np.linspace(0, 90, len(t))
    _mot(tmp_path / "mocap.mot", t, {"knee_angle_r": knee, "hip_flexion_r": knee / 2})
    _mot(tmp_path / "video.mot", t[:-2], {"knee_angle_r": knee[:-2] + 3.0,
                                         "hip_flexion_r": knee[:-2] / 2 + np.tile([1.0, -1.0], 4)})

    errors = oc.released_errors(tmp_path / "video.mot", tmp_path / "mocap.mot")

    assert errors["knee_angle_r"] == pytest.approx({"rmse": 3.0, "mae": 3.0, "frames": 8})
    assert errors["hip_flexion_r"]["rmse"] == pytest.approx(1.0)


def test_offset_free_agreement_reports_correlation_and_rmse_without_the_mean_difference():
    x = np.sin(np.linspace(0, 6, 200))
    r, rmse = oc.agreement(-(40 * x) + 12.0, 40 * x)
    assert r == pytest.approx(-1.0)
    assert rmse == pytest.approx(0.0, abs=1e-9)        # sign and offset both removed


def test_external_participant_takes_fold_zero_and_the_guard_names_it():
    from rtcosmik import landmark_maps
    from rtcosmik.config_loader import settings

    chosen = landmark_maps.select(settings, "subject4")
    landmark_maps.require_out_of_fold(chosen, "subject4", settings)

    assert chosen.fold == 0
    identity = chosen.identity("subject4")
    assert identity["in_sample"] is False and identity["external_participant"] is True
    with pytest.raises(landmark_maps.LeakageError):
        landmark_maps.require_out_of_fold(chosen, "subject6", settings)   # no videos: not declared


def test_dataset_settings_run_at_the_video_rate_in_portrait():
    from rtcosmik.config_loader import settings
    s = oc.dataset_settings(settings)
    assert (s.fs, s.width, s.height) == (60, 720, 1280)
    assert s.dt == pytest.approx(1 / 60)
    assert s.paper["clock_offset"] == 0
    assert settings.fs == 40                            # the shared settings are untouched
