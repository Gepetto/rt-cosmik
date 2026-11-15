"""Checkerboard calibration: intrinsics per camera, pose per camera pair."""
import glob

import cv2 as cv
import numpy as np

from rtcosmik.config_loader import settings

_CORNER_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
_STEREO_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


def checkerboard_grid():
    """The board of the settings: ``((rows, columns), corners)``, corners (N, 3) in metres."""
    rows = settings.checkerboard_rows
    columns = settings.checkerboard_columns
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    return (rows, columns), objp * settings.checkerboard_square_size


def read_image_folder(pattern):
    """The images matching a glob, in sorted order: shot n must have the same name in every camera."""
    return [cv.imread(name, 1) for name in sorted(glob.glob(pattern))]


def find_checkerboard(image, size, refine=True):
    """The board's corners in one image, or None when it is not visible."""
    if image is None:
        return None
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, size, None)
    if not found:
        return None
    if refine:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), _CORNER_CRITERIA)
    return corners


def calibrate_intrinsics(images_pattern, show=True):
    """One camera's ``(reprojection error in px, matrix, distortion)`` from its board images."""
    images = read_image_folder(images_pattern)
    size, objp = checkerboard_grid()

    objpoints, imgpoints = [], []
    for frame in images:
        corners = find_checkerboard(frame, size)
        if corners is None:
            continue
        if show:
            cv.drawChessboardCorners(frame, size, corners, True)
            cv.imshow("img", frame)
            cv.waitKey(50)
        objpoints.append(objp)
        imgpoints.append(corners)
    if show:
        cv.destroyAllWindows()
    if not objpoints:
        raise ValueError(f"no checkerboard found in {images_pattern}")

    height, width = images[0].shape[:2]
    rmse, mtx, dist, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None)
    return rmse, mtx, dist


def corners_consistently_ordered(corners1, corners2):
    """Whether two detections run the same way along the board's diagonal."""
    diagonal1 = corners1[-1][0] - corners1[0][0]
    diagonal2 = corners2[-1][0] - corners2[0][0]
    cosine = np.dot(diagonal1, diagonal2) / (
        np.linalg.norm(diagonal1) * np.linalg.norm(diagonal2))
    return cosine > 0.9


def calibrate_stereo_pair(mtx1, dist1, mtx2, dist2, images_pattern_1, images_pattern_2,
                          show=True):
    """Pose of camera 2 relative to camera 1 from the shots both see, intrinsics held fixed.

    Returns ``(rmse, R, T)`` in OpenCV's convention, ``p_2 = R @ p_1 + T``.
    """
    c1_images = read_image_folder(images_pattern_1)
    c2_images = read_image_folder(images_pattern_2)
    size, objp = checkerboard_grid()

    objpoints, imgpoints1, imgpoints2 = [], [], []
    for frame1, frame2 in zip(c1_images, c2_images):
        corners1 = find_checkerboard(frame1, size)
        corners2 = find_checkerboard(frame2, size)
        if corners1 is None or corners2 is None:
            continue
        if not corners_consistently_ordered(corners1, corners2):
            continue
        if show:
            cv.drawChessboardCorners(frame1, size, corners1, True)
            cv.imshow("img", frame1)
            cv.drawChessboardCorners(frame2, size, corners2, True)
            cv.imshow("img2", frame2)
            cv.waitKey(0)  # the operator checks each pair
        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)
    if show:
        cv.destroyAllWindows()
    if not objpoints:
        raise ValueError("no shot shows the checkerboard to both cameras")

    height, width = c1_images[0].shape[:2]
    rmse, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2,
        (width, height), criteria=_STEREO_CRITERIA, flags=cv.CALIB_FIX_INTRINSIC)
    return rmse, R, T


def count_shared_checkerboard_views(images_pattern_1, images_pattern_2):
    """``(shared, total)``: how many shots show the board to both cameras."""
    images_1 = read_image_folder(images_pattern_1)
    images_2 = read_image_folder(images_pattern_2)
    size, _ = checkerboard_grid()
    shared = sum(1 for f1, f2 in zip(images_1, images_2)
                 if find_checkerboard(f1, size, refine=False) is not None
                 and find_checkerboard(f2, size, refine=False) is not None)
    return shared, min(len(images_1), len(images_2))
