echo "Running calibration..."

echo "Running cam1 intrinsics..."
python3 ../../cams_calibration/rgb/calibrate_cameras.py 1

echo "Running cam2 intrinsics..."
python3 ../../cams_calibration/rgb/calibrate_cameras.py 2

echo "Setting world frame..."
python3 ../../cams_calibration/rgb/set_world_frame.py

echo "Running cam1 extrinsics..."
python3 calib_mocaptocam_indiv.py 1

echo "Running cam2 extrinsics..."
python3 calib_mocaptocam_indiv.py 2

echo "Calib cam2cam..."
python3 cam2cam_calib.py

echo "Calibration done!"