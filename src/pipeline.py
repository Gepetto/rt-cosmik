# What we call piepeline is the process that takes as input the camera streams and process up to the inverse kinematics output

from src.pose_estimator.pose_estimator import PoseEstimator
from src.triangulation.triangulation import triangulate_points
from src.augmenter.lstm_v2 import augmentTRC, loadModel
from src.filtering.iir import IIRFilter
from src.ik.ik import RT_IK


def run_pipeline_process(input_queues, output_queues):