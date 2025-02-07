#python example/python/pose_tracker_2.py cpu rtmpose-ort/rtmdet-nano/ rtmpose-trt/rtmpose-m/ 0
#python3 -m unittests.test_pose_tracker cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m /root/workspace/ros_ws/src/rt-cosmik/output/camera0_output.avi 
#python3 -m unittests.test_pose_tracker cuda /root/workspace/mmdeploy/rtmpose-trt/rtmdet-nano /root/workspace/mmdeploy/rtmpose-trt/rtmpose-m 0

import argparse
import os
import numpy as np
import cv2
from mmdeploy_runtime import PoseTracker
import math as m
import csv
def parse_args():
    parser = argparse.ArgumentParser(description='show how to use SDK Python API')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument('det_model', help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('pose_model', help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('video', help='video path or camera index')
    parser.add_argument('--output_dir', help='output directory', default=None)
    args = parser.parse_args()
    if args.video.isnumeric():
        args.video = int(args.video)
    return args


def visualize(frame, keypoints,bboxes, output_dir, frame_id, thr=0.5, resize=1280):
    # Updated skeleton for 26 points
    skeleton= [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (17, 18), (1, 2), (5, 18),(6, 18), # Head, shoulders, and neck connections

    (5, 7), (7, 9),                                                              # Right arm connections

    (6, 8), (8, 10),                                                             # Left arm connections

    (18, 19), (19, 11), (19, 12),                                                      # Shoulders to hips connections

    (11, 13), (13, 15), (15, 20), (15, 22), (15, 24),                            # Left leg and foot connections

    (12, 14), (14, 16), (16, 21), (16, 23), (16, 25),                            # Right leg and foot connections
                                                      # Hip connection
]

    # Updated palette
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]

    # Updated link color
    link_color = [
        1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
        2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]

    # Updated point color
    point_color = [
        0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4,
        5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5,
        5, 6, 6, 6, 6, 1, 1, 1, 1
    ]

    scale = resize / max(frame.shape[0], frame.shape[1])
    #keypoints, bboxes, _ = results
    #print(bboxes)
    scores = keypoints[..., 2]
    keypoints = (keypoints[..., :2] * scale).astype(int)
    bboxes *= scale
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    for kpts, score, bbox in zip(keypoints, scores, bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        show = [0] * len(kpts)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1, cv2.LINE_AA)
                show[u] = show[v] = 1
        for kpt, show, color in zip(kpts, show, point_color):
            if show:
                cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)
    if output_dir:
        cv2.imwrite(f'{output_dir}/{str(frame_id).zfill(6)}.jpg', img)
    else:
        cv2.imshow('pose_tracker', img)
        return cv2.waitKey(1) != 'q'
    return True


def main():
    args = parse_args()

    video = cv2.VideoCapture(args.video)

    tracker = PoseTracker(
        det_model=args.det_model,
        pose_model=args.pose_model,
        device_name=args.device_name)

    # Adjust coco_sigmas if needed for 26 keypoints (replace these with correct values for your model)
    coco_sigmas = [0.026] * 26  # Placeholder, adjust based on your model

    state = tracker.create_state(
        det_interval=1, det_min_bbox_size=100, keypoint_sigmas=coco_sigmas)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    csv_file = os.path.join("/root/workspace/ros_ws/src/rt-cosmik/output", 'keypoints.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header (Assuming you have 26 keypoints with x, y, z for each)
        header = ['frame_id'] + [f'keypoint_{i}_x' for i in range(26)] + [f'keypoint_{i}_y' for i in range(26)] 
        writer.writerow(header)

        frame_id = 0
        first_person_bbox = None
        is_someone_detected=False
        frame_of_first_detection=0
        while True:
            success, frame = video.read()
            if not success:
                break
            results = tracker(state, frame, detect=-1)
            keypoints, bboxes, _ = results
            
            if len(bboxes) > 0 and is_someone_detected==False:
                first_person_bbox = bboxes[0] 
            
            if first_person_bbox is not None:
                closest_person_idx = None
                min_distance = float('inf')

                for i, bbox in enumerate(bboxes):
                    distance = abs(first_person_bbox[2] - bbox[2])  
                    if distance < min_distance:
                        min_distance = distance
                        closest_person_idx = i

                        
                if closest_person_idx is not None:
                    first_person_bbox = (bboxes[closest_person_idx] + first_person_bbox)/2.0 #moyenne mobile
                    keypoints = keypoints[closest_person_idx:closest_person_idx + 1]
                    bboxes = bboxes[closest_person_idx:closest_person_idx + 1]
                    keypoints_to_write = (keypoints[..., :2] ).astype(float)
                    print(keypoints_to_write)


                    row = [frame_id] + keypoints_to_write.flatten().tolist()
                    writer.writerow(row)
                    


            if not visualize(frame, keypoints,bboxes, args.output_dir, frame_id):
                break
            frame_id += 1


if __name__ == '__main__':
    main()

