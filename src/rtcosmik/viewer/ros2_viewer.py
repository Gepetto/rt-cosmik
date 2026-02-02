from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


class Ros2Context:
    def __init__(self, node_name: str, freeflyer: bool) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node: Node = rclpy.create_node(node_name)
        self.q_pub = self.node.create_publisher(JointState, "/human_RT_joint_angles", 10)
        self.keypoints_pub = self.node.create_publisher(MarkerArray, "/keypoints", 10)
        self.markers_pub = self.node.create_publisher(MarkerArray, "/lstm_markers", 10)
        self.br = TransformBroadcaster(self.node) if freeflyer else None


def ros2_init(freeflyer: bool) -> Ros2Context:
    return Ros2Context(node_name="human_rt_ik", freeflyer=freeflyer)


def _timestamp(node: Node):
    return node.get_clock().now().to_msg()


def publish_keypoints_as_marker_array(
    keypoints: Iterable[Iterable[float]],
    marker_pub,
    node: Node,
    keypoint_names: Iterable[str],
    frame_id: str = "world",
) -> None:
    marker_array = MarkerArray()
    marker_template = Marker()
    marker_template.header.frame_id = frame_id
    marker_template.header.stamp = _timestamp(node)
    marker_template.ns = "keypoints"
    marker_template.type = Marker.SPHERE
    marker_template.action = Marker.ADD
    marker_template.scale.x = 0.05
    marker_template.scale.y = 0.05
    marker_template.scale.z = 0.05
    marker_template.color.a = 1.0

    palette = [
        [51, 153, 255],
        [0, 255, 0],
        [255, 128, 0],
        [255, 255, 255],
        [255, 153, 255],
        [102, 178, 255],
        [255, 51, 51],
    ]

    keypoints_color = [
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]

    keypoint_names_list = list(keypoint_names)
    for i, keypoint in enumerate(keypoints):
        marker = Marker()
        marker.header = marker_template.header
        marker.ns = marker_template.ns
        marker.type = marker_template.type
        marker.action = marker_template.action
        marker.scale = marker_template.scale
        marker.color.a = marker_template.color.a
        marker.id = i
        marker.text = keypoint_names_list[i] if i < len(keypoint_names_list) else f"keypoint_{i}"

        color_info = palette[keypoints_color[i]]
        marker.color.r = color_info[0] / 255
        marker.color.g = color_info[1] / 255
        marker.color.b = color_info[2] / 255

        marker.pose.position = Point(x=keypoint[0], y=keypoint[1], z=keypoint[2])
        marker_array.markers.append(marker)

    marker_pub.publish(marker_array)


def publish_augmented_markers(
    keypoints: Iterable[Iterable[float]],
    marker_pub,
    node: Node,
    keypoint_names: Iterable[str],
    frame_id: str = "world",
) -> None:
    marker_array = MarkerArray()
    marker_template = Marker()
    marker_template.header.frame_id = frame_id
    marker_template.header.stamp = _timestamp(node)
    marker_template.ns = "markers"
    marker_template.type = Marker.SPHERE
    marker_template.action = Marker.ADD
    marker_template.scale.x = 0.05
    marker_template.scale.y = 0.05
    marker_template.scale.z = 0.05
    marker_template.color.a = 1.0
    marker_template.color.r = 0.0
    marker_template.color.g = 0.0
    marker_template.color.b = 1.0

    keypoint_names_list = list(keypoint_names)
    for i, keypoint in enumerate(keypoints):
        marker = Marker()
        marker.header = marker_template.header
        marker.ns = marker_template.ns
        marker.type = marker_template.type
        marker.action = marker_template.action
        marker.scale = marker_template.scale
        marker.color.a = marker_template.color.a
        marker.color.r = marker_template.color.r
        marker.color.g = marker_template.color.g
        marker.color.b = marker_template.color.b
        marker.id = i
        marker.text = keypoint_names_list[i] if i < len(keypoint_names_list) else f"marker_{i}"

        marker.pose.position = Point(x=keypoint[0], y=keypoint[1], z=keypoint[2])
        marker_array.markers.append(marker)

    marker_pub.publish(marker_array)


def _publish_pelvis_transform(
    q: np.ndarray,
    br: TransformBroadcaster,
    node: Node,
    world_frame: str,
    pelvis_frame: str,
) -> None:
    q_trans = np.array([q[0], q[1], q[2]])
    q_quat = pin.Quaternion(q[3:7])
    t_current = pin.SE3(q_quat, q_trans)

    correction_rot = pin.utils.rotate("x", np.pi / 2)
    t_correction = pin.SE3(correction_rot, np.zeros(3))
    t_corrected = t_correction * t_current

    corrected_rotation = pin.Quaternion(t_corrected.rotation)
    corrected_translation = t_corrected.translation

    transform = TransformStamped()
    transform.header.stamp = _timestamp(node)
    transform.header.frame_id = world_frame
    transform.child_frame_id = pelvis_frame
    transform.transform.translation.x = corrected_translation[0]
    transform.transform.translation.y = corrected_translation[1]
    transform.transform.translation.z = corrected_translation[2]
    transform.transform.rotation.x = corrected_rotation[0]
    transform.transform.rotation.y = corrected_rotation[1]
    transform.transform.rotation.z = corrected_rotation[2]
    transform.transform.rotation.w = corrected_rotation[3]
    br.sendTransform(transform)


def _publish_segment_transforms(
    model,
    data,
    q: np.ndarray,
    br: TransformBroadcaster,
    node: Node,
    world_frame: str,
    skip_frames: Optional[Iterable[str]] = None,
) -> None:
    pin.framesForwardKinematics(model, data, q)
    skip = set(skip_frames or [])
    for frame in model.frames.tolist():
        if frame.name in skip:
            continue
        placement = data.oMf[model.getFrameId(frame.name)]
        transform = TransformStamped()
        transform.header.stamp = _timestamp(node)
        transform.header.frame_id = world_frame
        transform.child_frame_id = frame.name
        transform.transform.translation.x = float(placement.translation[0])
        transform.transform.translation.y = float(placement.translation[1])
        transform.transform.translation.z = float(placement.translation[2])
        quat = pin.Quaternion(placement.rotation)
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        br.sendTransform(transform)


def publish_kinematics(
    q: np.ndarray,
    pub,
    dof_names,
    node: Node,
    br: Optional[TransformBroadcaster] = None,
    model=None,
    data=None,
    publish_segments: bool = False,
    world_frame: str = "world",
    pelvis_frame: str = "pelvis",
) -> None:
    if br is not None:
        _publish_pelvis_transform(q, br, node, world_frame, pelvis_frame)
        q_to_send = q[7:]
    else:
        q_to_send = q

    joint_state_msg = JointState()
    joint_state_msg.header.stamp = _timestamp(node)
    joint_state_msg.name = list(dof_names)
    joint_state_msg.position = q_to_send.tolist()
    pub.publish(joint_state_msg)

    if publish_segments and br is not None and model is not None and data is not None:
        _publish_segment_transforms(
            model,
            data,
            q,
            br,
            node,
            world_frame,
            skip_frames={"universe", pelvis_frame},
        )
