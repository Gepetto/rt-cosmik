import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import JointState


def publish_keypoints_as_marker_array(keypoints, marker_pub, keypoint_names, frame_id="world"):
    """
    Publishes a list of keypoints as a MarkerArray in ROS.
    Args:
        keypoints (list of list of float): A list of keypoints, where each keypoint is a list of three floats [x, y, z].
        marker_pub (rospy.Publisher): ROS publisher to publish the MarkerArray.
        keypoint_names (list of str): A list of names for the keypoints.
        frame_id (str, optional): The frame ID to use for the markers. Defaults to "world".
    Returns:
        None
    """

    marker_array = MarkerArray()
    marker_template = Marker()
    marker_template.header.frame_id = frame_id
    marker_template.header.stamp = rospy.Time.now()
    marker_template.ns = "keypoints"
    marker_template.type = Marker.SPHERE
    marker_template.action = Marker.ADD
    marker_template.scale.x = 0.05  # Adjust size as needed
    marker_template.scale.y = 0.05
    marker_template.scale.z = 0.05
    marker_template.color.a = 1.0  # Fully opaque
    
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    
    keypoints_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 
            1, 2, 1, 2, 1, 2, 1, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2
        ]

    for i, keypoint in enumerate(keypoints):
        marker = Marker()
        # Copy attributes from the template to the new marker
        marker.header = marker_template.header
        marker.ns = marker_template.ns
        marker.type = marker_template.type
        marker.action = marker_template.action
        marker.scale = marker_template.scale
        marker.color.a = marker_template.color.a
        marker.id = i
        marker.text = keypoint_names[i] if i < len(keypoint_names) else f"keypoint_{i}"

        # Set color based on index 
        color_info = palette[keypoints_color[i]]

        if i < len(keypoints_color):
            marker.color.r = color_info[0]/255
            marker.color.g = color_info[1]/255
            marker.color.b = color_info[2]/255
        else:
            marker.color.r = 1.0  # Fallback color
            marker.color.g = 0.0
            marker.color.b = 0.0

        # Set position of the keypoint
        marker.pose.position = Point(x=keypoint[0], y=keypoint[1], z=keypoint[2])
        marker_array.markers.append(marker)

    marker_pub.publish(marker_array)

def publish_augmented_markers(keypoints, marker_pub, keypoint_names, frame_id="world"):
    """
    Publishes augmented markers for a given set of keypoints.
    Args:
        keypoints (list of tuples): A list of augmented markers where each marker is a tuple (x, y, z).
        marker_pub (rospy.Publisher): ROS publisher to publish the MarkerArray.
        keypoint_names (list of str): A list of names for each keypoint.
        frame_id (str, optional): The frame ID to associate with the markers. Defaults to "world".
    Returns:
        None
    """

    marker_array = MarkerArray()
    marker_template = Marker()
    marker_template.header.frame_id = frame_id
    marker_template.header.stamp = rospy.Time.now()
    marker_template.ns = "markers"
    marker_template.type = Marker.SPHERE
    marker_template.action = Marker.ADD
    marker_template.scale.x = 0.05  # Adjust size as needed
    marker_template.scale.y = 0.05
    marker_template.scale.z = 0.05
    marker_template.color.a = 1.0  # Fully opaque
    marker_template.color.r = 0.0  # Fallback color
    marker_template.color.g = 0.0
    marker_template.color.b = 1.0

    for i, keypoint in enumerate(keypoints):
        marker = Marker()
        # Copy attributes from the template to the new marker
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
        marker.text = keypoint_names[i] if i < len(keypoint_names) else f"marker_{i}"

        # Set position of the keypoint
        marker.pose.position = Point(x=keypoint[0], y=keypoint[1], z=keypoint[2])
        marker_array.markers.append(marker)

    marker_pub.publish(marker_array)

def publish_kinematics(q, br, pub, dof_names):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = "pelvis"
    t.transform.translation.x = q[0]
    t.transform.translation.y = q[1]
    t.transform.translation.z = q[2]
    t.transform.rotation.x = q[3]
    t.transform.rotation.y = q[4]
    t.transform.rotation.z = q[5]
    t.transform.rotation.w = q[6]
    br.sendTransform(t)

    q_to_send=q[7:]

    # Publish joint angles 
    joint_state_msg=JointState()
    joint_state_msg.header.stamp=rospy.Time.now()
    joint_state_msg.name = dof_names
    joint_state_msg.position = q_to_send.tolist()
    pub.publish(joint_state_msg)
    print("kinematics published")