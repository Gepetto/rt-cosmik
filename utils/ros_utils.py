import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def publish_keypoints_as_marker_array(keypoints, marker_pub, keypoint_names, frame_id="world"):
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
