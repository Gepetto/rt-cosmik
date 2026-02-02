from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="cosmik_ros2",
                executable="run_pipeline",
                name="cosmik_pipeline",
                output="screen",
            ),
        ]
    )
