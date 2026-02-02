from setuptools import setup

package_name = "cosmik_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/run_pipeline.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RT-COSMIK Maintainers",
    maintainer_email="msabbah@laas.fr",
    description="ROS 2 bridge and launch files for RT-COSMIK outputs.",
    license="BSD-2-Clause",
    entry_points={
        "console_scripts": [
            "run_pipeline = cosmik_ros2.run_pipeline_node:main",
        ],
    },
)
