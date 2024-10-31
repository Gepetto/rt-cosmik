cmake_minimum_required(VERSION 3.10)
project(rt-cosmik)
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
    rospy
    xacro
)
catkin_package()

# Install urdf
install(DIRECTORY
  urdf/
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)


# Install meshes
install(DIRECTORY
  meshes/
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
)