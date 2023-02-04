cmake_minimum_required(VERSION 3.0.2)
project(yolo_ros)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a Python executable
catkin_install_python(
  PROGRAMS src/yolo_ros_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)