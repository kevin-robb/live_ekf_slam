cmake_minimum_required(VERSION 3.0.2)
project(ekf_pkg)

find_package(catkin REQUIRED COMPONENTS
  base_pkg
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  roslib
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS base_pkg
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# Build C++ files
set(SOURCES src/ekf.cpp src/ekf_node.cpp)
add_executable(ekf_node ${SOURCES})
target_link_libraries(ekf_node ${catkin_LIBRARIES})

add_dependencies(ekf_node base_pkg_generate_messages_cpp)