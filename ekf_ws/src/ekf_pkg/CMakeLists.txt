cmake_minimum_required(VERSION 3.0.2)
project(ekf_pkg)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

add_message_files(
  FILES
  EKFState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
include
  # ${ekf_pkg}?
  ${catkin_INCLUDE_DIRS}/include/**
)

# Build C++ files
set(SOURCES src/ekf.cpp src/ekf_node.cpp)
add_executable(ekf_node ${SOURCES})
target_link_libraries(ekf_node ${catkin_LIBRARIES})