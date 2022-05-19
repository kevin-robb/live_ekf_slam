cmake_minimum_required(VERSION 3.0.2)
project(ukf_pkg)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  std_msgs
)


add_message_files(
  FILES
  UKFState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
include
  # ${ekf_pkg}?
  ${catkin_INCLUDE_DIRS}/include/**
)

# Build C++ files
set(SOURCES src/ukf.cpp src/ukf_node.cpp)
add_executable(ukf_node ${SOURCES})
target_link_libraries(ukf_node ${catkin_LIBRARIES})