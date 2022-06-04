cmake_minimum_required(VERSION 3.0.2)
project(ukf_pkg)

find_package(catkin REQUIRED COMPONENTS
  data_pkg
  geometry_msgs
  roscpp
  rospy
  std_msgs
  roslib
)

include_directories("/usr/include/eigen3")

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS data_pkg
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

# Build C++ files
set(SOURCES src/ukf.cpp src/ukf_node.cpp)
add_executable(ukf_node ${SOURCES})
target_link_libraries(ukf_node ${catkin_LIBRARIES})

add_dependencies(ukf_node data_pkg_generate_messages_cpp)
