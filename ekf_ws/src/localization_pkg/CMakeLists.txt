cmake_minimum_required(VERSION 3.0.2)
project(localization_pkg)

find_package(catkin REQUIRED COMPONENTS
  base_pkg
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  roslib
)

include_directories("/usr/include/eigen3")
include_directories("/usr/include/yaml-cpp")

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS base_pkg
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

# Trying to get yaml working.
find_package(yaml-cpp REQUIRED)
include_directories(HEADERS_EXECUTABLE
    ${YAML_INCLUDE_DIRS}
)
# target_link_libraries(${CMAKE_PROJECT_NAME}
#     yaml-cpp
# )

# Build C++ files
set(SOURCES src/ekf.cpp src/ukf.cpp src/pose_graph.cpp src/localization_node.cpp)
add_executable(localization_node ${SOURCES})
target_link_libraries(localization_node ${catkin_LIBRARIES} yaml-cpp)

add_dependencies(localization_node base_pkg_generate_messages_cpp)
