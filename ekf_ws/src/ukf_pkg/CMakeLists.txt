cmake_minimum_required(VERSION 3.0.2)
project(ukf_pkg)

# find eigen.
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  std_msgs
  roslib
  # cmake_modules
  # Eigen3 REQUIRED
  # PkgConfig
)

# pkg_search_module(Eigen3 REQUIRED eigen3)
include_directories("/usr/include/eigen3")

# add_library(eigen ../../../../../usr/include/eigen3/unsupported/Eigen/MatrixFunctions) #new

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
  # DEPENDS Eigen3
)

# eigen unsupported.
# FIND_LIBRARY(Eigen_LIBRARIES eigen3 /usr/include/eigen3/unsupported/Eigen)

include_directories(
include
  # ${ekf_pkg}?
  ${catkin_INCLUDE_DIRS}/include/**
  ${Eigen_INCLUDE_DIRS}
)

# Build C++ files
set(SOURCES src/ukf.cpp src/ukf_node.cpp)
add_executable(ukf_node ${SOURCES})
target_link_libraries(ukf_node ${catkin_LIBRARIES})
# ${eigen_LIBRARIES}