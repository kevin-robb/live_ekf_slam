cmake_minimum_required(VERSION 3.0.2)
project(pf_pkg)

find_package(catkin REQUIRED COMPONENTS
  base_pkg
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS base_pkg
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
