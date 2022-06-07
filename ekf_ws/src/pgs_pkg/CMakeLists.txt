cmake_minimum_required(VERSION 3.0.2)
project(pgs_pkg)

find_package(catkin REQUIRED COMPONENTS
  data_pkg
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
