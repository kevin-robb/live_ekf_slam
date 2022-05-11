cmake_minimum_required(VERSION 3.0.2)
project(landmark_detection_pkg)

find_package(catkin REQUIRED COMPONENTS
  apriltag_ros
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
