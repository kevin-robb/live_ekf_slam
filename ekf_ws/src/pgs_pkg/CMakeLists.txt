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

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(G2O REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS data_pkg
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
)

# Build C++ files
set(SOURCES src/pgs_node.cpp)
add_executable(pgs_node ${SOURCES})
target_link_libraries(pgs_node ${catkin_LIBRARIES} ${G2O_CORE_LIBRARY})

add_dependencies(pgs_node data_pkg_generate_messages_cpp)