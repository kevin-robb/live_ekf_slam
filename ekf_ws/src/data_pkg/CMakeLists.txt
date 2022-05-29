cmake_minimum_required(VERSION 3.0.2)
project(data_pkg)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  Command.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


# trying to get import params to work.

# file(GLOB python_scripts_DIR
#     "scripts/*.py"
# )

# install(DIRECTORY launch
#     DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# catkin_install_python(PROGRAMS ${python_scripts_DIR}
#                       DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})