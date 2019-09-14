cmake_minimum_required(VERSION 2.8.3)
project(coral_usb)

find_package(
  catkin REQUIRED
  catkin_virtualenv
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/EdgeTPUObjectDetector.cfg
)

catkin_package()

catkin_generate_virtualenv(
  PYTHON_VERSION 3.5
  USE_SYSTEM_PACKAGES TRUE
  ISOLATE_REQUIREMENTS FALSE
)

file(GLOB NODE_SCRIPTS_FILES node_scripts/*)
catkin_install_python(
  PROGRAMS ${NODE_SCRIPTS_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

install(FILES requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
