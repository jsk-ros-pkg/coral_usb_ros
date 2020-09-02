cmake_minimum_required(VERSION 2.8.3)
project(coral_usb)

find_package(
  catkin REQUIRED
  catkin_virtualenv
  dynamic_reconfigure
)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/EdgeTPUObjectDetector.cfg
  cfg/EdgeTPUFaceDetector.cfg
  cfg/EdgeTPUHumanPoseEstimator.cfg
)

catkin_package()

catkin_generate_virtualenv(
  INPUT_REQUIREMENTS requirements.in
  PYTHON_INTERPRETER python3
  USE_SYSTEM_PACKAGES TRUE
  ISOLATE_REQUIREMENTS FALSE
  CHECK_VENV FALSE
)

file(GLOB NODE_SCRIPTS_FILES node_scripts/*)
catkin_install_python(
  PROGRAMS ${NODE_SCRIPTS_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(
  PROGRAMS scripts/download_models.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY cfg docker launch training
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

install(FILES requirements.in requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
