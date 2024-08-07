cmake_minimum_required(VERSION 2.8.3)
project(coral_usb)

find_package(
  catkin REQUIRED
  catkin_virtualenv
  dynamic_reconfigure
  message_generation
)

catkin_python_setup()

add_service_files(
  FILES
  StartNode.srv
  StopNode.srv
  ListNodes.srv
)

generate_messages()

generate_dynamic_reconfigure_options(
  cfg/EdgeTPUObjectDetector.cfg
  cfg/EdgeTPUPanoramaObjectDetector.cfg
  cfg/EdgeTPUTileObjectDetector.cfg
  cfg/EdgeTPUFaceDetector.cfg
  cfg/EdgeTPUPanoramaFaceDetector.cfg
  cfg/EdgeTPUTileFaceDetector.cfg
  cfg/EdgeTPUHumanPoseEstimator.cfg
  cfg/EdgeTPUPanoramaHumanPoseEstimator.cfg
  cfg/EdgeTPUPanoramaSemanticSegmenter.cfg
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

find_package(Git REQUIRED)
if (NOT EXISTS ${PROJECT_SOURCE_DIR}/python/coral_usb/posenet/pose_engine.py)
  add_custom_target(posenet_update ALL
    COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive python/coral_usb/posenet
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
endif()

if (${catkin_virtualenv_VERSION} VERSION_LESS "0.6.1")
  message(WARNING "Please install catkin_virtualenv>=0.6.1.")
  message(WARNING "Current catkin_virtualen version is ${catkin_virtualenv_VERSION}")
else()
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
endif()

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
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_install_python(
    PROGRAMS
      test/dummy_edgetpu_object_detector.py
      test/dummy_edgetpu_panorama_object_detector.py
      test/dummy_edgetpu_tile_object_detector.py
      test/dummy_edgetpu_face_detector.py
      test/dummy_edgetpu_panorama_face_detector.py
      test/dummy_edgetpu_tile_face_detector.py
      test/dummy_edgetpu_human_pose_estimator.py
      test/dummy_edgetpu_panorama_human_pose_estimator.py
      test/dummy_edgetpu_semantic_segmenter.py
      test/dummy_edgetpu_panorama_semantic_segmenter.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

  find_package(rostest REQUIRED)
  # object detector
  add_rostest(test/dummy_edgetpu_object_detector.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
  add_rostest(test/dummy_edgetpu_panorama_object_detector.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
  add_rostest(test/dummy_edgetpu_tile_object_detector.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
  # face detector
  add_rostest(test/dummy_edgetpu_face_detector.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
  add_rostest(test/dummy_edgetpu_panorama_face_detector.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
  add_rostest(test/dummy_edgetpu_tile_face_detector.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
  # human pose estimator
  add_rostest(test/dummy_edgetpu_human_pose_estimator.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
  add_rostest(test/dummy_edgetpu_panorama_human_pose_estimator.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
  # semantic segmenter 
  add_rostest(test/dummy_edgetpu_semantic_segmenter.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
  add_rostest(test/dummy_edgetpu_panorama_semantic_segmenter.test
    DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv
  )
endif()
