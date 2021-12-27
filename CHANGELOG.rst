^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package coral_usb
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Merge pull request `#76 <https://github.com/knorth55/coral_usb_ros/issues/76>`_ from knorth55/fix-ci
* add python-numpy
* Merge pull request `#75 <https://github.com/knorth55/coral_usb_ros/issues/75>`_ from knorth55/add-device-error
* show device error for invalid device id or no device
* Merge pull request `#74 <https://github.com/knorth55/coral_usb_ros/issues/74>`_ from knorth55/knorth55-patch-1
* disable textlint
* Update linter.yaml
* Merge pull request `#73 <https://github.com/knorth55/coral_usb_ros/issues/73>`_ from knorth55/use-package
* use package:// instead of find
* update readme
* Contributors: Shingo Kitagawa

0.0.6 (2021-11-13)
------------------
* Merge pull request `#69 <https://github.com/knorth55/coral_usb_ros/issues/69>`_ from k-okada/noetic
  Information for Noetic user
* flake8
* warn if user does not belong to plugdev
* 99-coral-usb-ros.rules is not requried
  libedgetpu1-legacy-max installs rules under /lib/udev/rules.d
  ```
  $ dpkg -L libedgetpu1-legacy-max
  /.
  /lib
  /lib/udev
  /lib/udev/rules.d
  /lib/udev/rules.d/60-libedgetpu1-legacy-max.rules
  ```
* on noetic, we do not need to remove /opt/ros/{}/lib/python2.7/dist-packages path because 1) it is not exists, 2) we can use default opencv module
* update README to add noetic information
* md045
* Update README.md
* Merge pull request `#67 <https://github.com/knorth55/coral_usb_ros/issues/67>`_ from sktometometo/PR/update-rosinstall
  Update fc.rosinstall to delete jsk_common and catkin_virtualenv entry
* update fc.rosinstall to delete jsk_common and catkin_virtualenv entry
* Update README.md
* update readme
* update readme
* update readme
* Contributors: Kei Okada, Koki Shinjo, Shingo Kitagawa

0.0.5 (2021-08-13)
------------------
* markdownlint
* update readme
* update readme
* add EdgeTPUPanoramaSemanticSegmenter.cfg
* Merge pull request `#65 <https://github.com/knorth55/coral_usb_ros/issues/65>`_ from knorth55/panorama-nms
  add nms for panorama detection
* add start_dynamic_reconfigure
* add panorama human pose estimator in readme
* add EdgeTPUHumanPoseEstimatorConfig
* do not append when no bbox detected
* add edgetpu_panorama_face_detector in readme
* support panorama nodes in node_manager
* update readme to add panorama object detector
* add nms option dynamic reconfigure
* fix nms in detector_base
* use non_maximum_suppression for panorama detection
* use panorama config
* add non_maximum_suppression
* add EdgeTPUPanoramaFace/ObjectDetector.cfg
* Merge pull request `#64 <https://github.com/knorth55/coral_usb_ros/issues/64>`_ from knorth55/panorama-overlap
* update visualization functions for overlap
* add get_panorama_sliced_image
* Merge pull request `#62 <https://github.com/knorth55/coral_usb_ros/issues/62>`_ from sktometometo/feature/overlap-panorama-gap
* Merge pull request `#1 <https://github.com/knorth55/coral_usb_ros/issues/1>`_ from knorth55/feature/overlap-panorama-gap
  flake8
* flake8
* fix slice split and image concat process
* add overlap slice
* Contributors: Koki Shinjo, Shingo Kitagawa

0.0.4 (2021-06-16)
------------------
* set linetype
* Merge pull request `#60 <https://github.com/knorth55/coral_usb_ros/issues/60>`_ from 708yamaguchi/namespace-arg
* Merge branch 'master' into namespace-arg
* update linter workflows
* Change arg name
* Add arg to change namespace of edgetpu node
* use cv2 visualization for detector_base
* refactor human_pose_estimator
* Merge pull request `#59 <https://github.com/knorth55/coral_usb_ros/issues/59>`_ from k-okada/use_cv_draw_point
  use cv2.circle instead of vis_point/matplot.lot for effective cpu power
* use cv2.circle instead of vis_point/matplot.lot for effective cpu resources
* Merge pull request `#58 <https://github.com/knorth55/coral_usb_ros/issues/58>`_ from shmpwk/fix-model-label
  Change the way model file (and label file) are loaded for object detector and face detector
* use resource_retriever
* refactor detector_base
* refactor model path
* update README.md
* fix dynamic parameters
* update cfg
* Merge branch 'master' into fix-model-label
* edit readme for EdgeTPUFaceDetector param
* change the representation of model_file to adapt dynamic reconfigure for EdgeTPUFaceDetector
* change for EdgeTPUPanoramaObjectDetector
* change dynamic parameters
* ignore to commit __pycache\_\_
* change the representation of model_file and label_file to adapt to dynamic recongirure of EdgeTPUObjectDetector
* Contributors: Kei Okada, Naoya Yamaguchi, Shingo Kitagawa, Shumpei Wakabayashi, shmpwk

0.0.3 (2021-03-20)
------------------
* use lower version of pillow
* update pillows
* fix typo
* Merge pull request `#56 <https://github.com/knorth55/coral_usb_ros/issues/56>`_ from ishiguroJSK/patch-1
* Update README.md
* Update README.md
* Update README.md
* add overlap arguments
* fix panorama semantic_segmenter
* update default n_split arg
* pdate default n_split parameter
* add get_panorama_slices
* fix typo
* add edgetpu_panorama_semantic_segmenter
* refactor human_pose_estimator and detector_base
* return empty when no result is detected
* reshape points
* fix typo in human_pose_estimator
* add edgetpu_panorama_face_detector
* use n_split
* hacking
* remove panorama_detector_base.py
* add edgetpu_panorama_human_pose_estimator
* refactor panorama_detector_base
* add _process_result
* refactor detector_base
* add _estimate_pose
* add panorama_detector_base and panorama_object_detector
* add _detect_objects
* fix typo
* fix typo
* fix readme
* update reademe
* do not run jscpd linter
* fix dynamic_reconfigure namespace `#53 <https://github.com/knorth55/coral_usb_ros/issues/53>`_
  related to https://github.com/ros-visualization/rqt_reconfigure/issues/92
* Merge pull request `#50 <https://github.com/knorth55/coral_usb_ros/issues/50>`_ from knorth55/device-path
* add device_id
* ad knorth55/project-posenet
* remove posenet
* move all param in yaml and add yaml arg
* add resource_retriever in run_depend
* update readme version badge
* Merge pull request `#47 <https://github.com/knorth55/coral_usb_ros/issues/47>`_ from knorth55/add-switcher
* add default
* add prefix
* add node manager launch
* add node_manager.py
* add start and stop methods
* add services
* use get_filename
* add namespace args
* fix EdgeTPUDetectorBase
* move semantic_segmenter to python/
* move human_pose_estimator to python/
* move codes to python
* refactor nodes
* Merge pull request `#45 <https://github.com/knorth55/coral_usb_ros/issues/45>`_ from knorth55/use-legacy
* update key server
* use legacy version
* Merge pull request `#42 <https://github.com/knorth55/coral_usb_ros/issues/42>`_ from knorth55/add-human-rects
* update readme
* publish ClassificationResult in edgetpu_human_pose_estimator
* publish human rects in edgetpu_human_pose_estimator
* fix bgr -> rgb
* Merge pull request `#40 <https://github.com/knorth55/coral_usb_ros/issues/40>`_ from k-okada/add_compress
* Merge pull request `#41 <https://github.com/knorth55/coral_usb_ros/issues/41>`_ from knorth55/add-hacking
* add hacking in linter
* fix h103
* add documentation for compressed transport
* support compressed images, support IMAGE_TRANSPORT ros-args to launch files, publish compressed topic
* fix Dockerfile for build
* clean up apt cache in layers
* fix readme linter
* enable markdown
* Merge pull request `#39 <https://github.com/knorth55/coral_usb_ros/issues/39>`_ from knorth55/add-superlinter
* flake8
* update linter
* add superlinter
* Contributors: Kei Okada, Shingo Kitagawa, Yasuhiro Ishiguro

0.0.2 (2020-11-05)
------------------
* Update README.md
* Merge pull request `#37 <https://github.com/knorth55/coral_usb_ros/issues/37>`_ from knorth55/use-github-actions
* update README.md
* add github actions
* remove jsk_common
* remove .travis
* Merge pull request `#36 <https://github.com/knorth55/coral_usb_ros/issues/36>`_ from knorth55/fix-run-sh
* fix run.sh and train.sh in epic_kitchens_55
* fix run.sh to properly pass arguments
* Merge pull request `#35 <https://github.com/knorth55/coral_usb_ros/issues/35>`_ from Kanazawanaoaki/arg-run-gpu
  add --gpu args in train.sh
* add --gpu args
* Update README.md
* Merge pull request `#34 <https://github.com/knorth55/coral_usb_ros/issues/34>`_ from knorth55/add-vis-duration
* fix typo in README
* add enable_visualization doc
* add enable_visualization param
* update readme
* update edgetpu_semantic_segmenter gif
* add visualize_duration in edgetpu_semantic_segmenter
* add visualize_duration in edgetpu_face_detector
* add visualize_duration in edgetpu_object_detector
* add visualize_duration in edgetpu_human_pose_estimator
* Merge pull request `#33 <https://github.com/knorth55/coral_usb_ros/issues/33>`_ from k-okada/patch-2
* add more python3  modules to compile
* Merge pull request `#32 <https://github.com/knorth55/coral_usb_ros/issues/32>`_ from knorth55/training-data-augmentation
* add augmentation options for other models
* update training steps
* add more data_augmentation_options
* update CHANGELOG.rst
* fix urllib for python3
* fix .travis.roinstall
* add catkin_virtualenv 0.6.1 in rosinstall
* fix typo
* update Dockerfile
* update readme
* set git protocol
* use bionic for travis
* add more tests
* update rosinstalls
* update .travis
* Merge pull request `#27 <https://github.com/knorth55/coral_usb_ros/issues/27>`_ from knorth55/fix-build
* disable venv check
* use catkin_virtualenv 0.6.1
* remove catkin_virtualenv in kinetic
* Contributors: Kei Okada, Naoaki Kanazawa, Shingo Kitagawa

0.0.1 (2020-07-14)
------------------
* remove unnecesarry space
* update package.xml
* update .travis
* add opencv-python in kinetic
* update gpu for epic_kitchens_55
* Merge pull request `#23 <https://github.com/knorth55/coral_usb_ros/issues/23>`_ from knorth55/train-epic-kitchen
* update readme
* move epic_kitchens -> epic_kitchens_55
* update training parameters
* update train.sh parameters
* use smaller test dataset
* remove --num_eval_steps from labelme_voc
* add sample_1_of_n_eval_examples flag
* use NUM_EXAMPLES in labelme_voc
* use NUM_EXAMPLES
* refactor create_tf_record.py
* update train parameters
* add epic_kitchens training
* Merge pull request `#25 <https://github.com/knorth55/coral_usb_ros/issues/25>`_ from knorth55/update-posenet
* update modelfilepath
* update posenet to master
* remove trailing space
* fix BGR -> RGB
* fix create_tf_record.py
* update run.sh
* kitchen -> labelme_voc
* Merge pull request `#21 <https://github.com/knorth55/coral_usb_ros/issues/21>`_ from knorth55/add-semantic-segmentor
* update README.md
* flake8
* add EdgeTPUSemanticSegmenter
* download segmentation models
* Merge pull request `#20 <https://github.com/knorth55/coral_usb_ros/issues/20>`_ from knorth55/fix-dynamic-reconfigure
* update Dockerfile
* update Dockerfile
* add dynamic_reconfigure
* split fc.rosinstall to fc.rosinstall.kinetic
* fix typo in README.md
* Update README.md
* add training/labelbe_voc/README.md
* Merge pull request `#19 <https://github.com/knorth55/coral_usb_ros/issues/19>`_ from knorth55/add-docker
* add docker
* update readme
* Merge pull request `#18 <https://github.com/knorth55/coral_usb_ros/issues/18>`_ from knorth55/add-train-docker
* update run.sh
* udpate training/README.md
* Merge branch 'master' into add-train-docker
* add training/README.md
* update README
* move docker -> training/labelme_voc
* need to source /opt/ros/${ROS_DISTRO}/setup.bash, before source ~/coral_ws/deve/setup.bash
  otherwise we got
  ```
  $ roslaunch
  Traceback (most recent call last):
  File "/opt/ros/melodic/bin/roslaunch", line 34, in <module>
  import roslaunch
  ImportError: No module named roslaunch
  ```
* update travis
* melodic requires python3-opencv ? (`#16 <https://github.com/knorth55/coral_usb_ros/issues/16>`_)
* Merge pull request `#1 <https://github.com/knorth55/coral_usb_ros/issues/1>`_ from knorth55/add_docker
  add --gpu flag, --user flag, --userns flag and fix typo
* Merge branch 'add_docker' into add_docker
* fix typo in prepare_checkpoint_and_dataset.sh
  there is nothing in ckpt/
* add --userns=host for avoid root mount
* add --user to avoid mkdir in root
* enable --gpu
* set username to docker container name
* fix bugs prepare_checkpoint_and_dataset.sh; +chmod a+r /*
* fix typo
* add --gpu flag
* need to chmod ckpt
* support tensorbard
* check TTY and set -ti or not when running docker
* need to source /opt/ros/${ROS_DISTRO}/setup.bash, before source ~/cor… (`#17 <https://github.com/knorth55/coral_usb_ros/issues/17>`_)
* add edgetpu compile
* add docker file to train dataset
* need to source /opt/ros/${ROS_DISTRO}/setup.bash, before source ~/coral_ws/deve/setup.bash
  otherwise we got
  ```
  $ roslaunch
  Traceback (most recent call last):
  File "/opt/ros/melodic/bin/roslaunch", line 34, in <module>
  import roslaunch
  ImportError: No module named roslaunch
  ```
* update travis
* melodic requires python3-opencv ? (`#16 <https://github.com/knorth55/coral_usb_ros/issues/16>`_)
* Contributors: Kei Okada, Shingo Kitagawa

0.0.0 (2019-12-23)
------------------
* Merge pull request `#13 <https://github.com/knorth55/coral_usb_ros/issues/13>`_ from knorth55/update-travis
  update jsk_travis
* update jsk_travis
* add badges in readme
* Merge pull request `#11 <https://github.com/knorth55/coral_usb_ros/issues/11>`_ from knorth55/add-travis
  add travis
* use http
* update travis script
* remove opencv-python
* add -y in .travis_before_script.sh
* update travis
* add travis
* update visualization image
* update readme
* update readme
* Merge pull request `#10 <https://github.com/knorth55/coral_usb_ros/issues/10>`_ from kochigami/modify-readme
  modify README: /kinetic/ros => /ros/kinetic
* modify README: /kinetic/ros => /ros/kinetic
* Merge pull request `#9 <https://github.com/knorth55/coral_usb_ros/issues/9>`_ from YoshiaAbe/patch-1
  add -p to mkdir
* add -p to mkdir
* update gif
* add gif
* update readme
* update readme
* add node information in readme
* update README.md
* fix scaling in human pose estimator
* add model_file arg in edgetpu_face_detector.launch and edgetpu_human_pose_estimator.launch
* refactor edgetpu_object_detector.launch
* add +x in download_models.py
* Merge pull request `#7 <https://github.com/knorth55/coral_usb_ros/issues/7>`_ from makit0sh/object_detection_retrain
  added launch arg to change model for object detection
* added launch arg to change model for object detection
* update fc.rosinstall
* Update README.md
* add fc.rosintall.melodic
* Update README.md
* Merge pull request `#6 <https://github.com/knorth55/coral_usb_ros/issues/6>`_ from k-okada/master
  udpate for melodic users
* add more comments on edgetpu
* catkin_generate_virtualenv set to PYTHON_VERSION 3
* add instruction for melodic
* packge.xml add more python3 depends
* Update README.md
* set matplotlib version
* Update README.md
* fix launch name
* update LICENSE
* update README
* add EdgeTPUHumanPoseEstimator
* Merge pull request `#5 <https://github.com/knorth55/coral_usb_ros/issues/5>`_ from knorth55/add-face-detector
  Add face detector
* add edgetpu_face_detector.launch
* add edgetpu_face_detector.py
* Update README.md
* update fc.rosinstall
* add hot bugfix
* Merge pull request `#4 <https://github.com/knorth55/coral_usb_ros/issues/4>`_ from sktometometo/feature/fix_dependencies_20190915
  add python3 debian package dependencies
* update to use fixed jsk_topic_tools
  https://github.com/jsk-ros-pkg/jsk_common/pull/1636
* Merge pull request `#3 <https://github.com/knorth55/coral_usb_ros/issues/3>`_ from sktometometo/feature/fix_typo_20190915_2
  fix typo in REAMD.md
* add python3 debian package dependencies
* fix typo in REAMD.md
* Merge pull request `#2 <https://github.com/knorth55/coral_usb_ros/issues/2>`_ from sktometometo/remotes/sktometometo/feature/fix_typo
  fix typo and add rosdep install in README.md
* fix typo and add rosdep install in README.md
* fix edgetpu_object_detector
* fix typo
* add download_models script
* update readme
* add fc.rosinstall
* add respawn
* install launch directory
* add edgetpu_object_detector.py
* add coral_usb ros package
* Initial commit
* Contributors: Kanae Kochigami, Kei Okada, Koki Shinjo, Shingo Kitagawa, YoshiaAbe, jsk-fetchuser, makit0sh
