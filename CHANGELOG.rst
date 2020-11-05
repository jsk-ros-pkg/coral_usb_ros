^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package coral_usb
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
