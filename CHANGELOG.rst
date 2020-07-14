^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package coral_usb
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
