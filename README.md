# coral_usb_ros

ROS package for Coral Edge TPU USB Accelerator 

## Environment

- Ubuntu 16.04 + Kinetic
- Ubuntu 18.04 + Melodic

If you want to run this on Ubuntu 14.04 + Indigo, please see [indigo branch](https://github.com/knorth55/coral_usb_ros/tree/indigo).

If you want to run this on PR2, please see [pr2 branch](https://github.com/knorth55/coral_usb_ros/tree/pr2).

## Notice

We need `python3.5` or `python3.6` to run this package.

## Nodes
- Object detector: `edgetpu_object_detector.py`
- Face detector: `edgetpu_face_detector.py`
- Human Pose Estimator: `edgetpu_human_pose_estimator.py`

## Setup

### Install Edge TPU Dependencies

Follow this [page](https://coral.withgoogle.com/docs/accelerator/get-started/).

##### [Install the Edge TPU runtime](https://coral.withgoogle.com/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime)

```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-max # If you do not have USB3, install libedgetpu1-std
sudo apt-get install python3-edgetpu
```

#### [Install just the TensorFlow Lite interpreter](https://www.tensorflow.org/lite/guide/python)
```
sudo apt-get install python3-pip
wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp36-cp36m-linux_x86_64.whl
pip3 install tflite_runtime-1.14.0-cp36-cp36m-linux_x86_64.whl
```

####

### Workspace build (kinetic)

```
mkdir ~/ros/coral_ws/src
cd ~/ros/coral_ws/src
git clone https://github.com/knorth55/coral_usb_ros.git
wstool init
wstool merge coral_usb_ros/fc.rosinstall
wstool update
rosdep install --from-paths . --ignore-src -y -r
cd ~/ros/coral_ws
catkin init
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
catkin build
```

### Workspace build (melodic)

```
mkdir ~/ros/coral_ws/src
cd ~/ros/coral_ws/src
git clone https://github.com/knorth55/coral_usb_ros.git
wstool init
wstool merge coral_usb_ros/fc.rosinstall.melodic
wstool update
rosdep install --from-paths . --ignore-src -y -r
cd ~/ros/coral_ws
catkin init
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin build
```

### Model download

```
source ~/ros/coral_ws/devel/setup.bash
roscd coral_usb/scripts
python download_models.py
```

## Sample

### Run `image_publisher`

```bash
# source normal workspace, not edge tpu workspace
source ~/ros/kinetic/devel/setup.bash
rosrun jsk_perception image_publisher.py _file_name:=$(rospack find jsk_perception)/sample/object_detection_example_1.jpg
```

### Run Edge TPU launch

```bash
# source edge tpu workspace
source ~/ros/coral_ws/devel/setup.bash
# object detector
roslaunch coral_usb edgetpu_object_detector.launch INPUT_IMAGE:=/image_publisher/output
# face detector
roslaunch coral_usb edgetpu_face_detector.launch INPUT_IMAGE:=/image_publisher/output
# human pose estimator
roslaunch coral_usb edgetpu_human_pose_estimator.launch INPUT_IMAGE:=/image_publisher/output
```

### Run `image_view`

```bash
# source normal workspace, not edge tpu workspace
source ~/ros/kinetic/devel/setup.bash
rosrun image_view image_view image:=/edgetpu_object_detector/output/image
```
