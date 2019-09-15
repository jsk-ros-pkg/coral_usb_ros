# coral_usb_ros

ROS package for Coral Edge TPU USB Accelerator 

## Notice

We need `python3.5` to run this package.


## Setup

### Install Edge TPU Dependencies

Follow this [page](https://coral.withgoogle.com/docs/accelerator/get-started/).

### Workspace build 

```
mkdir ~/ros/coral_ws/src
cd ~/ros/coral_ws/src
git clone git@github.com:knorth55/coral_usb_ros.git
ln -sf ~/ros/coral_ws/src/coral_usb_ros/fc.rosinstall ~/ros/coral_ws/src/.rosinstall
wstool up
rosdep install --from-paths . --ignore-src -y -r
cd ~/ros/coral_ws
catkin init
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
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
rosrun jsk_perception image_publisher _file_name:=$(rospack find jsk_perception)/sample/object_detection_example_1.jpg
```

### Run Edge TPU launch

```bash
# source edge tpu workspace
source ~/ros/coral_ws/devel/setup.bash
roslaunch coral_usb edgetpu_object_detector.launch INPUT_IMAGE:=/image_publisher/output
```

### Run `image_view`

```bash
# source normal workspace, not edge tpu workspace
source ~/ros/kinetic/devel/setup.bash
rosrun image_view image_view image:=/edgetpu_object_detector/output/image
```
