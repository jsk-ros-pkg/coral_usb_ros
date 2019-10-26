# coral_usb_ros

ROS package for Coral Edge TPU USB Accelerator 

## Environment

- Ubuntu 16.04 + Kinetic
- Ubuntu 18.04 + Melodic

If you want to run this on Ubuntu 14.04 + Indigo, please see [indigo branch](https://github.com/knorth55/coral_usb_ros/tree/indigo).

If you want to run this on PR2, please see [pr2 branch](https://github.com/knorth55/coral_usb_ros/tree/pr2).

## Notice

We need `python3.5` or `python3.6` to run this package.

## ROS Node list
- Object detector: `edgetpu_object_detector.py`
- Face detector: `edgetpu_face_detector.py`
- Human Pose Estimator: `edgetpu_human_pose_estimator.py`

For more detailed information, see [here](https://github.com/knorth55/coral_usb_ros#ROS-node-information).

## Setup

### Install Edge TPU Dependencies

Follow this [page](https://coral.withgoogle.com/docs/accelerator/get-started/).

##### [Install the Edge TPU runtime](https://coral.withgoogle.com/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime)

```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
# If you do not have USB3, install libedgetpu1-std
sudo apt-get install libedgetpu1-max
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
source /opt/kinetic/ros/setup.bash
mkdir ~/coral_ws/src
cd ~/coral_ws/src
git clone https://github.com/knorth55/coral_usb_ros.git
wstool init
wstool merge coral_usb_ros/fc.rosinstall
wstool update
rosdep install --from-paths . --ignore-src -y -r
cd ~/coral_ws
catkin init
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
catkin build
```

### Workspace build (melodic)

```
source /opt/melodic/ros/setup.bash
mkdir ~/coral_ws/src
cd ~/coral_ws/src
git clone https://github.com/knorth55/coral_usb_ros.git
wstool init
wstool merge coral_usb_ros/fc.rosinstall.melodic
wstool update
rosdep install --from-paths . --ignore-src -y -r
cd ~/coral_ws
catkin init
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin build
```

### Model download

```
source ~/coral_ws/devel/setup.bash
roscd coral_usb/scripts
rosrun coral_usb download_models.py
```

## Sample

### Run `image_publisher`

```bash
# source normal workspace, not edge tpu workspace
# /opt/ros/kinetic/setup.bash or /opt/ros/melodic/setup.bash
source /opt/ros/kinetic/setup.bash
rosrun jsk_perception image_publisher.py _file_name:=$(rospack find jsk_perception)/sample/object_detection_example_1.jpg
```

### Run Edge TPU launch

```bash
# source edge tpu workspace
source ~/coral_ws/devel/setup.bash
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
# /opt/ros/kinetic/setup.bash or /opt/ros/melodic/setup.bash
source /opt/ros/kinetic/setup.bash
# object detector
rosrun image_view image_view image:=/edgetpu_object_detector/output/image
# face detector
rosrun image_view image_view image:=/edgetpu_face_detector/output/image
# human pose estimator
rosrun image_view image_view image:=/edgetpu_human_pose_estimator/output/image

```

## ROS Node information

### Object detector: `edgetpu_object_detector.py`

**Subscribing Topic**

- `~input/image` (`sensor_msgs/Image`)

Input image

**Publishing Topic**

- `~output/rects` (`jsk_recognition_msgs/RectArray`)

Rectangles of detected objects

- `~output/class` (`jsk_recognition_msgs/ClassificationResult`)

Classification results of detected objects

- `~output/image` (`sensor_msgs/Image`)

Visualization of detection results

**Parameters**

- `~classifier_name` (`String`, default: `rospy.get_name()`)

Classifier name

- `~model_file` (`String`, default: `$(rospack find coral_usb)/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite`)

Model file path

- `~label_file` (`String`, default: `$(rospack find coral_usb)/models/coco_labels.txt`)

Label file path.

**Dynamic parameters**

- `~score_thresh`: (`Float`, default: `0.6`)

Score threshold for object detection

- `~top_k`: (`Int`, default: `100`)

Maximum number of detected objects


### Face detector: `edgetpu_face_detector.py`

**Subscribing Topic**

- `~input/image` (`sensor_msgs/Image`)

Input image

**Publishing Topic**

- `~output/rects` (`jsk_recognition_msgs/RectArray`)

Rectangles of detected faces

- `~output/class` (`jsk_recognition_msgs/ClassificationResult`)

Classification results of detected faces

- `~output/image` (`sensor_msgs/Image`)

Visualization of detection results

**Parameters**

- `~classifier_name` (`String`, default: `rospy.get_name()`)

Classifier name

- `~model_file` (`String`, default: `$(rospack find coral_usb)/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite`)

Model file path

**Dynamic parameters**

- `~score_thresh`: (`Float`, default: `0.6`)

Score threshold for face detection

- `~top_k`: (`Int`, default: `100`)

Maximum number of detected faces


### Human pose estimator: `edgetpu_human_pose_estimator.py`

**Subscribing Topic**

- `~input/image` (`sensor_msgs/Image`)

Input image

**Publishing Topic**

- `~output/poses` (`jsk_recognition_msgs/PeoplePoseArray`)

Estimated human poses

- `~output/image` (`sensor_msgs/Image`)

Visualization of estimation results

**Parameters**

- `~classifier_name` (`String`, default: `rospy.get_name()`)

Classifier name

- `~model_file` (`String`, default: `$(rospack find coral_usb)/python/coral_usb/posenet/models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite`)

Model file path

**Dynamic parameters**

- `~score_thresh`: (`Float`, default: `0.2`)

Score threshold for human pose estimation

- `~joint_score_thresh`: (`Float`, default: `0.2`)

Score threshold of each joint for human pose estimation
