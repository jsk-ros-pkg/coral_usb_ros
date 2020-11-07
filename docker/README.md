# docker image for `coral_usb_ros`

## Usage

```bash
# melodic
docker build --build-arg ROS_DISTRO=melodic --build-arg UBUNTU_VERSION=bionic -t knorth55/coral_usb_ros:melodic-latest .
# kinetic
docker build --build-arg ROS_DISTRO=kinetic --build-arg UBUNTU_VERSION=xenial -t knorth55/coral_usb_ros:kinetic-latest .
```
