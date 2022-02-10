#!/usr/bin/env bash

set -e

if [[ $1 = "" ]]; then
    echo "Please specify ros distro"
    echo "Usage: build.bash [ROS_DISTRO]"
    exit 1;
fi

ROS_DISTRO=$1;
IMAGE_NAME="knorth55/coral_usb_ros:${ROS_DISTRO}-latest";
if [[ ${ROS_DISTRO} = "noetic" ]]; then
    UBUNTU_VERSION="focal";
elif [[ ${ROS_DISTRO} = "melodic" ]]; then
    UBUNTU_VERSION="bionic";
elif [[ ${ROS_DISTRO} = "kinetic" ]]; then
    UBUNTU_VERSION="xenial";
elif [[ ${ROS_DISTRO} = "indigo" ]]; then
    UBUNTU_VERSION="trusty";
else
    echo "Unsupported distro: $1";
    exit 1;
fi

docker build --build-arg ROS_DISTRO=${ROS_DISTRO} \
             --build-arg UBUNTU_VERSION=${UBUNTU_VERSION} \
             -t ${IMAGE_NAME} .

set +e
