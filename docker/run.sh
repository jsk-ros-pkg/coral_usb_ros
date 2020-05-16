#!/bin/bash

function message() {
  local color=$1; shift;
  local message=$@

  # Output command being executed
  echo -e "\e[${color}m${message}\e[0m"
}

THIS_DIR="$PWD"

if [ -z "$1" ]; then
    message 31 "Set path to dataset is required"
    exit 1
fi

DATASET_DIR=$(realpath $1); shift 1;
DATASET_NAME=$(basename $DATASET_DIR)
if [ ! -e $DATASET_DIR/train/JPEGImages -o \
       ! -e $DATASET_DIR/train/class_names.txt ]; then
    message 31 "Invalid VOC format annotation"
    exit 1
fi

PORT=6006
if [ "$1" == "--port" ]; then
    PORT=$2
    shift 2
fi
if [ "$1" == "bash" -o "$1" == "/bin/bash" ]; then
    DOCKER_OPTION="";
else
    DOCKER_OPTION="--dataset_dir /tensorflow/models/research/${DATASET_NAME}";
fi
if [ "$1" == "tensorboard" ]; then
    TENSORBOARD_OPTION="--port $PORT";
else
    TENSORBOARD_OPTION="";
fi


mkdir -p ${DATASET_DIR}/learn
if [ -t 1 ]; then
    TTY_OPT='-ti'
else
    TTY_OPT=''
fi

set -x
docker run --rm --privileged -p $PORT:$PORT \
    --gpus all \
    --name train-edgetpu-object-detection-${DATASET_NAME}-$$ \
    --user=$(id -u):$(id -g) \
    --mount type=bind,src=${DATASET_DIR}/learn,dst=/tensorflow/models/research/learn \
    --mount type=bind,src=${DATASET_DIR},dst=/tensorflow/models/research/${DATASET_NAME} \
    ${TTY_OPT} train-edgetpu-object-detection ${DOCKER_OPTION} ${TENSORBOARD_OPTION} $@
set +x

message 32 "Done generating model file for edgetpu object detection"
message 32 " - ${DATASET_DIR}/learn/models/labels.txt"
message 32 " - ${DATASET_DIR}/learn/models/output_tflite_graph.tflite"
message 32 " - ${DATASET_DIR}/learn/models/output_tflite_graph_edgetpu.tflite"


