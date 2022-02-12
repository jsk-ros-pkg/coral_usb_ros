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
DATA_PREFIX=${DATASET_NAME}
PORT=6006
DOCKER_OPTION=""
DOCKER_PORT_OPTION=""
RUN_TENSORBOARD=0
RUN_BASH=0

if [ ! -e $DATASET_DIR/train/JPEGImages -o \
       ! -e $DATASET_DIR/train/class_names.txt ]; then
    message 31 "Invalid VOC format annotation"
    exit 1
fi

if [ "$1" == "bash" -o "$1" == "/bin/bash" ]; then
    DOCKER_OPTION="";
    RUN_BASH=1
else
    for i in `seq 1 $#`; do
        if [ "${!i}" == "tensorboard" ]; then
            RUN_TENSORBOARD=1
        elif [ "${!i}" == "--port" ]; then
            j=$(expr $i + 1)
            PORT=${!j}
        fi;
    done
    DOCKER_OPTION="--data_format labelme";
    DOCKER_OPTION="${DOCKER_OPTION} --dataset_dir /tensorflow/models/research/${DATASET_NAME}";
    DOCKER_OPTION="${DOCKER_OPTION} --data_prefix ${DATA_PREFIX}";
    if [[  RUN_TENSORBOARD -eq 1 ]]; then
        DOCKER_PORT_OPTION="-p $PORT:$PORT"
    fi
fi

mkdir -p ${DATASET_DIR}/learn
if [ -t 1 ]; then
    TTY_OPT='-ti'
else
    TTY_OPT=''
fi

set -x
docker run --rm --privileged ${DOCKER_PORT_OPTION} \
    --gpus all \
    --user=$(id -u):$(id -g) --userns=host \
    --name $USER-train-edgetpu-object-detection-${DATASET_NAME}-$$ \
    --mount type=bind,src=${DATASET_DIR}/learn,dst=/tensorflow/models/research/learn \
    --mount type=bind,src=${DATASET_DIR},dst=/tensorflow/models/research/${DATASET_NAME} \
    ${TTY_OPT} train-edgetpu-object-detection ${DOCKER_OPTION} $@
set +x

if [ $RUN_BASH -eq 0 -a $RUN_TENSORBOARD -eq 0 ]; then
  message 32 "Done generating model file for edgetpu object detection"
  message 32 " - ${DATASET_DIR}/learn/models/labels.txt"
  message 32 " - ${DATASET_DIR}/learn/models/output_tflite_graph.tflite"
  message 32 " - ${DATASET_DIR}/learn/models/output_tflite_graph_edgetpu.tflite"
fi
