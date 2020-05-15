#!/bin/bash

#######################################
# Display usage and examples
#######################################
usage() {
  cat << END_OF_USAGE
  Downloads checkpoint and dataset needed for the tutorial.

  --dataset_dir        Set path to VOC dataset directory
  --network_type       Can be one of [mobilenet_v1_ssd, mobilenet_v2_ssd],
                       mobilenet_v2_ssd by default.
  --train_whole_model  Whether or not to train all layers of the model. false
                       by default, in which only the last few layers are trained.
  --num_training_steps Number of training steps to run, 500 by default.
  --num_eval_steps     Number of evaluation steps to run, 100 by default.
  --checkpoint_num     Checkpoint number, by default 50.
  --gpu                Specify GPU number, by defualt 0
  --help               Display this help.


Fine tuning in docker container

 $0 --train_whole_model false --network_type mobilenet_v2_ssd --num_training_steps 500 --num_eval_steps 100 --checkpoint_num 500 --daetaset_dir <path_to>/dataset

Whole retraining in docker container

 $0 --train_whole_model true --network_type mobilenet_v2_ssd --num_training_steps 50000 --num_eval_steps 2000 --checkpoint_num 50000 --daetaset_dir <path_to>/dataset

END_OF_USAGE
}

#######################################
# Display with color in console
#
# Arguments:
#   color: color code
#   message: message to display
#######################################
function message() {
  local color=$1; shift;
  local message=$@

  # Output command being executed
  echo -e "\e[${color}m${message}\e[0m"
}

#######################################
# Display command in console
#
# Arguments:
#   command: action to run
#######################################
function run_impl() {
  local command=$@

  # Output command being executed
  message 32 $command
  # actually run command
  $command
  result=$?
  return $result
}

#######################################
# Run a command
#   Return the exit status of the command
#######################################
function run() {
  run_impl $@ || exit $?
}

#######################################
# Display error message and exit
#######################################
function error() {
  message 31 $@, Exitting
  exit -1
}

#######################################
#######################################
#######################################

# 'docke run -ti IMAGE_NAME bash' for debug
if [ "$1" == "bash" -o "$1" == "/bin/bash" ]; then
    message 32 "execute $@"
    exec $@
fi

PORT=6006
train_whole_model=false
network_type=mobilenet_v2_ssd
num_training_steps=500
num_eval_steps=100
checkpoint_num=500
gpu=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_dir)
      DATASET_DIR=$2
      shift 2;;
    --port)
      PORT=$2
      shift 2;;
    --network_type)
      network_type=$2
      shift 2 ;;
    --train_whole_model)
      train_whole_model=$2
      shift 2;;
    --num_training_steps)
      num_training_steps=$2
      shift 2;;
    --num_eval_steps)
      num_eval_steps=$2
      shift 2;;
    --checkpoint_num)
      checkpoint_num=$2
      shift 2;;
    --gpu)
      gpu=$2
      shift 2;;
    --help)
      usage
      exit 0 ;;
    tensorboard)
      message 32 "execute $1 --port $PORT --logdir $DATASET_DIR/learn/train"
      exec $@ --port $PORT --logdir $DATASET_DIR/learn/train
      shift 1;;
    *)
      echo "ERROR: Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

[ ! -e "$DATASET_DIR" ] && error "Could not found '$DATASET_DIR' dataset directory";
run tree -L 2 $DATASET_DIR

message 32 "train_whole_model  : $train_whole_model"
message 32 "network_type       : $network_type"
message 32 "num_training_steps : $num_training_steps"
message 32 "num_eval_steps     : $num_eval_steps"
message 32 "checkpoint_num     : $checkpoint_num"
message 32 "DATASET_DIR        : $DATASET_DIR"

run cd /tensorflow/models/research/scripts

run ./prepare_checkpoint_and_dataset.sh --train_whole_model $train_whole_model --network_type $network_type --dataset_dir $DATASET_DIR
# retraining on GPU 0
export CUDA_VISIBLE_DEVICES=$gpu
message 32 "CUDA_VISIBLE_DEVICES : $gpu"
run ./retrain_detection_model.sh --num_training_steps $num_training_steps --num_eval_steps $num_eval_steps --dataset_dir $DATASET_DIR
# change to edgetpu model
run ./convert_checkpoint_to_edgetpu_tflite.sh --checkpoint_num $checkpoint_num --dataset_dir $DATASET_DIR
run cd /tensorflow/models/research/learn/models
run edgetpu_compiler output_tflite_graph.tflite
