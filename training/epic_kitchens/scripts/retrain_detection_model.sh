#!/bin/bash

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Starts retraining detection model.

  --dataset_dir                 Set path to VOC dataset directory
  --num_training_steps          Number of training steps to run, 500 by default.
  --sample_1_of_n_eval_examples Sample rate for evaluation, by default 1.
  --help                        Display this help.
END_OF_USAGE
}

num_training_steps=500
sample_1_of_n_eval_examples=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_training_steps)
      num_training_steps=$2
      shift 2 ;;
    --dataset_dir)
      DATASET_DIR=$2
      shift 2;;
    --sample_1_of_n_eval_examples)
      sample_1_of_n_eval_examples=$2
      shift 2;;
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

source "$PWD/constants.sh"

mkdir -p "${TRAIN_DIR}"

cd "${OBJ_DET_DIR}"
python ${RESEARCH_DIR}/object_detection/model_main.py \
  --pipeline_config_path="${CKPT_DIR}/pipeline.config" \
  --model_dir="${TRAIN_DIR}" \
  --num_train_steps="${num_training_steps}" \
  --sample_1_of_n_eval_examples="${sample_1_of_n_eval_examples}"
