# -*- coding: utf-8 -*-

import argparse
import os

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--ckpt_dir')
    parser.add_argument('--data_prefix')
    args = parser.parse_args()

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    data_prefix = args.data_prefix
    config_path = os.path.join(ckpt_dir, 'pipeline.config')

    class_names_path = os.path.join(data_dir, 'train', 'class_names.txt')
    test_record_path = os.path.join(
        data_dir, '{}_dataset_test.record'.format(data_prefix))
    with open(class_names_path, 'r') as f:
        class_names = f.readlines()
    class_names = [name.rstrip() for name in class_names]
    fg_class_names = class_names[1:]
    n_class = len(fg_class_names)
    n_example = sum(
        1 for _ in tf.python_io.tf_record_iterator(test_record_path))

    os.system(
        'sed -i "s%CKPT_DIR_TO_CONFIGURE%{0}%g" "{1}"'
        .format(ckpt_dir, config_path))
    os.system(
        'sed -i "s%DATASET_DIR_TO_CONFIGURE%{0}%g" "{1}"'
        .format(data_dir, config_path))
    os.system(
        'sed -i "s%NUM_CLASSES_TO_CONFIGURE%{0}%g" "{1}"'
        .format(n_class, config_path))
    os.system(
        'sed -i "s%DATA_PREFIX_TO_CONFIGURE%{0}%g" "{1}"'
        .format(data_prefix, config_path))
    os.system(
        'sed -i "s%NUM_EXAMPLES%{0}%g" "{1}"'
        .format(n_example, config_path))


if __name__ == '__main__':
    main()
