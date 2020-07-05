# -*- coding: utf-8 -*-

import argparse
import os

from epic_kitchens_utils import epic_kitchens_bbox_label_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--ckpt_dir')
    args = parser.parse_args()

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    config_path = os.path.join(ckpt_dir, 'pipeline.config')

    n_class = len(epic_kitchens_bbox_label_names)
    os.system(
        'sed -i "s%CKPT_DIR_TO_CONFIGURE%{0}%g" "{1}"'
        .format(ckpt_dir, config_path))
    os.system(
        'sed -i "s%DATASET_DIR_TO_CONFIGURE%{0}%g" "{1}"'
        .format(data_dir, config_path))
    os.system(
        'sed -i "s%NUM_CLASSES_TO_CONFIGURE%{0}%g" "{1}"'
        .format(n_class, config_path))


if __name__ == '__main__':
    main()
