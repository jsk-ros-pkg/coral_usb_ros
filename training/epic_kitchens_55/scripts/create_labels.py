# -*- coding: utf-8 -*-

import argparse
import os

from epic_kitchens_utils import epic_kitchens_bbox_label_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    text = u""
    for i, name in enumerate(epic_kitchens_bbox_label_names):
        txt = u"""{0} {1}
""".format(i, name)
        text = text + txt
    output_path = os.path.join(
        args.output_dir, 'labels.txt')
    with open(output_path, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
