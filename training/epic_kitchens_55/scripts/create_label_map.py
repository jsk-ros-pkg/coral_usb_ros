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
        txt = u"""item {{
  id: {0}
  name: '{1}'
}}

""".format(i+1, name)
        text = text + txt
    text = text[:-1]
    output_path = os.path.join(
        args.output_dir, 'epic_kitchens_dataset_label_map.pbtxt')
    with open(output_path, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
