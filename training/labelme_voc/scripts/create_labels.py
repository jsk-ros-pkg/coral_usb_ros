# -*- coding: utf-8 -*-

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    root_dir = os.path.join(args.data_dir, 'train')
    class_names_path = os.path.join(root_dir, 'class_names.txt')
    with open(class_names_path, 'r') as f:
        class_names = f.readlines()
    class_names = [name.rstrip() for name in class_names]
    fg_class_names = class_names[1:]
    text = u""
    for i, name in enumerate(fg_class_names):
        txt = u"""{0} {1}
""".format(i, name)
        text = text + txt
    output_path = os.path.join(
        args.output_dir, 'labels.txt')
    with open(output_path, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
