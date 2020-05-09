from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import numpy as np
import os

import PIL.Image
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw dataset.')
flags.DEFINE_string('output_dir', '', 'Dir to output TFRecord')
FLAGS = flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_tf_example(
        img_path, class_label_path, instance_label_path, class_names):
    # image
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(image.width)
    height = int(image.height)
    filename = img_path.split('/')[-1]

    class_label = np.load(class_label_path)
    instance_label = np.load(instance_label_path)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    R = np.max(instance_label)
    for inst_lbl in range(R):
        if inst_lbl == 0:
            continue
        inst_mask = instance_label == inst_lbl
        cls_lbl = np.argmax(np.bincount(class_label[inst_mask]))
        classes.append(cls_lbl)
        classes_text.append(class_names[cls_lbl].encode('utf8'))
        yind, xind = np.where(inst_mask)
        xmin.append(float(xind.min() / float(width)))
        ymin.append(float(yind.min() / float(height)))
        xmax.append(float(xind.max() / float(width)))
        ymax.append(float(yind.max() / float(height)))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(filename.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        }))
    return example


def create_tf_record(root_dir, output_path):
    imgs_dir = os.path.join(root_dir, 'JPEGImages')
    class_labels_dir = os.path.join(root_dir, 'SegmentationClass')
    instance_labels_dir = os.path.join(root_dir, 'SegmentationObject')

    class_names_path = os.path.join(root_dir, 'class_names.txt')
    with open(class_names_path, 'r') as f:
        class_names = f.readlines()
    class_names = [name.rstrip() for name in class_names]

    img_paths = []
    class_label_paths = []
    instance_label_paths = []
    imgs_dir = os.path.join(root_dir, 'JPEGImages')
    class_labels_dir = os.path.join(root_dir, 'SegmentationClass')
    instance_labels_dir = os.path.join(root_dir, 'SegmentationObject')
    for img_name in sorted(os.listdir(imgs_dir)):
        img_path = os.path.join(imgs_dir, img_name)
        basename = img_name.rstrip('.jpg')
        class_label_path = os.path.join(
            class_labels_dir, basename + '.npy')
        instance_label_path = os.path.join(
            instance_labels_dir, basename + '.npy')
        img_paths.append(img_path)
        class_label_paths.append(class_label_path)
        instance_label_paths.append(instance_label_path)

    writer = tf.python_io.TFRecordWriter(output_path)
    print('Reading from 73B2 kitchen dataset.')
    logging.info('Reading from 73B2 kitchen dataset.')
    for i, (img_path, class_label_path, instance_label_path) in enumerate(
            zip(img_paths, class_label_paths, instance_label_paths)):
        if i % 100 == 0:
            print('On image {} of {}'.format(i, len(img_paths)))
            logging.info('On image {} of {}'.format(i, len(img_paths)))

        tf_example = get_tf_example(
            img_path, class_label_path, instance_label_path, class_names)
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    data_dir = FLAGS.data_dir
    for set_name in ['train', 'test']:
        root_dir = os.path.join(data_dir, set_name)
        output_path = os.path.join(
            FLAGS.output_dir, 'kitchen_dataset_{}.record'.format(set_name))
        create_tf_record(root_dir, output_path)


if __name__ == '__main__':
    tf.app.run()
