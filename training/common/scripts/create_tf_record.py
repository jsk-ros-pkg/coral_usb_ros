from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import glob
import hashlib
import io
import logging
import numpy as np
import os

import pandas as pd
import PIL.Image
import tensorflow as tf
import xml.etree.ElementTree as ET

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw dataset.')
flags.DEFINE_string('output_dir', '', 'Dir to output TFRecord')
flags.DEFINE_string('data_format', '', 'Dataset format')
flags.DEFINE_string('data_prefix', '', 'Prefix for TFRecord')
FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)


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


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x))
            for filename, x in zip(gb.groups.keys(), gb.groups)]


def get_tf_example_labelme(
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
    for inst_lbl in np.unique(instance_label):
        inst_mask = instance_label == inst_lbl
        cls_count = np.bincount(class_label[inst_mask])
        # check if it only has background or not
        if len(cls_count) == 1:
            continue
        cls_lbl = np.argmax(cls_count[1:]) + 1
        classes.append(cls_lbl)
        classes_text.append(class_names[cls_lbl].encode('utf8'))
        yind, xind = np.where(inst_mask)
        xmin.append(float(xind.min() / float(width)))
        ymin.append(float(yind.min() / float(height)))
        xmax.append(float(xind.max() / float(width)))
        ymax.append(float(yind.max() / float(height)))

    if len(classes) == 0:
        print('No object annotation in {}'.format(img_path))
        logging.warn('No object annotation in {}'.format(img_path))

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


def get_tf_example_labelimg(group, image_dir, label_map_dict):
    with tf.gfile.GFile(
            os.path.join(image_dir, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(float(row['xmin']) / float(width))
        xmaxs.append(float(row['xmax']) / float(width))
        ymins.append(float(row['ymin']) / float(height))
        ymaxs.append(float(row['ymax']) / float(height))
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map_dict[row['class']])
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(
            classes_text),
        'image/object/class/label':
        dataset_util.int64_list_feature(classes),
    }))
    return example


def create_tf_record_labelme(data_dir, output_dir, set_name, data_prefix):
    output_path = os.path.join(
        output_dir, '{}_dataset_{}.record'.format(data_prefix, set_name))
    root_dir = os.path.join(data_dir, set_name)
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
    print('Reading dataset from {}.'.format(root_dir))
    logging.info('Reading dataset from {}.'.format(root_dir))
    for i, (img_path, class_label_path, instance_label_path) in enumerate(
            zip(img_paths, class_label_paths, instance_label_paths)):
        if i % 100 == 0:
            print('On image {} of {}'.format(i, len(img_paths)))
            logging.info('On image {} of {}'.format(i, len(img_paths)))

        tf_example = get_tf_example_labelme(
            img_path, class_label_path, instance_label_path, class_names)
        writer.write(tf_example.SerializeToString())
    writer.close()


def create_tf_record_labelimg(data_dir, output_dir, set_name, data_prefix):
    output_path = os.path.join(
        output_dir, '{}_dataset_{}.record'.format(data_prefix, set_name))
    xml_dir = os.path.join(data_dir, set_name)
    image_dir = xml_dir
    labels_path = os.path.join(
        data_dir, '{}_dataset_label_map.pbtxt'.format(data_prefix))
    writer = tf.python_io.TFRecordWriter(output_path)
    label_map_dict = label_map_util.get_label_map_dict(labels_path)
    examples = xml_to_csv(xml_dir)
    grouped = split(examples, 'filename')
    for group in grouped:
        example = get_tf_example_labelimg(group, image_dir, label_map_dict)
        writer.write(example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(output_path))


def create_tf_record(
        data_dir, output_dir, set_name, data_prefix, data_format
):
    if data_format == 'labelimg':
        create_tf_record_labelimg(
            data_dir, output_dir, set_name, data_prefix)
    elif data_format == 'labelme':
        create_tf_record_labelme(
            data_dir, output_dir, set_name, data_prefix)
    else:
        print('Unsupported data format: {}'.format(data_format))


def main(_):
    data_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
    data_format = FLAGS.data_format
    data_prefix = FLAGS.data_prefix
    for set_name in ['train', 'test']:
        create_tf_record(
            data_dir, output_dir, set_name,
            data_prefix, data_format=data_format)


if __name__ == '__main__':
    tf.app.run()
