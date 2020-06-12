from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import subprocess

import pandas as pd
import PIL.Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

from epic_kitchens_utils import epic_kitchens_bbox_label_names
from epic_kitchens_utils import epic_kitchens_data_dir_names


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw dataset.')
flags.DEFINE_string('anno_dir', '', 'Root directory to raw annotation.')
flags.DEFINE_string('output_dir', '', 'Dir to output TFRecord')
flags.DEFINE_string('ckpt_dir', '', 'Dir to ckpt')
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


def get_tf_example(img_path, img_id, annotation, class_names):
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

    part_id, video_id, frame_id = img_id.split('/')
    anno_mask = \
        (annotation['participant_id'] == part_id) \
        & (annotation['video_id'] == video_id)  \
        & (annotation['frame'] == int(frame_id))
    anno_data = annotation[anno_mask]

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    for bb_str, cls_lbl in anno_data[['bounding_boxes', 'noun_class']].values:
        if cls_lbl == 0:
            continue
        bb = eval(bb_str)
        if len(bb) == 0:
            continue
        for bb_ in bb:
            ymin, xmin, h, w = bb_
            ymax = ymin + h
            xmax = xmin + w
            xmins.append(float(xmin) / float(width))
            xmaxs.append(float(xmax) / float(width))
            ymins.append(float(ymin) / float(height))
            ymaxs.append(float(ymax) / float(height))
            classes.append(cls_lbl)
            classes_text.append(class_names[cls_lbl - 1])

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
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        }))
    return example


def create_tf_record(root_dir, dir_names, annotation, output_path):
    ids = []
    for dir_name in dir_names:
        data_dir = os.path.join(root_dir, dir_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            tar_path = os.path.join(root_dir, '{}.tar'.format(dir_name))
            subprocess.call(['tar', 'xvf', tar_path, '-C', data_dir])
        for img_name in sorted(os.listdir(data_dir)):
            ids.append('{}/{}'.format(dir_name, img_name.split('.')[0]))

    anno_ids = annotation[annotation['bounding_boxes'] != '[]']
    anno_ids = anno_ids[['participant_id', 'video_id', 'frame']].values
    anno_ids[:, 2] = ['{0:010d}'.format(x) for x in anno_ids[:, 2]]
    anno_ids = ['/'.join(x) for x in anno_ids]
    ids = sorted(list(set(ids) & set(anno_ids)))

    writer = tf.python_io.TFRecordWriter(output_path)
    print('Reading dataset from {}.'.format(root_dir))
    logging.info('Reading dataset from {}.'.format(root_dir))
    for i, img_id in enumerate(ids):
        img_path = os.path.join(root_dir, '{}.jpg'.format(img_id))
        if i % 100 == 0:
            print('On image {} of {}'.format(i, len(ids)))
            logging.info('On image {} of {}'.format(i, len(ids)))
        tf_example = get_tf_example(
            img_path, img_id, annotation, epic_kitchens_bbox_label_names)
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    data_dir = FLAGS.data_dir
    anno_dir = FLAGS.anno_dir
    ckpt_dir = FLAGS.ckpt_dir
    root_dir = os.path.join(data_dir, 'object_detection_images/train')
    anno_path = os.path.join(anno_dir, 'EPIC_train_object_labels.csv')
    annotation = pd.read_csv(anno_path)

    train_dir_names, test_dir_names = train_test_split(
        epic_kitchens_data_dir_names, test_size=0.05, shuffle=None)

    train_output_path = os.path.join(
        FLAGS.output_dir,
        'epic_kitchens_dataset_train.record')
    if not os.path.exists(train_output_path):
        create_tf_record(
            root_dir, train_dir_names, annotation, train_output_path)
    test_output_path = os.path.join(
        FLAGS.output_dir,
        'epic_kitchens_dataset_test.record')
    if not os.path.exists(test_output_path):
        create_tf_record(
            root_dir, test_dir_names, annotation, test_output_path)

    config_path = os.path.join(ckpt_dir, 'pipeline.config')
    n_example = sum(
        1 for _ in tf.python_io.tf_record_iterator(test_output_path))
    os.system(
        'sed -i "s%NUM_EXAMPLES%{0}%g" "{1}"'
        .format(n_example, config_path))


if __name__ == '__main__':
    tf.app.run()
