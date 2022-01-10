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
flags.DEFINE_string('test_record_path', '', 'Root directory to test record.')
flags.DEFINE_string('ckpt_dir', '', 'Dir to ckpt')
FLAGS = flags.FLAGS


def main(_):
    ckpt_dir = FLAGS.ckpt_dir
    test_record_path = FLAGS.test_record_path
    config_path = os.path.join(ckpt_dir, 'pipeline.config')
    n_example = sum(
        1 for _ in tf.python_io.tf_record_iterator(test_record_path))
    os.system(
        'sed -i "s%NUM_EXAMPLES%{0}%g" "{1}"'
        .format(n_example, config_path))

if __name__ == '__main__':
    tf.app.run()
