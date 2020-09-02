#!/usr/bin/env python

import os
import tarfile
import urllib

import rospkg

url = 'https://dl.google.com/coral/canned_models/all_models.tar.gz'
seg_urls = [
    'https://github.com/google-coral/edgetpu/raw/master/test_data/'
    'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite',
    'https://github.com/google-coral/edgetpu/raw/master/test_data/'
    'deeplabv3_mnv2_pascal_quant_edgetpu.tflite',
]


rospack = rospkg.RosPack()
pkg_path = rospack.get_path('coral_usb')
models_path = os.path.join(pkg_path, './models')
tar_path = os.path.join(models_path, 'all_models.tar.gz')
if not os.path.exists(models_path):
    os.makedirs(models_path)
urllib.urlretrieve(url, tar_path)

with tarfile.open(tar_path) as tar_f:
    tar_f.extractall(models_path)

for seg_url in seg_urls:
    filename = seg_url.split('/')[-1]
    filepath = os.path.join(models_path, filename)
    urllib.urlretrieve(seg_url, filepath)
