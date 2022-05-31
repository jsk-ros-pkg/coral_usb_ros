import copy
import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# OpenCV import for python3
if os.environ['ROS_PYTHON_VERSION'] == '3':
    import cv2
else:
    sys.path.remove('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA
    import cv2  # NOQA
    sys.path.append('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA

from chainercv.visualizations import vis_semantic_segmentation
from edgetpu.basic.basic_engine import BasicEngine

from coral_usb.cfg import EdgeTPUPanoramaSemanticSegmenterConfig
from coral_usb.node_base import EdgeTPUNodeBase
from coral_usb.util import get_panorama_sliced_image
from coral_usb.util import get_panorama_slices

from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image


class EdgeTPUSemanticSegmenter(EdgeTPUNodeBase):

    _engine_class = BasicEngine
    _config_class = None
    _default_model_file = 'package://coral_usb/models/' + \
        'deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
    _default_label_file = None

    def __init__(self, model_file=None, label_file=None, namespace='~'):
        super(EdgeTPUSemanticSegmenter, self).__init__(
            model_file=model_file, label_file=label_file, namespace=namespace)

        # for semantic segmenter
        if (self.label_names is None and self.label_ids is None):
            self.label_names = [
                'background',
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor'
            ]
            self.label_ids = list(range(len(self.label_names)))

        # publishers
        self.pub_label = self.advertise(
            namespace + 'output/label', Image, queue_size=1)

        # initialization
        self.img = None
        self.encoding = None
        self.header = None
        self.label = None

    def _init_parameters(self):
        self.input_shape = self.engine.get_input_tensor_shape()[1:3]

    def _segment_step(self, img):
        H, W = img.shape[:2]
        input_H, input_W = self.input_shape
        input_tensor = cv2.resize(img, (input_W, input_H))
        input_tensor = input_tensor.flatten()
        _, label = self.engine.run_inference(input_tensor)
        label = label.reshape(self.input_shape)
        label = cv2.resize(
            label, (W, H), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        return label

    def _segment(self, img):
        return self._segment_step(img)

    def image_cb(self, msg):
        if not hasattr(self, 'engine'):
            return
        if self.transport_hint == 'compressed':
            np_arr = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
            encoding = msg.format
        else:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            encoding = msg.encoding

        label = self._segment(img)
        label_msg = self.bridge.cv2_to_imgmsg(label, '32SC1')
        label_msg.header = msg.header
        self.pub_label.publish(label_msg)

        if self.enable_visualization:
            with self.lock:
                self.img = img
                self.encoding = encoding
                self.header = msg.header
                self.label = label

    def visualize_cb(self, event):
        if (not self.visualize or self.img is None or self.encoding is None
                or self.header is None or self.label is None):
            return

        with self.lock:
            img = self.img.copy()
            encoding = copy.copy(self.encoding)
            header = copy.deepcopy(self.header)
            label = self.label.copy()

        fig = plt.figure(
            tight_layout={'pad': 0})
        ax = plt.subplot(1, 1, 1)
        ax.axis('off')
        ax, legend_handles = vis_semantic_segmentation(
            img.transpose((2, 0, 1)), label,
            label_names=self.label_names, alpha=0.7,
            all_label_names_in_legend=True, ax=ax)
        ax.legend(
            handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        vis_img = np.fromstring(
            fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_img.shape = (h, w, 3)
        fig.clf()
        plt.close()
        if self.pub_image.get_num_connections() > 0:
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, 'rgb8')
            # BUG: https://answers.ros.org/question/316362/sensor_msgsimage-generates-float-instead-of-int-with-python3/  # NOQA
            vis_msg.step = int(vis_msg.step)
            vis_msg.header = header
            self.pub_image.publish(vis_msg)
        if self.pub_image_compressed.get_num_connections() > 0:
            # publish compressed http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber  # NOQA
            vis_compressed_msg = CompressedImage()
            vis_compressed_msg.header = header
            # image format https://github.com/ros-perception/image_transport_plugins/blob/f0afd122ed9a66ff3362dc7937e6d465e3c3ccf7/compressed_image_transport/src/compressed_publisher.cpp#L116  # NOQA
            vis_compressed_msg.format = encoding + '; jpeg compressed bgr8'
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            vis_compressed_msg.data = np.array(
                cv2.imencode('.jpg', vis_img_rgb)[1]).tostring()
            self.pub_image_compressed.publish(vis_compressed_msg)


class EdgeTPUPanoramaSemanticSegmenter(EdgeTPUSemanticSegmenter):

    _config_class = EdgeTPUPanoramaSemanticSegmenterConfig

    def __init__(self, namespace='~'):
        super(EdgeTPUPanoramaSemanticSegmenter, self).__init__(
            namespace=namespace,
        )

    def _segment(self, orig_img):
        _, orig_W = orig_img.shape[:2]
        panorama_slices = get_panorama_slices(
            orig_W, self.n_split, overlap=False)

        label = []
        for panorama_slice in panorama_slices:
            img = get_panorama_sliced_image(orig_img, panorama_slice)
            lbl = self._segment_step(img)
            label.append(lbl)
        if len(label) > 0:
            label = np.concatenate(label, axis=1).astype(np.int32)
        else:
            label = np.empty((0, 0), dtype=np.int32)
        return label

    def config_cb(self, config, level):
        self.n_split = config.n_split
        return config
