import copy
import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import threading

# OpenCV import for python3.5
sys.path.remove('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA
import cv2  # NOQA
sys.path.append('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA

from chainercv.visualizations import vis_semantic_segmentation
from cv_bridge import CvBridge
from edgetpu.basic.basic_engine import BasicEngine
import rospkg
import rospy

from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image


class EdgeTPUSemanticSegmenter(ConnectionBasedTransport):

    def __init__(self):
        # get image_trasport before ConnectionBasedTransport subscribes ~input
        self.transport_hint = rospy.get_param('~image_transport', 'raw')
        rospy.loginfo("Using transport {}".format(self.transport_hint))

        super(EdgeTPUSemanticSegmenter, self).__init__()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coral_usb')
        self.bridge = CvBridge()
        self.classifier_name = rospy.get_param(
            '~classifier_name', rospy.get_name())
        model_file = os.path.join(
            pkg_path,
            './models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite')
        model_file = rospy.get_param('~model_file', model_file)
        label_file = rospy.get_param('~label_file', None)
        duration = rospy.get_param('~visualize_duration', 0.1)
        self.enable_visualization = rospy.get_param(
            '~enable_visualization', True)

        self.engine = BasicEngine(model_file)
        self.input_shape = self.engine.get_input_tensor_shape()[1:3]

        if label_file is None:
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
        else:
            self.label_ids, self.label_names = self._load_labels(label_file)

        self.pub_label = self.advertise(
            '~output/label', Image, queue_size=1)

        # visualize timer
        if self.enable_visualization:
            self.lock = threading.Lock()
            self.pub_image = self.advertise(
                '~output/image', Image, queue_size=1)
            self.pub_image_compressed = self.advertise(
                '~output/image/compressed', CompressedImage, queue_size=1)
            self.timer = rospy.Timer(
                rospy.Duration(duration), self.visualize_cb)
            self.img = None
            self.header = None
            self.label = None

    def subscribe(self):
        if self.transport_hint == 'compressed':
            self.sub_image = rospy.Subscriber(
                '{}/compressed'.format(rospy.resolve_name('~input')),
                CompressedImage, self.image_cb, queue_size=1, buff_size=2**26)
        else:
            self.sub_image = rospy.Subscriber(
                '~input', Image, self.image_cb, queue_size=1, buff_size=2**26)

    def unsubscribe(self):
        self.sub_image.unregister()

    @property
    def visualize(self):
        return self.pub_image.get_num_connections() > 0 or \
            self.pub_image_compressed.get_num_connections() > 0

    def config_callback(self, config, level):
        self.score_thresh = config.score_thresh
        self.top_k = config.top_k
        return config

    def _load_labels(self, path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            labels = {int(num): text.strip() for num, text in lines}
            return list(labels.keys()), list(labels.values())

    def image_cb(self, msg):
        if self.transport_hint == 'compressed':
            np_arr = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
        else:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        H, W = img.shape[:2]
        input_H, input_W = self.input_shape
        input_tensor = cv2.resize(img, (input_W, input_H))
        input_tensor = input_tensor.flatten()

        _, label = self.engine.run_inference(input_tensor)
        label = label.reshape(self.input_shape)
        label = cv2.resize(
            label, (W, H), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)

        label_msg = self.bridge.cv2_to_imgmsg(label, '32SC1')
        label_msg.header = msg.header
        self.pub_label.publish(label_msg)

        if self.enable_visualization:
            with self.lock:
                self.img = img
                self.header = msg.header
                self.label = label

    def visualize_cb(self, event):
        if (not self.visualize or self.img is None
                or self.header is None or self.label is None):
            return

        with self.lock:
            img = self.img.copy()
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
            vis_compressed_msg.format = "jpeg"
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            vis_compressed_msg.data = np.array(
                cv2.imencode('.jpg', vis_img_rgb)[1]).tostring()
            self.pub_image_compressed.publish(vis_compressed_msg)
