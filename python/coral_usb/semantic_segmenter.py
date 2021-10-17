import copy
import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import threading
import grp

# OpenCV import for python3.5
if os.environ['ROS_PYTHON_VERSION'] == '3':
    import cv2
else:
    sys.path.remove('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA
    import cv2  # NOQA
    sys.path.append('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA

from chainercv.visualizations import vis_semantic_segmentation
from cv_bridge import CvBridge
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.basic.edgetpu_utils import EDGE_TPU_STATE_ASSIGNED
from edgetpu.basic.edgetpu_utils import EDGE_TPU_STATE_NONE
from edgetpu.basic.edgetpu_utils import ListEdgeTpuPaths
from resource_retriever import get_filename
import rospy

from coral_usb.cfg import EdgeTPUPanoramaSemanticSegmenterConfig
from coral_usb.util import get_panorama_sliced_image
from coral_usb.util import get_panorama_slices

from dynamic_reconfigure.server import Server
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image


class EdgeTPUSemanticSegmenter(ConnectionBasedTransport):

    def __init__(self, namespace='~'):
        # get image_trasport before ConnectionBasedTransport subscribes ~input
        self.transport_hint = rospy.get_param(
            namespace + 'image_transport', 'raw')
        rospy.loginfo("Using transport {}".format(self.transport_hint))

        super(EdgeTPUSemanticSegmenter, self).__init__()
        self.bridge = CvBridge()
        self.classifier_name = rospy.get_param(
            namespace + 'classifier_name', rospy.get_name())
        model_file = 'package://coral_usb/models/' + \
            'deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
        model_file = rospy.get_param(namespace + 'model_file', model_file)
        label_file = rospy.get_param(namespace + 'label_file', None)
        if model_file is not None:
            self.model_file = get_filename(model_file, False)
        if label_file is not None:
            label_file = get_filename(label_file, False)
        self.duration = rospy.get_param(namespace + 'visualize_duration', 0.1)
        self.enable_visualization = rospy.get_param(
            namespace + 'enable_visualization', True)

        device_id = rospy.get_param(namespace + 'device_id', None)
        if device_id is None:
            device_path = None
        else:
            device_path = ListEdgeTpuPaths(EDGE_TPU_STATE_NONE)[device_id]
            assigned_device_paths = ListEdgeTpuPaths(EDGE_TPU_STATE_ASSIGNED)
            if device_path in assigned_device_paths:
                rospy.logwarn(
                    'device {} is already assigned: {}'.format(
                        device_id, device_path))

        if not grp.getgrnam('plugdev').gr_gid in os.getgroups():
            rospy.logerr('Current user does not belong to plugdev group')
            rospy.logerr('Please run `sudo adduser $(whoami) plugdev`')
            rospy.logerr('And make sure to re-login the terminal by `su -l $(whoami)`')

        self.engine = BasicEngine(self.model_file, device_path)
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

        # dynamic reconfigure
        self.start_dynamic_reconfigure(namespace)

        self.namespace = namespace
        self.pub_label = self.advertise(
            namespace + 'output/label', Image, queue_size=1)

        # visualize timer
        if self.enable_visualization:
            self.lock = threading.Lock()
            self.pub_image = self.advertise(
                namespace + 'output/image', Image, queue_size=1)
            self.pub_image_compressed = self.advertise(
                namespace + 'output/image/compressed',
                CompressedImage, queue_size=1)
            self.timer = rospy.Timer(
                rospy.Duration(self.duration), self.visualize_cb)
            self.img = None
            self.header = None
            self.label = None

    def start_dynamic_reconfigure(self, namespace):
        # dynamic reconfigure
        pass

    def start(self):
        self.engine = BasicEngine(self.model_file)
        self.input_shape = self.engine.get_input_tensor_shape()[1:3]
        self.subscribe()
        if self.enable_visualization:
            self.timer = rospy.Timer(
                rospy.Duration(self.duration), self.visualize_cb)

    def stop(self):
        self.unsubscribe()
        del self.sub_image
        if self.enable_visualization:
            self.timer.shutdown()
            del self.timer
        del self.engine

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

    def _load_labels(self, path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            labels = {int(num): text.strip() for num, text in lines}
            return list(labels.keys()), list(labels.values())

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
        else:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        label = self._segment(img)
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


class EdgeTPUPanoramaSemanticSegmenter(EdgeTPUSemanticSegmenter):
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

    def config_callback(self, config, level):
        self.n_split = config.n_split
        return config

    def start_dynamic_reconfigure(self, namespace):
        # dynamic reconfigure
        dyn_namespace = namespace
        if namespace == '~':
            dyn_namespace = ''
        self.srv = Server(
            EdgeTPUPanoramaSemanticSegmenterConfig,
            self.config_callback, namespace=dyn_namespace)
