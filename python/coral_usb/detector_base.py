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

from chainercv.visualizations import vis_bbox
from cv_bridge import CvBridge
from edgetpu.basic.edgetpu_utils import EDGE_TPU_STATE_ASSIGNED
from edgetpu.basic.edgetpu_utils import EDGE_TPU_STATE_NONE
from edgetpu.basic.edgetpu_utils import ListEdgeTpuPaths
from edgetpu.detection.engine import DetectionEngine
import PIL.Image
from resource_retriever import get_filename
import rospy

from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image


class EdgeTPUDetectorBase(ConnectionBasedTransport):

    def __init__(self, model_file=None, label_file=None, namespace='~'):
        # get image_trasport before ConnectionBasedTransport subscribes ~input
        self.transport_hint = rospy.get_param(
            namespace + 'image_transport', 'raw')
        rospy.loginfo("Using transport {}".format(self.transport_hint))

        super(EdgeTPUDetectorBase, self).__init__()
        self.bridge = CvBridge()
        self.classifier_name = rospy.get_param(
            namespace + 'classifier_name', rospy.get_name())
        model_file = rospy.get_param(namespace + 'model_file', model_file)
        label_file = rospy.get_param(namespace + 'label_file', label_file)
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

        self.engine = DetectionEngine(self.model_file, device_path=device_path)
        if label_file is None:
            self.label_ids = None
            self.label_names = None
        else:
            self.label_ids, self.label_names = self._load_labels(label_file)

        self.pub_rects = self.advertise(
            namespace + 'output/rects', RectArray, queue_size=1)
        self.pub_class = self.advertise(
            namespace + 'output/class', ClassificationResult, queue_size=1)

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
            self.bboxes = None
            self.labels = None
            self.scores = None

    def start(self):
        self.engine = DetectionEngine(self.model_file)
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

    def _process_result(self, objs, H, W, y_offset=None, x_offset=None):
        bboxes = []
        labels = []
        scores = []
        for obj in objs:
            x_min, y_min, x_max, y_max = obj.bounding_box.flatten().tolist()
            y_max = int(np.round(y_max * H))
            y_min = int(np.round(y_min * H))
            if y_offset:
                y_max = y_max + y_offset
                y_min = y_min + y_offset
            x_max = int(np.round(x_max * W))
            x_min = int(np.round(x_min * W))
            if x_offset:
                x_max = x_max + x_offset
                x_min = x_min + x_offset
            bboxes.append([y_min, x_min, y_max, x_max])
            labels.append(self.label_ids.index(int(obj.label_id)))
            scores.append(obj.score)
        bboxes = np.array(bboxes, dtype=np.int).reshape((len(bboxes), 4))
        labels = np.array(labels, dtype=np.int)
        scores = np.array(scores, dtype=np.float)
        return bboxes, labels, scores

    def _detect_step(self, img):
        H, W = img.shape[:2]
        objs = self.engine.DetectWithImage(
            PIL.Image.fromarray(img), threshold=self.score_thresh,
            keep_aspect_ratio=True, relative_coord=True,
            top_k=self.top_k)
        return self._process_result(objs, H, W)

    def _detect(self, img):
        return self._detect_step(img)

    def image_cb(self, msg):
        if not hasattr(self, 'engine'):
            return
        if self.transport_hint == 'compressed':
            np_arr = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
        else:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        bboxes, labels, scores = self._detect(img)

        rect_msg = RectArray(header=msg.header)
        for bbox in bboxes:
            y_min, x_min, y_max, x_max = bbox
            rect = Rect(
                x=x_min, y=y_min,
                width=x_max - x_min, height=y_max - y_min)
            rect_msg.rects.append(rect)

        cls_msg = ClassificationResult(
            header=msg.header,
            classifier=self.classifier_name,
            target_names=self.label_names,
            labels=labels,
            label_names=[self.label_names[lbl] for lbl in labels],
            label_proba=scores)

        self.pub_rects.publish(rect_msg)
        self.pub_class.publish(cls_msg)

        if self.enable_visualization:
            with self.lock:
                self.img = img
                self.header = msg.header
                self.bboxes = bboxes
                self.labels = labels
                self.scores = scores

    def visualize_cb(self, event):
        if (not self.visualize or self.img is None
                or self.header is None or self.bboxes is None
                or self.labels is None or self.scores is None):
            return

        with self.lock:
            img = self.img.copy()
            header = copy.deepcopy(self.header)
            bboxes = self.bboxes.copy()
            labels = self.labels.copy()
            scores = self.scores.copy()

        fig = plt.figure(
            tight_layout={'pad': 0})
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        vis_bbox(
            img.transpose((2, 0, 1)),
            bboxes, labels, scores,
            label_names=self.label_names, ax=ax)
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


class EdgeTPUPanoramaDetectorBase(EdgeTPUDetectorBase):

    def __init__(self, model_file=None, label_file=None, namespace='~'):
        super(EdgeTPUPanoramaDetectorBase, self).__init__(
            model_file=model_file, label_file=label_file, namespace=namespace
        )
        self.n_split = rospy.get_param('~n_split', 2)

    def _detect(self, orig_img):
        _, orig_W = orig_img.shape[:2]
        x_offsets = np.arange(self.n_split) * int(orig_W / self.n_split)
        x_offsets = x_offsets.astype(np.int)
        bboxes = []
        labels = []
        scores = []
        for i in range(self.n_split):
            x_offset = x_offsets[i]
            if self.n_split == i + 1:
                x_end_offset = -1
            else:
                x_end_offset = x_offsets[i + 1]
            img = orig_img[:, x_offset:x_end_offset, :]
            bbox, label, score = self._detect_step(img)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        if len(bboxes) > 0:
            bboxes = np.concatenate(bboxes, axis=0).astype(np.int)
            labels = np.concatenate(labels, axis=0).astype(np.int)
            scores = np.concatenate(scores, axis=0).astype(np.float)
        else:
            bboxes = np.empty((0, 4), dtype=np.int)
            labels = np.empty((0, ), dtype=np.int)
            scores = np.empty((0, ), dtype=np.float)
        return bboxes, labels, scores
