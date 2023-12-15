import copy
import matplotlib
import matplotlib.cm
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

from edgetpu.detection.engine import DetectionEngine
import PIL.Image
from resource_retriever import get_filename

from coral_usb.node_base import DummyEdgeTPUNodeBase
from coral_usb.node_base import EdgeTPUNodeBase
from coral_usb.util import generate_random_bbox
from coral_usb.util import get_panorama_sliced_image
from coral_usb.util import get_panorama_slices
from coral_usb.util import get_tile_sliced_image
from coral_usb.util import get_tile_slices
from coral_usb.util import non_maximum_suppression

from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import CompressedImage


class EdgeTPUDetectorBase(EdgeTPUNodeBase):

    _engine_class = DetectionEngine

    def __init__(self, model_file=None, label_file=None, namespace='~'):
        # initialization
        self.img = None
        self.encoding = None
        self.header = None
        self.bboxes = None
        self.labels = None
        self.scores = None

        super(EdgeTPUDetectorBase, self).__init__(
            model_file=model_file, label_file=label_file, namespace=namespace)

        # publishers
        self.pub_rects = self.advertise(
            namespace + 'output/rects', RectArray, queue_size=1)
        self.pub_class = self.advertise(
            namespace + 'output/class', ClassificationResult, queue_size=1)

    def config_cb(self, config, level):
        self.score_thresh = config.score_thresh
        self.top_k = config.top_k
        self.model_file = get_filename(config.model_file, False)
        if 'label_file' in config:
            self.label_file = get_filename(config.label_file, False)
            self._load_labels()
        if self.model_file is not None:
            self._load_model()
        return config

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

    def _detect_step(self, img, y_offset=None, x_offset=None):
        H, W = img.shape[:2]
        objs = self.engine.DetectWithImage(
            PIL.Image.fromarray(img), threshold=self.score_thresh,
            keep_aspect_ratio=True, relative_coord=True,
            top_k=self.top_k)
        return self._process_result(
            objs, H, W, y_offset=y_offset, x_offset=x_offset)

    def _detect(self, img):
        return self._detect_step(img)

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

        if self.enable_visualization:
            with self.lock:
                self.img = img
                self.encoding = encoding
                self.header = msg.header
                self.bboxes = bboxes
                self.labels = labels
                self.scores = scores

        if not self.always_publish and len(cls_msg.labels) <= 0:
            return
        self.pub_rects.publish(rect_msg)
        self.pub_class.publish(cls_msg)

    def visualize_cb(self, event):
        if (not self.visualize or self.img is None or self.encoding is None
                or self.header is None or self.bboxes is None
                or self.labels is None or self.scores is None):
            return

        with self.lock:
            vis_img = self.img.copy()
            encoding = copy.copy(self.encoding)
            header = copy.deepcopy(self.header)
            bboxes = self.bboxes.copy()
            labels = self.labels.copy()
            scores = self.scores.copy()

        # bbox
        cmap = matplotlib.cm.get_cmap('hsv')
        n = max(len(bboxes) - 1, 10)
        for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
            rgba = np.array(cmap(1. * i / n))
            color = rgba[:3] * 255
            label_text = '{}, {:.2f}'.format(self.label_names[label], score)
            p1y = max(bbox[0], 0)
            p1x = max(bbox[1], 0)
            p2y = min(bbox[2], vis_img.shape[0])
            p2x = min(bbox[3], vis_img.shape[1])
            cv2.rectangle(
                vis_img, (p1x, p1y), (p2x, p2y),
                color, thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(
                vis_img, label_text, (p1x, max(p1y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                thickness=2, lineType=cv2.LINE_AA)

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
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            vis_compressed_msg.data = np.array(
                cv2.imencode('.jpg', vis_img_bgr)[1]).tostring()
            self.pub_image_compressed.publish(vis_compressed_msg)


class EdgeTPUPanoramaDetectorBase(EdgeTPUDetectorBase):

    def __init__(self, model_file=None, label_file=None, namespace='~'):
        super(EdgeTPUPanoramaDetectorBase, self).__init__(
            model_file=model_file, label_file=label_file, namespace=namespace
        )

    def _detect(self, orig_img):
        _, orig_W = orig_img.shape[:2]
        panorama_slices = get_panorama_slices(
            orig_W, self.n_split, overlap=self.overlap)

        bboxes = []
        labels = []
        scores = []
        for panorama_slice in panorama_slices:
            img = get_panorama_sliced_image(orig_img, panorama_slice)
            bbox, label, score = self._detect_step(
                img, x_offset=panorama_slice.start)
            if len(bbox) > 0:
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

        if not self.nms:
            return bboxes, labels, scores

        # run with nms
        nms_bboxes = []
        nms_labels = []
        nms_scores = []
        for lbl in np.unique(labels):
            mask = labels == lbl
            nms_bbox = bboxes[mask]
            nms_label = labels[mask]
            nms_score = scores[mask]
            keep = non_maximum_suppression(
                nms_bbox, self.nms_thresh, nms_score)
            nms_bbox = nms_bbox[keep]
            nms_label = nms_label[keep]
            nms_score = nms_score[keep]
            if len(nms_bbox) > 0:
                nms_bboxes.append(nms_bbox)
                nms_labels.append(nms_label)
                nms_scores.append(nms_score)
        if len(nms_bboxes) > 0:
            nms_bboxes = np.concatenate(nms_bboxes, axis=0).astype(np.int)
            nms_labels = np.concatenate(nms_labels, axis=0).astype(np.int)
            nms_scores = np.concatenate(nms_scores, axis=0).astype(np.float)
        else:
            nms_bboxes = np.empty((0, 4), dtype=np.int)
            nms_labels = np.empty((0, ), dtype=np.int)
            nms_scores = np.empty((0, ), dtype=np.float)
        return nms_bboxes, nms_labels, nms_scores

    def config_cb(self, config, level):
        self.nms = config.nms
        self.nms_thresh = config.nms_thresh
        self.n_split = config.n_split
        self.overlap = config.overlap
        config = super(EdgeTPUPanoramaDetectorBase, self).config_cb(
            config, level)
        return config


class EdgeTPUTileDetectorBase(EdgeTPUDetectorBase):

    def __init__(self, model_file=None, label_file=None, namespace='~'):
        super(EdgeTPUTileDetectorBase, self).__init__(
            model_file=model_file, label_file=label_file, namespace=namespace
        )

    def _detect(self, orig_img):
        orig_H, orig_W = orig_img.shape[:2]
        tile_slices = get_tile_slices(
            orig_H, orig_W, overlap=self.overlap)

        bboxes = []
        labels = []
        scores = []
        for tile_slice in tile_slices:
            img = get_tile_sliced_image(orig_img, tile_slice)
            bbox, label, score = self._detect_step(
                img,
                y_offset=tile_slice[0].start,
                x_offset=tile_slice[1].start)
            if len(bbox) > 0:
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

        if not self.nms:
            return bboxes, labels, scores

        # run with nms
        nms_bboxes = []
        nms_labels = []
        nms_scores = []
        for lbl in np.unique(labels):
            mask = labels == lbl
            nms_bbox = bboxes[mask]
            nms_label = labels[mask]
            nms_score = scores[mask]
            keep = non_maximum_suppression(
                nms_bbox, self.nms_thresh, nms_score)
            nms_bbox = nms_bbox[keep]
            nms_label = nms_label[keep]
            nms_score = nms_score[keep]
            if len(nms_bbox) > 0:
                nms_bboxes.append(nms_bbox)
                nms_labels.append(nms_label)
                nms_scores.append(nms_score)
        if len(bboxes) > 0:
            nms_bboxes = np.concatenate(nms_bboxes, axis=0).astype(np.int)
            nms_labels = np.concatenate(nms_labels, axis=0).astype(np.int)
            nms_scores = np.concatenate(nms_scores, axis=0).astype(np.float)
        else:
            nms_bboxes = np.empty((0, 4), dtype=np.int)
            nms_labels = np.empty((0, ), dtype=np.int)
            nms_scores = np.empty((0, ), dtype=np.float)
        return nms_bboxes, nms_labels, nms_scores

    def config_cb(self, config, level):
        self.nms = config.nms
        self.nms_thresh = config.nms_thresh
        self.overlap = config.overlap
        config = super(EdgeTPUTileDetectorBase, self).config_cb(
            config, level)
        return config


class DummyEdgeTPUDetectorBase(EdgeTPUDetectorBase, DummyEdgeTPUNodeBase):

    def _detect_step(self, img, y_offset=None, x_offset=None):
        H, W = img.shape[:2]
        n = np.random.randint(5, 10)
        bboxes = generate_random_bbox(
            n, (H, W), 10, int(min(H, W) / 2.0))
        if y_offset:
            bboxes[:, 0] += y_offset
            bboxes[:, 2] += y_offset
        if x_offset:
            bboxes[:, 1] += x_offset
            bboxes[:, 3] += x_offset
        labels = np.random.randint(
            0, len(self.label_ids), size=(n, ))
        scores = np.random.uniform(0.0, 1.0, (n, ))
        return bboxes, labels, scores
