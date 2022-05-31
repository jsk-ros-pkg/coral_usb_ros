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

from coral_usb.cfg import EdgeTPUHumanPoseEstimatorConfig
from coral_usb.cfg import EdgeTPUPanoramaHumanPoseEstimatorConfig
from coral_usb.node_base import EdgeTPUNodeBase
from coral_usb.posenet.pose_engine import PoseEngine
from coral_usb.util import get_panorama_sliced_image
from coral_usb.util import get_panorama_slices

from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import PeoplePose
from jsk_recognition_msgs.msg import PeoplePoseArray
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import CompressedImage


class EdgeTPUHumanPoseEstimator(EdgeTPUNodeBase):

    _engine_class = PoseEngine
    _config_class = EdgeTPUHumanPoseEstimatorConfig
    _default_model_file = 'package://coral_usb/python/coral_usb/posenet/' + \
        'models/mobilenet/' + \
        'posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'
    _default_label_file = None

    def __init__(self, model_file=None, namespace='~'):
        super(EdgeTPUHumanPoseEstimator, self).__init__(
            model_file=model_file, label_file=None, namespace=namespace)

        # for human pose estimator
        self.label_ids = [0]
        self.label_names = ['human']

        # publishers
        self.pub_pose = self.advertise(
            namespace + 'output/poses', PeoplePoseArray, queue_size=1)
        self.pub_rects = self.advertise(
            namespace + 'output/rects', RectArray, queue_size=1)
        self.pub_class = self.advertise(
            namespace + 'output/class', ClassificationResult, queue_size=1)

        # initialization
        self.img = None
        self.encoding = None
        self.header = None
        self.visibles = None
        self.points = None

    def _init_parameters(self):
        self.resized_H = self.engine.image_height
        self.resized_W = self.engine.image_width

    # overwrite _load_model for mirror argument
    def _load_model(self):
        self.engine = self._engine_class(
            self.model_file, device_path=self.device_path, mirror=False)
        self._init_parameters()

    def config_cb(self, config, level):
        self.score_thresh = config.score_thresh
        self.joint_score_thresh = config.joint_score_thresh
        return config

    def _process_result(
            self, poses, y_scale, x_scale, y_offset=None, x_offset=None):
        points = []
        key_names = []
        visibles = []
        bboxes = []
        labels = []
        scores = []
        for pose in poses:
            if pose.score < self.score_thresh:
                continue
            point = []
            key_name = []
            visible = []
            xs = []
            ys = []
            score = []
            for key_nm, keypoint in pose.keypoints.items():
                resized_key_y, resized_key_x = keypoint.yx
                key_y = resized_key_y / y_scale
                key_x = resized_key_x / x_scale
                if y_offset:
                    key_y = key_y + y_offset
                if x_offset:
                    key_x = key_x + x_offset
                point.append((key_y, key_x))
                xs.append(key_x)
                ys.append(key_y)
                score.append(keypoint.score)
                key_name.append(key_nm)
                if keypoint.score < self.joint_score_thresh:
                    visible.append(False)
                else:
                    visible.append(True)
            points.append(point)
            key_names.append(key_name)
            visibles.append(visible)
            y_max = int(np.round(max(ys)))
            y_min = int(np.round(min(ys)))
            x_max = int(np.round(max(xs)))
            x_min = int(np.round(min(xs)))
            bboxes.append([y_min, x_min, y_max, x_max])
            labels.append(0)
            scores.append(score)
        points = np.array(points, dtype=np.int)
        if len(points) == 0:
            points = points.reshape((len(points), 0, 2))
        else:
            points = points.reshape((len(points), -1, 2))
        visibles = np.array(visibles, dtype=np.bool)
        bboxes = np.array(bboxes, dtype=np.int).reshape((len(bboxes), 4))
        labels = np.array(labels, dtype=np.int)
        scores = np.array(scores, dtype=np.float)
        return points, key_names, visibles, bboxes, labels, scores

    def _estimate_step(self, img, y_offset=None, x_offset=None):
        resized_img = cv2.resize(img, (self.resized_W, self.resized_H))
        H, W, _ = img.shape
        y_scale = self.resized_H / H
        x_scale = self.resized_W / W
        poses, _ = self.engine.DetectPosesInImage(resized_img.astype(np.uint8))
        return self._process_result(
            poses, y_scale, x_scale, y_offset=y_offset, x_offset=x_offset)

    def _estimate(self, img):
        return self._estimate_step(img)

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

        points, key_names, visibles, bboxes, labels, scores \
            = self._estimate(img)

        poses_msg = PeoplePoseArray(header=msg.header)
        rects_msg = RectArray(header=msg.header)
        for point, key_name, visible, bbox, label, score in zip(
                points, key_names, visibles, bboxes, labels, scores):
            pose_msg = PeoplePose()
            for pt, key_nm, vs, sc in zip(point, key_name, visible, score):
                if vs:
                    key_y, key_x = pt
                    pose_msg.limb_names.append(key_nm)
                    pose_msg.scores.append(sc)
                    pose_msg.poses.append(
                        Pose(position=Point(x=key_x, y=key_y)))
            poses_msg.poses.append(pose_msg)
            y_min, x_min, y_max, x_max = bbox
            rect = Rect(
                x=x_min, y=y_min,
                width=x_max - x_min, height=y_max - y_min)
            rects_msg.rects.append(rect)

        cls_msg = ClassificationResult(
            header=msg.header,
            classifier=self.classifier_name,
            target_names=self.label_names,
            labels=labels,
            label_names=[self.label_names[lbl] for lbl in labels],
            label_proba=[np.average(score) for score in scores]
        )

        self.pub_pose.publish(poses_msg)
        self.pub_rects.publish(rects_msg)
        self.pub_class.publish(cls_msg)

        if self.enable_visualization:
            with self.lock:
                self.img = img
                self.encoding = encoding
                self.header = msg.header
                self.points = points
                self.visibles = visibles

    def visualize_cb(self, event):
        if (not self.visualize or self.img is None or self.encoding is None
                or self.points is None or self.visibles is None):
            return

        with self.lock:
            vis_img = self.img.copy()
            encoding = copy.copy(self.encoding)
            header = copy.deepcopy(self.header)
            points = self.points.copy()
            visibles = self.visibles.copy()

        # keypoints
        cmap = matplotlib.cm.get_cmap('hsv')
        for i, (point, visible) in enumerate(zip(points, visibles)):
            n = len(point) - 1
            for j, (pp, vis) in enumerate(zip(point, visible)):
                if vis:
                    py = pp[0] % vis_img.shape[0]
                    px = pp[1] % vis_img.shape[1]
                    rgba = np.array(cmap(1. * j / n))
                    color = rgba[:3] * 255
                    cv2.circle(vis_img, (px, py), 8, color, thickness=-1)

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


class EdgeTPUPanoramaHumanPoseEstimator(EdgeTPUHumanPoseEstimator):

    _config_class = EdgeTPUPanoramaHumanPoseEstimatorConfig

    def __init__(self, namespace='~'):
        super(EdgeTPUPanoramaHumanPoseEstimator, self).__init__(
            namespace=namespace
        )

    def _estimate(self, orig_img):
        _, orig_W = orig_img.shape[:2]
        panorama_slices = get_panorama_slices(
            orig_W, self.n_split, overlap=self.overlap)

        points = []
        key_names = []
        visibles = []
        bboxes = []
        labels = []
        scores = []
        for panorama_slice in panorama_slices:
            img = get_panorama_sliced_image(orig_img, panorama_slice)
            point, key_name, visible, bbox, label, score \
                = self._estimate_step(img, x_offset=panorama_slice.start)
            if len(point) > 0:
                points.append(point)
                key_names.extend(key_name)
                visibles.append(visible)
                bboxes.append(bbox)
                labels.append(label)
                scores.append(score)
        if len(points) > 0:
            points = np.concatenate(points, axis=0).astype(np.int)
            visibles = np.concatenate(visibles, axis=0).astype(np.bool)
            bboxes = np.concatenate(bboxes, axis=0).astype(np.int)
            labels = np.concatenate(labels, axis=0).astype(np.int)
            scores = np.concatenate(scores, axis=0).astype(np.float)
        else:
            points = np.empty((0, 0, 2), dtype=np.int)
            visibles = np.empty((0, 0, ), dtype=np.bool)
            bboxes = np.empty((0, 4), dtype=np.int)
            labels = np.empty((0, ), dtype=np.int)
            scores = np.empty((0, ), dtype=np.float)
        return points, key_names, visibles, bboxes, labels, scores

    def config_cb(self, config, level):
        self.n_split = config.n_split
        self.overlap = config.overlap
        config = super(
            EdgeTPUPanoramaHumanPoseEstimator, self).config_cb(
                config, level)
        return config
