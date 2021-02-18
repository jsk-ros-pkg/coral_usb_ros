import copy
import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import threading

# OpenCV import for python3.5
sys.path.remove('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA
import cv2  # NOQA
sys.path.append('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA

from chainercv.visualizations import vis_point
from cv_bridge import CvBridge
from edgetpu.basic.edgetpu_utils import EDGE_TPU_STATE_ASSIGNED
from edgetpu.basic.edgetpu_utils import EDGE_TPU_STATE_NONE
from edgetpu.basic.edgetpu_utils import ListEdgeTpuPaths
from resource_retriever import get_filename
import rospkg
import rospy

from coral_usb.util import get_panorama_slices

from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import PeoplePose
from jsk_recognition_msgs.msg import PeoplePoseArray
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

from coral_usb.cfg import EdgeTPUHumanPoseEstimatorConfig
from coral_usb.posenet.pose_engine import PoseEngine


class EdgeTPUHumanPoseEstimator(ConnectionBasedTransport):

    def __init__(self, namespace='~'):
        # get image_trasport before ConnectionBasedTransport subscribes ~input
        self.transport_hint = rospy.get_param(
            namespace + 'image_transport', 'raw')
        rospy.loginfo("Using transport {}".format(self.transport_hint))

        super(EdgeTPUHumanPoseEstimator, self).__init__()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coral_usb')
        self.bridge = CvBridge()
        self.classifier_name = rospy.get_param(
            namespace + 'classifier_name', rospy.get_name())
        model_file = os.path.join(
            pkg_path,
            './python/coral_usb/posenet/models/mobilenet/'
            'posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
        model_file = rospy.get_param(namespace + 'model_file', model_file)
        if model_file is not None:
            self.model_file = get_filename(model_file, False)
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

        self.engine = PoseEngine(
            self.model_file, mirror=False, device_path=device_path)
        self.resized_H = self.engine.image_height
        self.resized_W = self.engine.image_width

        # only for human
        self.label_ids = [0]
        self.label_names = ['human']

        # dynamic reconfigure
        dyn_namespace = namespace
        if namespace == '~':
            dyn_namespace = ''
        self.srv = Server(
            EdgeTPUHumanPoseEstimatorConfig,
            self.config_callback, namespace=dyn_namespace)

        self.pub_pose = self.advertise(
            namespace + 'output/poses', PeoplePoseArray, queue_size=1)
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
            self.visibles = None
            self.points = None

    def start(self):
        self.engine = PoseEngine(self.model_file, mirror=False)
        self.resized_H = self.engine.image_height
        self.resized_W = self.engine.image_width
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
            if y_offset:
                y_max = y_max + y_offset
                y_min = y_min + y_offset
            x_max = int(np.round(max(xs)))
            x_min = int(np.round(min(xs)))
            if x_offset:
                x_max = x_max + x_offset
                x_min = x_min + x_offset
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
        else:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

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
                self.header = msg.header
                self.points = points
                self.visibles = visibles

    def visualize_cb(self, event):
        if (not self.visualize or self.img is None
                or self.points is None or self.visibles is None):
            return

        with self.lock:
            img = self.img.copy()
            header = copy.deepcopy(self.header)
            points = self.points.copy()
            visibles = self.visibles.copy()

        fig = plt.figure(
            tight_layout={'pad': 0})
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        vis_point(img.transpose((2, 0, 1)), points, visibles, ax=ax)
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


class EdgeTPUPanoramaHumanPoseEstimator(EdgeTPUHumanPoseEstimator):
    def __init__(self, namespace='~'):
        super(EdgeTPUPanoramaHumanPoseEstimator, self).__init__(
            namespace=namespace
        )
        self.n_split = rospy.get_param('~n_split', 3)
        self.overlap = rospy.get_param('~overlap', True)

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
            img = orig_img[:, panorama_slice, :]
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
