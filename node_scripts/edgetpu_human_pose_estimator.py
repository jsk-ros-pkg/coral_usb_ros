#!/usr/bin/env python


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
import rospkg
import rospy

from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from jsk_recognition_msgs.msg import PeoplePose
from jsk_recognition_msgs.msg import PeoplePoseArray
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

from coral_usb.cfg import EdgeTPUHumanPoseEstimatorConfig
from coral_usb import PoseEngine


class EdgeTPUHumanPoseEstimator(ConnectionBasedTransport):

    def __init__(self):
        # get image_trasport before ConnectionBasedTransport subscribes ~input
        self.transport_hint = rospy.get_param('~image_transport', 'raw')
        rospy.loginfo("Using transport {}".format(self.transport_hint))
        #
        super(EdgeTPUHumanPoseEstimator, self).__init__()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coral_usb')
        self.bridge = CvBridge()
        self.classifier_name = rospy.get_param(
            '~classifier_name', rospy.get_name())
        model_file = os.path.join(
            pkg_path,
            './python/coral_usb/posenet/models/'
            'posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
        model_file = rospy.get_param('~model_file', model_file)
        duration = rospy.get_param('~visualize_duration', 0.1)
        self.enable_visualization = rospy.get_param(
            '~enable_visualization', True)

        self.engine = PoseEngine(model_file, mirror=False)
        self.resized_H = self.engine.image_height
        self.resized_W = self.engine.image_width

        # dynamic reconfigure
        self.srv = Server(
            EdgeTPUHumanPoseEstimatorConfig, self.config_callback)

        self.pub_pose = self.advertise(
            '~output/poses', PeoplePoseArray, queue_size=1)

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
            self.visibles = None
            self.points = None

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

    def image_cb(self, msg):
        if self.transport_hint == 'compressed':
            np_arr = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
        else:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        resized_img = cv2.resize(img, (self.resized_W, self.resized_H))
        H, W, _ = img.shape
        y_scale = self.resized_H / H
        x_scale = self.resized_W / W

        poses, _ = self.engine.DetectPosesInImage(resized_img.astype(np.uint8))

        poses_msg = PeoplePoseArray()
        poses_msg.header = msg.header
        points = []
        visibles = []
        for pose in poses:
            if pose.score < self.score_thresh:
                continue
            pose_msg = PeoplePose()
            point = []
            visible = []
            for lbl, keypoint in pose.keypoints.items():
                resized_key_y, resized_key_x = keypoint.yx
                key_y = resized_key_y / y_scale
                key_x = resized_key_x / x_scale
                point.append((key_y, key_x))
                if keypoint.score < self.joint_score_thresh:
                    visible.append(False)
                    continue
                pose_msg.limb_names.append(lbl)
                pose_msg.scores.append(keypoint.score)
                pose_msg.poses.append(
                    Pose(position=Point(x=key_x, y=key_y)))
                visible.append(True)
            poses_msg.poses.append(pose_msg)
            points.append(point)
            visibles.append(visible)
        self.pub_pose.publish(poses_msg)

        if self.enable_visualization:
            with self.lock:
                self.img = img
                self.header = msg.header
                self.points = np.array(points, dtype=np.int32)
                self.visibles = np.array(visibles, dtype=np.bool)

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


if __name__ == '__main__':
    rospy.init_node('edgetpu_human_pose_estimator')
    detector = EdgeTPUHumanPoseEstimator()
    rospy.spin()
