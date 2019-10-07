#!/usr/bin/env python


import matplotlib
matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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
from sensor_msgs.msg import Image

from coral_usb.cfg import EdgeTPUHumanPoseEstimatorConfig
from coral_usb import PoseEngine


class EdgeTPUHumanPoseEstimator(ConnectionBasedTransport):

    def __init__(self):
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

        self.engine = PoseEngine(model_file, mirror=False)

        # dynamic reconfigure
        self.srv = Server(
            EdgeTPUHumanPoseEstimatorConfig, self.config_callback)

        self.pub_pose = self.advertise(
            '~output/poses', PeoplePoseArray, queue_size=1)
        self.pub_image = self.advertise(
            '~output/image', Image, queue_size=1)

    def subscribe(self):
        self.sub_image = rospy.Subscriber(
            '~input', Image, self.image_cb, queue_size=1, buff_size=2**26)

    def unsubscribe(self):
        self.sub_image.unregister()

    @property
    def visualize(self):
        return self.pub_image.get_num_connections() > 0

    def config_callback(self, config, level):
        self.score_thresh = config.score_thresh
        self.joint_score_thresh = config.joint_score_thresh
        return config

    def image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        resized_img = cv2.resize(img, (641, 481))
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
                point.append(keypoint.yx)
                if keypoint.score < self.joint_score_thresh:
                    visible.append(False)
                    continue
                pose_msg.limb_names.append(lbl)
                pose_msg.scores.append(keypoint.score)
                pose_msg.poses.append(
                    Pose(position=Point(x=keypoint.yx[1],
                                        y=keypoint.yx[0])))
                visible.append(True)
            poses_msg.poses.append(pose_msg)
            points.append(point)
            visibles.append(visible)
        self.pub_pose.publish(poses_msg)

        points = np.array(points, dtype=np.int32)
        visibles = np.array(visibles, dtype=np.bool)

        if self.visualize:
            vis_img = resized_img.transpose(2, 0, 1)
            vis_point(vis_img, points, visibles)
            fig = plt.gcf()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            vis_img = np.fromstring(
                fig.canvas.tostring_rgb(), dtype=np.uint8)
            fig.clf()
            vis_img.shape = (h, w, 3)
            plt.close()
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, 'rgb8')
            # BUG: https://answers.ros.org/question/316362/sensor_msgsimage-generates-float-instead-of-int-with-python3/  # NOQA
            vis_msg.step = int(vis_msg.step)
            vis_msg.header = msg.header
            self.pub_image.publish(vis_msg)


if __name__ == '__main__':
    rospy.init_node('edgetpu_human_pose_estimator')
    detector = EdgeTPUHumanPoseEstimator()
    rospy.spin()
