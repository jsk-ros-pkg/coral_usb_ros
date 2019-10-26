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

from chainercv.visualizations import vis_bbox
from cv_bridge import CvBridge
from edgetpu.detection.engine import DetectionEngine
import PIL.Image
import rospkg
import rospy

from dynamic_reconfigure.server import Server
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image

from coral_usb.cfg import EdgeTPUFaceDetectorConfig


class EdgeTPUFaceDetector(ConnectionBasedTransport):

    def __init__(self):
        super(EdgeTPUFaceDetector, self).__init__()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coral_usb')
        self.bridge = CvBridge()
        self.classifier_name = rospy.get_param(
            '~classifier_name', rospy.get_name())
        model_file = os.path.join(
            pkg_path,
            './models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')
        model_file = rospy.get_param('~model_file', model_file)

        self.engine = DetectionEngine(model_file)
        # only for human face
        self.label_ids = [0]
        self.label_names = ['face']

        # dynamic reconfigure
        self.srv = Server(EdgeTPUFaceDetectorConfig, self.config_callback)

        self.pub_rects = self.advertise(
            '~output/rects', RectArray, queue_size=1)
        self.pub_class = self.advertise(
            '~output/class', ClassificationResult, queue_size=1)
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
        self.top_k = config.top_k
        return config

    def image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        H, W = img.shape[:2]
        objs = self.engine.DetectWithImage(
            PIL.Image.fromarray(img), threshold=self.score_thresh,
            keep_aspect_ratio=True, relative_coord=True,
            top_k=self.top_k)

        bboxes = []
        scores = []
        labels = []
        rect_msg = RectArray(header=msg.header)
        for obj in objs:
            x_min, y_min, x_max, y_max = obj.bounding_box.flatten().tolist()
            x_min = int(np.round(x_min * W))
            y_min = int(np.round(y_min * H))
            x_max = int(np.round(x_max * W))
            y_max = int(np.round(y_max * H))
            bboxes.append([y_min, x_min, y_max, x_max])
            scores.append(obj.score)
            labels.append(self.label_ids.index(int(obj.label_id)))
            rect = Rect(
                x=x_min, y=y_min,
                width=x_max-x_min, height=y_max-y_min)
            rect_msg.rects.append(rect)
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        labels = np.array(labels)

        cls_msg = ClassificationResult(
            header=msg.header,
            classifier=self.classifier_name,
            target_names=self.label_names,
            labels=labels,
            label_names=[self.label_names[l] for l in labels],
            label_proba=scores)

        self.pub_rects.publish(rect_msg)
        self.pub_class.publish(cls_msg)

        if self.visualize:
            vis_img = img[:, :, ::-1].transpose((2, 0, 1))
            vis_bbox(
                vis_img, bboxes, labels, scores,
                label_names=self.label_names)
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
    rospy.init_node('edgetpu_face_detector')
    detector = EdgeTPUFaceDetector()
    rospy.spin()
