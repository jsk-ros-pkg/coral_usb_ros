#!/usr/bin/env python

import rospy

from coral_usb.human_pose_estimator import DummyEdgeTPUHumanPoseEstimator
from coral_usb.human_pose_estimator import EdgeTPUPanoramaHumanPoseEstimator


class DummyEdgeTPUPanoramaHumanPoseEstimator(
        EdgeTPUPanoramaHumanPoseEstimator, DummyEdgeTPUHumanPoseEstimator):
    pass


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_panorama_human_pose_estimator')
    estimator = DummyEdgeTPUPanoramaHumanPoseEstimator()
    rospy.spin()
