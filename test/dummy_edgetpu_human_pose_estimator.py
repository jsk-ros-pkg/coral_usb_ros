#!/usr/bin/env python

import rospy

from coral_usb.human_pose_estimator import DummyEdgeTPUHumanPoseEstimator


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_human_pose_estimator')
    estimator = DummyEdgeTPUHumanPoseEstimator()
    rospy.spin()
