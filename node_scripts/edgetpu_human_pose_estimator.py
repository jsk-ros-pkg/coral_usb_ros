#!/usr/bin/env python

import rospy

from coral_usb.human_pose_estimator import EdgeTPUHumanPoseEstimator


if __name__ == '__main__':
    rospy.init_node('edgetpu_human_pose_estimator')
    detector = EdgeTPUHumanPoseEstimator()
    rospy.spin()
