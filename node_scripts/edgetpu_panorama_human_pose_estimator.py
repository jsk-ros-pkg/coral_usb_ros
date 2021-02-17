#!/usr/bin/env python

import rospy

from coral_usb.human_pose_estimator import EdgeTPUPanoramaHumanPoseEstimator


if __name__ == '__main__':
    rospy.init_node('edgetpu_panorama_human_pose_estimator')
    detector = EdgeTPUPanoramaHumanPoseEstimator()
    rospy.spin()
