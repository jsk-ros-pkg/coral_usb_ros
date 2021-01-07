#!/usr/bin/env python

import rospy

from coral_usb.face_detector import EdgeTPUFaceDetector


if __name__ == '__main__':
    rospy.init_node('edgetpu_face_detector')
    detector = EdgeTPUFaceDetector()
    rospy.spin()
