#!/usr/bin/env python

import rospy

from coral_usb.face_detector import DummyEdgeTPUFaceDetector


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_face_detector')
    detector = DummyEdgeTPUFaceDetector()
    rospy.spin()
