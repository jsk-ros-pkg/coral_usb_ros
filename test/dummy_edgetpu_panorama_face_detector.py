#!/usr/bin/env python

import rospy

from coral_usb.face_detector import DummyEdgeTPUFaceDetector
from coral_usb.face_detector import EdgeTPUPanoramaFaceDetector


class DummyEdgeTPUPanoramaFaceDetector(
        EdgeTPUPanoramaFaceDetector, DummyEdgeTPUFaceDetector):
    pass


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_panorama_face_detector')
    detector = DummyEdgeTPUPanoramaFaceDetector()
    rospy.spin()
