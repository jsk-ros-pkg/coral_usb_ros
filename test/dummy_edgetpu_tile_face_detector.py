#!/usr/bin/env python

import rospy

from coral_usb.face_detector import DummyEdgeTPUFaceDetector
from coral_usb.face_detector import EdgeTPUTileFaceDetector


class DummyEdgeTPUTileFaceDetector(
        EdgeTPUTileFaceDetector, DummyEdgeTPUFaceDetector):
    pass


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_tile_face_detector')
    detector = DummyEdgeTPUTileFaceDetector()
    rospy.spin()
