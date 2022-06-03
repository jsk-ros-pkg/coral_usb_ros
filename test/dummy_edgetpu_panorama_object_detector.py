#!/usr/bin/env python

import rospy

from coral_usb.object_detector import DummyEdgeTPUObjectDetector
from coral_usb.object_detector import EdgeTPUPanoramaObjectDetector


class DummyEdgeTPUPanoramaObjectDetector(
        EdgeTPUPanoramaObjectDetector, DummyEdgeTPUObjectDetector):
    pass


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_panorama_object_detector')
    detector = DummyEdgeTPUPanoramaObjectDetector()
    rospy.spin()
