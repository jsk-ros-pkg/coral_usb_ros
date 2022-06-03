#!/usr/bin/env python

import rospy

from coral_usb.object_detector import DummyEdgeTPUObjectDetector
from coral_usb.object_detector import EdgeTPUTileObjectDetector


class DummyEdgeTPUTileObjectDetector(
        EdgeTPUTileObjectDetector, DummyEdgeTPUObjectDetector):
    pass


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_tile_object_detector')
    detector = DummyEdgeTPUTileObjectDetector()
    rospy.spin()
