#!/usr/bin/env python

import rospy

from coral_usb.object_detector import DummyEdgeTPUObjectDetector


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_object_detector')
    detector = DummyEdgeTPUObjectDetector()
    rospy.spin()
