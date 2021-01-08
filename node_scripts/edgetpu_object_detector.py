#!/usr/bin/env python

import rospy

from coral_usb.object_detector import EdgeTPUObjectDetector


if __name__ == '__main__':
    rospy.init_node('edgetpu_object_detector')
    detector = EdgeTPUObjectDetector()
    rospy.spin()
