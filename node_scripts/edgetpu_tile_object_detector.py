#!/usr/bin/env python

import rospy

from coral_usb.object_detector import EdgeTPUTileObjectDetector


if __name__ == '__main__':
    rospy.init_node('edgetpu_tile_object_detector')
    detector = EdgeTPUTileObjectDetector()
    rospy.spin()
