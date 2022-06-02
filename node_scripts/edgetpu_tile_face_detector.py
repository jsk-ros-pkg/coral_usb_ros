#!/usr/bin/env python

import rospy

from coral_usb.face_detector import EdgeTPUTileFaceDetector


if __name__ == '__main__':
    rospy.init_node('edgetpu_tile_face_detector')
    detector = EdgeTPUTileFaceDetector()
    rospy.spin()
