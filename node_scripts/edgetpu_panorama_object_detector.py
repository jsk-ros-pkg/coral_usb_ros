#!/usr/bin/env python

import rospy

from coral_usb.object_detector import EdgeTPUPanoramaObjectDetector


if __name__ == '__main__':
    rospy.init_node('edgetpu_panorama_object_detector')
    detector = EdgeTPUPanoramaObjectDetector()
    rospy.spin()
