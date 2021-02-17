#!/usr/bin/env python

import rospy

from coral_usb.face_detector import EdgeTPUPanoramaFaceDetector


if __name__ == '__main__':
    rospy.init_node('edgetpu_panorama_face_detector')
    detector = EdgeTPUPanoramaFaceDetector()
    rospy.spin()
