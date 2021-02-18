#!/usr/bin/env python

import rospy

from coral_usb.semantic_segmenter import EdgeTPUPanoramaSemanticSegmenter


if __name__ == '__main__':
    rospy.init_node('edgetpu_panorama_semantic_segmenter')
    segmenter = EdgeTPUPanoramaSemanticSegmenter()
    rospy.spin()
