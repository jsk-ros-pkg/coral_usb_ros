#!/usr/bin/env python

import rospy

from coral_usb.semantic_segmenter import EdgeTPUSemanticSegmenter


if __name__ == '__main__':
    rospy.init_node('edgetpu_semantic_segmenter')
    segmenter = EdgeTPUSemanticSegmenter()
    rospy.spin()
