#!/usr/bin/env python

import rospy

from coral_usb.semantic_segmenter import DummyEdgeTPUSemanticSegmenter


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_semantic_segmenter')
    segmenter = DummyEdgeTPUSemanticSegmenter()
    rospy.spin()
