#!/usr/bin/env python

import rospy

from coral_usb.semantic_segmenter import DummyEdgeTPUSemanticSegmenter
from coral_usb.semantic_segmenter import EdgeTPUPanoramaSemanticSegmenter


class DummyEdgeTPUPanoramaSemanticSegmenter(
        EdgeTPUPanoramaSemanticSegmenter, DummyEdgeTPUSemanticSegmenter):
    pass


if __name__ == '__main__':
    rospy.init_node('dummy_edgetpu_panorama_semantic_segmenter')
    segmenter = DummyEdgeTPUPanoramaSemanticSegmenter()
    rospy.spin()
