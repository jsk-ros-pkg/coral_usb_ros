#!/usr/bin/env python

import rospy

from coral_usb.node_manager import EdgeTPUNodeManager


if __name__ == '__main__':
    rospy.init_node('edgetpu_node_manager')
    manager = EdgeTPUNodeManager()
    rospy.spin()
