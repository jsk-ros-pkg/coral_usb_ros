import rospy
import threading

from coral_usb.face_detector import EdgeTPUFaceDetector
from coral_usb.human_pose_estimator import EdgeTPUHumanPoseEstimator
from coral_usb.object_detector import EdgeTPUObjectDetector
from coral_usb.semantic_segmenter import EdgeTPUSemanticSegmenter

from coral_usb.srv import StartNode
from coral_usb.srv import StartNodeResponse
from coral_usb.srv import StopNode
from coral_usb.srv import StopNodeResponse


class EdgeTPUNodeManager(object):

    def __init__(self):
        super(EdgeTPUNodeManager, self).__init__()
        self.prefix = rospy.get_param('~prefix', '')
        # nodes: [{name: edgetpu_face_detector, type: face_detector},]
        nodes = rospy.get_param('~nodes', [])
        self.node_names = [x['name'] for x in nodes]
        self.nodes = {}
        for node in nodes:
            if node['name'] in self.nodes:
                rospy.logwarn(
                    'node with same name is registered: {}'.format(node))
            self.nodes[node['name']] = node['type']
        self.running_node = None
        self.running_node_name = None
        self.past_nodes = {}
        self.lock = threading.Lock()

        # service call
        self.start_server = rospy.Service(
            '~start', StartNode, self._start_cb)
        self.stop_server = rospy.Service(
            '~stop', StopNode, self._stop_cb)

    def _stop_node(self):
        if self.running_node is None:
            return False
        rospy.loginfo('stopping {}'.format(self.running_node_name))
        try:
            self.running_node.stop()
        except Exception:
            return False
        with self.lock:
            self.past_nodes[self.running_node_name] = self.running_node
            self.running_node = None
        return True

    def _start_node(self, name):
        rospy.loginfo('starting {}'.format(name))
        if self.running_node is not None:
            self._stop_node()
        if name not in self.node_names:
            rospy.logerr('{} is not registered in node manager'.format(name))
            return False
        if name in self.past_nodes:
            self.past_nodes[name].start()
            self.running_node = self.past_nodes[name]
            self.running_node_name = name
            return True
        node_type = self.nodes[name]
        if node_type == 'face_detector':
            node_class = EdgeTPUFaceDetector
        elif node_type == 'object_detector':
            node_class = EdgeTPUObjectDetector
        elif node_type == 'human_pose_estimator':
            node_class = EdgeTPUHumanPoseEstimator
        elif node_type == 'semantic_segmenter':
            node_class = EdgeTPUSemanticSegmenter
        else:
            rospy.logerr('{} is not supported type'.format(node_type))
            return False
        with self.lock:
            self.running_node = node_class(namespace='/' + name + '/')
            self.running_node_name = name
        return True

    def _start_cb(self, req):
        success = self._start_node(req.name)
        return StartNodeResponse(success)

    def _stop_cb(self, req):
        success = self._stop_node()
        return StopNodeResponse(success)
