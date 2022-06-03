import rospy
import threading

from coral_usb.face_detector import EdgeTPUFaceDetector
from coral_usb.face_detector import EdgeTPUPanoramaFaceDetector
from coral_usb.human_pose_estimator import EdgeTPUHumanPoseEstimator
from coral_usb.human_pose_estimator import EdgeTPUPanoramaHumanPoseEstimator
from coral_usb.object_detector import EdgeTPUObjectDetector
from coral_usb.object_detector import EdgeTPUPanoramaObjectDetector
from coral_usb.semantic_segmenter import EdgeTPUPanoramaSemanticSegmenter
from coral_usb.semantic_segmenter import EdgeTPUSemanticSegmenter

from coral_usb.srv import ListNodes
from coral_usb.srv import ListNodesResponse
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
        default = rospy.get_param('~default', None)
        self.running_node = None
        self.running_node_name = None
        self.past_nodes = {}
        self.lock = threading.Lock()

        # service call
        self.start_server = rospy.Service(
            '~start', StartNode, self._start_cb)
        self.stop_server = rospy.Service(
            '~stop', StopNode, self._stop_cb)
        self.list_server = rospy.Service(
            '~list', ListNodes, self._list_cb)

        if default is not None:
            self._start_node(default)

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
        elif node_type == 'panorama_face_detector':
            node_class = EdgeTPUPanoramaFaceDetector
        elif node_type == 'panorama_object_detector':
            node_class = EdgeTPUPanoramaObjectDetector
        elif node_type == 'panorama_human_pose_estimator':
            node_class = EdgeTPUPanoramaHumanPoseEstimator
        elif node_type == 'panorama_semantic_segmenter':
            node_class = EdgeTPUPanoramaSemanticSegmenter
        else:
            rospy.logerr('{} is not supported type'.format(node_type))
            return False
        namespace = self.prefix + '/' + name + '/'
        with self.lock:
            self.running_node = node_class(namespace=namespace)
            self.running_node_name = name
        return True

    def _start_cb(self, req):
        success = self._start_node(req.name)
        return StartNodeResponse(success)

    def _stop_cb(self, req):
        success = self._stop_node()
        return StopNodeResponse(success)

    def _list_cb(self, req):
        return ListNodesResponse(node_names=self.node_names)
