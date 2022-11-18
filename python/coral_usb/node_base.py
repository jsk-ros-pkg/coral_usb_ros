import grp
import os
import re
import sys
import threading

# OpenCV import for python3
if os.environ['ROS_PYTHON_VERSION'] == '3':
    import cv2
else:
    sys.path.remove('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA
    import cv2  # NOQA
    sys.path.append('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA

# cv_bridge_python3 import
if os.environ['ROS_PYTHON_VERSION'] == '3':
    from cv_bridge import CvBridge
else:
    ws_python3_paths = [p for p in sys.path if 'devel/lib/python3' in p]
    if len(ws_python3_paths) == 0:
        # search cv_bridge in workspace and append
        ws_python2_paths = [
            p for p in sys.path if 'devel/lib/python2.7' in p]
        for ws_python2_path in ws_python2_paths:
            ws_python3_path = ws_python2_path.replace('python2.7', 'python3')
            if os.path.exists(os.path.join(ws_python3_path, 'cv_bridge')):
                ws_python3_paths.append(ws_python3_path)
        if len(ws_python3_paths) == 0:
            opt_python3_path = '/opt/ros/{}/lib/python3/dist-packages'.format(
                os.getenv('ROS_DISTRO'))
            sys.path = [opt_python3_path] + sys.path
            from cv_bridge import CvBridge
            sys.path.remove(opt_python3_path)
        else:
            sys.path = [ws_python3_paths[0]] + sys.path
            from cv_bridge import CvBridge
            sys.path.remove(ws_python3_paths[0])
    else:
        from cv_bridge import CvBridge

from dynamic_reconfigure.server import Server
from edgetpu.basic.edgetpu_utils import EDGE_TPU_STATE_ASSIGNED
from edgetpu.basic.edgetpu_utils import EDGE_TPU_STATE_NONE
from edgetpu.basic.edgetpu_utils import ListEdgeTpuPaths
from resource_retriever import get_filename
import rospy

from jsk_topic_tools import ConnectionBasedTransport

from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image


class EdgeTPUNodeBase(ConnectionBasedTransport):

    _engine_class = None
    _config_class = None
    _default_model_file = None
    _default_label_file = None

    def __init__(self, model_file=None, label_file=None, namespace='~'):
        super(EdgeTPUNodeBase, self).__init__()

        self.namespace = namespace
        self.bridge = CvBridge()
        # get image_trasport before ConnectionBasedTransport subscribes ~input
        self.transport_hint = rospy.get_param(
            namespace + 'image_transport', 'raw')
        rospy.loginfo("Using transport {}".format(self.transport_hint))
        self.classifier_name = rospy.get_param(
            namespace + 'classifier_name', rospy.get_name())

        # device id
        device_id = rospy.get_param(namespace + 'device_id', None)
        self._load_device_path(device_id)

        # model load
        if model_file is None:
            model_file = self._default_model_file
        self.model_file = rospy.get_param(
            namespace + 'model_file', model_file)

        if self.model_file is None:
            self.engine = None
        else:
            self.model_file = get_filename(self.model_file, False)
            self._load_model()

        # label load
        if label_file is None:
            label_file = self._default_label_file
        self.label_file = rospy.get_param(
            namespace + 'label_file', label_file)

        if self.label_file is None:
            self.label_ids = None
            self.label_names = None
        else:
            self.label_file = get_filename(self.label_file, False)
            self._load_labels()

        # input topic
        self.input_topic = rospy.get_param(
            namespace + 'input_topic', None)
        if self.input_topic is None:
            self.input_topic = rospy.resolve_name('~input')

        # dynamic reconfigure
        self.start_dynamic_reconfigure()

        # visualization param
        self.duration = rospy.get_param(namespace + 'visualize_duration', 0.1)
        self.enable_visualization = rospy.get_param(
            namespace + 'enable_visualization', True)

        # visualize timer
        if self.enable_visualization:
            self.lock = threading.Lock()
            self.pub_image = self.advertise(
                namespace + 'output/image', Image, queue_size=1)
            self.pub_image_compressed = self.advertise(
                namespace + 'output/image/compressed',
                CompressedImage, queue_size=1)
            self.timer = rospy.Timer(
                rospy.Duration(self.duration), self.visualize_cb)

    def subscribe(self):
        if self.transport_hint == 'compressed':
            self.sub_image = rospy.Subscriber(
                '{}/compressed'.format(self.input_topic),
                CompressedImage, self.image_cb, queue_size=1, buff_size=2**26)
        else:
            self.sub_image = rospy.Subscriber(
                self.input_topic, Image,
                self.image_cb, queue_size=1, buff_size=2**26)

    def unsubscribe(self):
        self.sub_image.unregister()

    @property
    def visualize(self):
        return self.pub_image.get_num_connections() > 0 or \
            self.pub_image_compressed.get_num_connections() > 0

    def _init_parameters(self):
        pass

    def _load_device_path(self, device_id):
        if not grp.getgrnam('plugdev').gr_gid in os.getgroups():
            rospy.logerr('Current user does not belong to plugdev group')
            rospy.logerr('Please run `sudo adduser $(whoami) plugdev`')
            rospy.logerr(
                'And make sure to re-login the terminal by `su -l $(whoami)`')
        if device_id is None:
            device_path = None
        else:
            device_paths = ListEdgeTpuPaths(EDGE_TPU_STATE_NONE)
            if len(device_paths) == 0:
                rospy.logerr('No device found.')
            elif device_id >= len(device_paths):
                rospy.logerr(
                    'Only {} devices are found, but device id {} is set.'
                    .format(len(device_paths), device_id))
            device_path = device_paths[device_id]
            assigned_device_paths = ListEdgeTpuPaths(EDGE_TPU_STATE_ASSIGNED)
            if device_path in assigned_device_paths:
                rospy.logwarn(
                    'device {} is already assigned: {}'.format(
                        device_id, device_path))
        self.device_path = device_path

    def _load_model(self):
        self.engine = self._engine_class(
            self.model_file, device_path=self.device_path)
        self._init_parameters()

    def _load_labels(self):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(self.label_file, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            labels = {int(num): text.strip() for num, text in lines}
        self.label_ids = list(labels.keys())
        self.label_names = list(labels.values())

    def start(self):
        if self.model_file is not None:
            self._load_model()
        self.subscribe()
        if self.enable_visualization:
            self.timer = rospy.Timer(
                rospy.Duration(self.duration), self.visualize_cb)

    def stop(self):
        self.unsubscribe()
        del self.sub_image
        if self.enable_visualization:
            self.timer.shutdown()
            del self.timer
        del self.engine

    def start_dynamic_reconfigure(self):
        if self._config_class is not None:
            # dynamic reconfigure
            dyn_namespace = self.namespace
            if dyn_namespace == '~':
                dyn_namespace = ''
            self.srv = Server(
                self._config_class,
                self.config_cb, namespace=dyn_namespace)

    def image_cb(self, msg):
        rospy.logerr('image_cb is not implemented.')

    def visualize_cb(self, event):
        rospy.logerr('visualize_cb is not implemented.')

    def config_cb(self, config, level):
        rospy.logerr('config_cb is not implemented.')


class DummyEdgeTPUNodeBase(EdgeTPUNodeBase):

    def _load_device_path(self, device_id):
        self.device_path = None

    def _load_labels(self):
        self.label_ids = None
        self.label_names = None

    def _load_model(self):
        self.engine = None
