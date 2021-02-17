import os

from dynamic_reconfigure.server import Server
import rospkg

from coral_usb.cfg import EdgeTPUObjectDetectorConfig
from coral_usb.detector_base import EdgeTPUDetectorBase
from coral_usb.detector_base import EdgeTPUPanoramaDetectorBase


class EdgeTPUObjectDetector(EdgeTPUDetectorBase):
    def __init__(self, namespace='~'):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coral_usb')
        model_file = os.path.join(
            pkg_path,
            './models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
        label_file = os.path.join(pkg_path, './models/coco_labels.txt')
        super(EdgeTPUObjectDetector, self).__init__(
            model_file, label_file, namespace)

        # dynamic reconfigure
        dyn_namespace = namespace
        if namespace == '~':
            dyn_namespace = ''
        self.srv = Server(
            EdgeTPUObjectDetectorConfig,
            self.config_callback, namespace=dyn_namespace)


class EdgeTPUPanoramaObjectDetector(EdgeTPUPanoramaDetectorBase):
    def __init__(self, namespace='~'):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coral_usb')
        model_file = os.path.join(
            pkg_path,
            './models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
        label_file = os.path.join(pkg_path, './models/coco_labels.txt')
        super(EdgeTPUPanoramaObjectDetector, self).__init__(
            model_file, label_file, namespace)

        # dynamic reconfigure
        dyn_namespace = namespace
        if namespace == '~':
            dyn_namespace = ''
        self.srv = Server(
            EdgeTPUObjectDetectorConfig,
            self.config_callback, namespace=dyn_namespace)
