import os

from dynamic_reconfigure.server import Server
import rospkg

from coral_usb.cfg import EdgeTPUFaceDetectorConfig
from coral_usb.detector_base import EdgeTPUDetectorBase


class EdgeTPUFaceDetector(EdgeTPUDetectorBase):
    def __init__(self, namespace='~'):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coral_usb')
        model_file = os.path.join(
            pkg_path,
            './models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')
        super(EdgeTPUFaceDetector, self).__init__(model_file, None, namespace)

        # only for human face
        self.label_ids = [0]
        self.label_names = ['face']

        # dynamic reconfigure
        dyn_namespace = namespace
        if namespace == '~':
            dyn_namespace = ''
        self.srv = Server(
            EdgeTPUFaceDetectorConfig,
            self.config_callback, namespace=dyn_namespace)
