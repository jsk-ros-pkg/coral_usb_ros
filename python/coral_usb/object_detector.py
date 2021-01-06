import os

from dynamic_reconfigure.server import Server
import rospkg

from coral_usb.cfg import EdgeTPUObjectDetectorConfig
from coral_usb.detector_base import EdgeTPUDetectorBase


class EdgeTPUObjectDetector(EdgeTPUDetectorBase):
    def __init__(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coral_usb')
        model_file = os.path.join(
            pkg_path,
            './models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
        label_file = os.path.join(pkg_path, './models/coco_labels.txt')
        super(EdgeTPUObjectDetector, self).__init__(model_file, label_file)

        # dynamic reconfigure
        self.srv = Server(EdgeTPUObjectDetectorConfig, self.config_callback)
