from dynamic_reconfigure.server import Server

from coral_usb.cfg import EdgeTPUObjectDetectorConfig
from coral_usb.detector_base import EdgeTPUDetectorBase
from coral_usb.detector_base import EdgeTPUPanoramaDetectorBase


class EdgeTPUObjectDetector(EdgeTPUDetectorBase):
    def __init__(self, namespace='~'):
        model_file = 'package://coral_usb/models/' + \
            'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        label_file = 'package://coral_usb/models/coco_labels.txt'
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
        model_file = 'package://coral_usb/models/' + \
            'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        label_file = 'package://coral_usb/models/coco_labels.txt'
        super(EdgeTPUPanoramaObjectDetector, self).__init__(
            model_file, label_file, namespace)

        # dynamic reconfigure
        dyn_namespace = namespace
        if namespace == '~':
            dyn_namespace = ''
        self.srv = Server(
            EdgeTPUObjectDetectorConfig,
            self.config_callback, namespace=dyn_namespace)
