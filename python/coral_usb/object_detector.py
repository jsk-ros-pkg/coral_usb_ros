from coral_usb.cfg import EdgeTPUObjectDetectorConfig
from coral_usb.cfg import EdgeTPUPanoramaObjectDetectorConfig
from coral_usb.cfg import EdgeTPUTileObjectDetectorConfig
from coral_usb.detector_base import DummyEdgeTPUDetectorBase
from coral_usb.detector_base import EdgeTPUDetectorBase
from coral_usb.detector_base import EdgeTPUPanoramaDetectorBase
from coral_usb.detector_base import EdgeTPUTileDetectorBase


class EdgeTPUObjectDetector(EdgeTPUDetectorBase):

    _config_class = EdgeTPUObjectDetectorConfig
    _default_model_file = 'package://coral_usb/models/' + \
        'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    _default_label_file = 'package://coral_usb/models/coco_labels.txt'

    def __init__(self, namespace='~'):
        super(EdgeTPUObjectDetector, self).__init__(None, None, namespace)


class EdgeTPUPanoramaObjectDetector(
        EdgeTPUObjectDetector, EdgeTPUPanoramaDetectorBase):

    _config_class = EdgeTPUPanoramaObjectDetectorConfig

    def __init__(self, namespace='~'):
        super(EdgeTPUPanoramaObjectDetector, self).__init__(namespace)


class EdgeTPUTileObjectDetector(
        EdgeTPUObjectDetector, EdgeTPUTileDetectorBase):

    _config_class = EdgeTPUTileObjectDetectorConfig

    def __init__(self, namespace='~'):
        super(EdgeTPUTileObjectDetector, self).__init__(namespace)


class DummyEdgeTPUObjectDetector(
        EdgeTPUObjectDetector, DummyEdgeTPUDetectorBase):

    def _load_labels(self):
        self.label_names = [
            'person',
            'bicycle',
            'car',
            'motorcycle',
            'airplane',
        ]
        self.label_ids = list(range(len(self.label_names)))
