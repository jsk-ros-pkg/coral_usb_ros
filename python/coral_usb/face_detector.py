from coral_usb.cfg import EdgeTPUFaceDetectorConfig
from coral_usb.cfg import EdgeTPUPanoramaFaceDetectorConfig
from coral_usb.cfg import EdgeTPUTileFaceDetectorConfig
from coral_usb.detector_base import EdgeTPUDetectorBase
from coral_usb.detector_base import EdgeTPUPanoramaDetectorBase
from coral_usb.detector_base import EdgeTPUTileDetectorBase


class EdgeTPUFaceDetector(EdgeTPUDetectorBase):

    _config_class = EdgeTPUFaceDetectorConfig
    _default_model_file = 'package://coral_usb/models/' + \
        'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    _default_label_file = None

    def __init__(self, namespace='~'):
        super(EdgeTPUFaceDetector, self).__init__(None, None, namespace)

        # only for human face
        self.label_ids = [0]
        self.label_names = ['face']


class EdgeTPUPanoramaFaceDetector(
        EdgeTPUFaceDetector, EdgeTPUPanoramaDetectorBase):

    _config_class = EdgeTPUPanoramaFaceDetectorConfig

    def __init__(self, namespace='~'):
        super(EdgeTPUPanoramaFaceDetector, self).__init__(namespace)


class EdgeTPUTileFaceDetector(
        EdgeTPUFaceDetector, EdgeTPUTileDetectorBase):

    _config_class = EdgeTPUTileFaceDetectorConfig

    def __init__(self, namespace='~'):
        super(EdgeTPUTileFaceDetector, self).__init__(namespace)
