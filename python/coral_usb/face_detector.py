import os

from dynamic_reconfigure.server import Server

from coral_usb.cfg import EdgeTPUFaceDetectorConfig
from coral_usb.detector_base import EdgeTPUDetectorBase
from coral_usb.detector_base import EdgeTPUPanoramaDetectorBase


class EdgeTPUFaceDetector(EdgeTPUDetectorBase):
    def __init__(self, namespace='~'):
        super(EdgeTPUFaceDetector, self).__init__(None, namespace)

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


class EdgeTPUPanoramaFaceDetector(EdgeTPUPanoramaDetectorBase):
    def __init__(self, namespace='~'):
        super(EdgeTPUPanoramaFaceDetector, self).__init__(None, namespace)

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
