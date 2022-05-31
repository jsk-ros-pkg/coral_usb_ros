from coral_usb.cfg import EdgeTPUObjectDetectorConfig
from coral_usb.cfg import EdgeTPUPanoramaObjectDetectorConfig
from coral_usb.cfg import EdgeTPUTileObjectDetectorConfig
from coral_usb.detector_base import EdgeTPUDetectorBase
from coral_usb.detector_base import EdgeTPUPanoramaDetectorBase
from coral_usb.detector_base import EdgeTPUTileDetectorBase


class EdgeTPUObjectDetector(EdgeTPUDetectorBase):

    _config_class = EdgeTPUObjectDetectorConfig

    def __init__(self, namespace='~'):
        super(EdgeTPUObjectDetector, self).__init__(None, None, namespace)


class EdgeTPUPanoramaObjectDetector(EdgeTPUPanoramaDetectorBase):

    _config_class = EdgeTPUPanoramaObjectDetectorConfig

    def __init__(self, namespace='~'):
        super(EdgeTPUPanoramaObjectDetector, self).__init__(
            None, None, namespace)


class EdgeTPUTileObjectDetector(EdgeTPUTileDetectorBase):

    _config_class = EdgeTPUTileObjectDetectorConfig

    def __init__(self, namespace='~'):
        super(EdgeTPUTileObjectDetector, self).__init__(
            None, None, namespace)
