import numpy as np
import PIL.Image

import rospy

from coral_usb.detector_base import EdgeTPUDetectorBase


class EdgeTPUPanoramaDetectorBase(EdgeTPUDetectorBase):

    def __init__(self, model_file=None, label_file=None, namespace='~'):
        super(EdgeTPUPanoramaDetectorBase, self).__init__(
            model_file=model_file, label_file=label_file, namespace=namespace
        )
        self.split_num = rospy.get_param('~split_num', 2)

    def _detect_objects(self, orig_img):
        _, orig_W = orig_img.shape[:2]
        x_offsets = np.arange(self.split_num) * int(orig_W / self.split_num)
        x_offsets = x_offsets.astype(np.int)
        bboxes = []
        labels = []
        scores = []
        for i in range(self.split_num):
            x_offset = x_offsets[i]
            if self.split_num == i + 1:
                x_end_offset = -1
            else:
                x_end_offset = x_offsets[i+1]
            img = orig_img[:, x_offset:x_end_offset, :]
            H, W = img.shape[:2]
            objs = self.engine.DetectWithImage(
                PIL.Image.fromarray(img), threshold=self.score_thresh,
                keep_aspect_ratio=True, relative_coord=True,
                top_k=self.top_k)
            bb, lbl, scr = self._process_result(
                objs, H, W, x_offset=x_offset)
            bboxes.append(bb)
            labels.append(lbl)
            scores.append(scr)
        bboxes = np.concatenate(bboxes, axis=0).astype(np.int)
        labels = np.concatenate(labels, axis=0).astype(np.int)
        scores = np.concatenate(scores, axis=0).astype(np.float)
        return bboxes, labels, scores
