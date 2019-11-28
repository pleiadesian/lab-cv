"""
@ File:     edgedetect.py
@ Author:   pleiadesian
@ Datetime: 2019-11-28 09:01
"""
import cv2
import numpy as np
import core.util as util
from enum import Enum


class EdgeDetectMethod(Enum):
    Standard = 0
    External = 1
    Internal = 2


def edge_detect(img, kernel=np.ones((3, 3), np.float32), method=EdgeDetectMethod.Standard):
    """
    morphological edge detector
    :param img
    :param kernel
    :param method
    :return: image after edge detection
    """
    m = np.reshape(img, [1, img.shape[0] * img.shape[1]])
    mean = m.sum()/(img.shape[0] * img.shape[1])
    ret, img = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)
    if method == EdgeDetectMethod.Standard:
        return util.bin_dilate(img, kernel) - util.bin_erode(img, kernel)
    elif method == EdgeDetectMethod.External:
        return util.bin_dilate(img, kernel) - img
    else:
        return img - util.bin_erode(img, kernel)
