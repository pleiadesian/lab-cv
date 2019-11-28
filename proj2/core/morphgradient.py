"""
@ File:     morphgradient.py
@ Author:   pleiadesian
@ Datetime: 2019-11-27 15:03
"""
import numpy as np
from enum import Enum
import core.util as util


class GradientMethod(Enum):
    Standard = 0
    External = 1
    Internal = 2


def morph_gradient(img, kernel=np.ones((3, 3), np.float32), method=GradientMethod.Standard):
    """
    Morphological gradient
    :param img
    :param kernel
    :param method
    :return: Morphological gradient for the image
    """
    if method == GradientMethod.Standard:
        return (util.gray_dilate(img, kernel) - util.gray_erode(img, kernel)) / 2
    elif method == GradientMethod.External:
        return (util.gray_dilate(img, kernel) - img) / 2
    else:
        return (img - util.gray_erode(img, kernel)) / 2
