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


def morph_gradient(img, kernel, method=GradientMethod.Standard):
    """
    Morphological gradient
    :param img
    :param kernel
    :param method
    :return: Morphological gradient for the image
    """
    if method == GradientMethod.Standard:
        return (dilate(img, kernel) - erode(img, kernel)) / 2
    elif method == GradientMethod.External:
        return (dilate(img, kernel) - img) / 2
    else:
        return (img - erode(img, kernel)) / 2


def erode(img, kernel):
    """
    gray scale erosion
    :param img
    :param kernel
    :return: image after gray scale erosion
    """
    kernel = np.rot90(kernel, 2)  # erosion kernel do not need rotation

    def erode_conv(window, kernel_rot):
        return (window - kernel_rot).min()
    return util.img_convolve(img, kernel, erode_conv)


def dilate(img, kernel):
    """
    gray scale dilation
    :param img
    :param kernel
    :return: image after gray scale dilation
    """
    def dilate_conv(window, kernel_rot):
        return (window + kernel_rot).max()
    return util.img_convolve(img, kernel, dilate_conv)