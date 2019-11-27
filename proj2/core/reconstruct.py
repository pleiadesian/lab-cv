"""
@ File:     reconstruct.py
@ Author:   pleiadesian
@ Datetime: 2019-11-27 09:48
"""
import numpy as np
from enum import Enum
import core.util as util


class ReconstructMethod(Enum):
    Erode = 0
    Dilate = 1
    Open = 2
    Close = 3


def grayscale_reconstruct(marker, mask, n=3, kernel=None, method=None):
    """
    gray scale reconstruction
    :param marker
    :param mask
    :param n: keep reconstructing n times at most
    :param kernel
    :param method: opening operation or closing operation
    :return: image after gray scale reconstruction
    """
    if kernel is None:
        kernel = np.ones((3,3), np.float32)

    if method == ReconstructMethod.Erode:
        restrict_mask = np.maximum
    elif method == ReconstructMethod.Dilate:
        restrict_mask = np.minimum
    else:
        print("unsupported method for gray scale reconstruction")
        return marker

    last_marker = marker
    curr_marker = marker
    for i in range(n):
        curr_marker = grayscale_oper(last_marker, kernel, method)
        curr_marker = restrict_mask(curr_marker, mask)
        if not np.any(last_marker != marker):
            return curr_marker
        last_marker = curr_marker
    return curr_marker


def grayscale_oper(img, kernel, method):
    """
    gray scale operation on image
    :param img
    :param kernel
    :param method
    :return: image after processing with kernel and method
    """
    if method == ReconstructMethod.Erode:
        return erode(img, kernel)
    elif method == ReconstructMethod.Dilate:
        return dilate(img, kernel)
    elif method == ReconstructMethod.Open:
        return dilate(erode(img, kernel), kernel)
    elif method == ReconstructMethod.Close:
        return erode(dilate(img, kernel), kernel)


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





