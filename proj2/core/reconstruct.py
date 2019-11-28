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


def opening_by_reconstruct(img, kernel=None, n=10):
    """
    OBR
    :param img
    :param kernel
    :param n: keep reconstructing n times at most
    :return: image after opening by reconstruction
    """
    if kernel is None:
        kernel = np.ones((3, 3), np.float32)
    seed_img = grayscale_oper(img, kernel, ReconstructMethod.Open)
    return grayscale_reconstruct(seed_img, img, n, kernel, ReconstructMethod.Dilate)


def closing_by_reconstruct(img, kernel=None, n=10):
    """
    CBR
    :param img
    :param kernel
    :param n: keep reconstructing n times at most
    :return: image after closing by reconstruction
    """
    if kernel is None:
        kernel = np.ones((3, 3), np.float32)
    seed_img = grayscale_oper(img, kernel, ReconstructMethod.Close)
    return grayscale_reconstruct(seed_img, img, n, kernel, ReconstructMethod.Erode)


def grayscale_reconstruct(marker, mask, n, kernel, method=None):
    """
    gray scale reconstruction
    :param marker
    :param mask
    :param n: keep reconstructing n times at most
    :param kernel
    :param method: opening operation or closing operation
    :return: image after gray scale reconstruction
    """
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
        return util.gray_erode(img, kernel)
    elif method == ReconstructMethod.Dilate:
        return util.gray_dilate(img, kernel)
    elif method == ReconstructMethod.Open:
        return util.gray_dilate(util.gray_erode(img, kernel), kernel)
    elif method == ReconstructMethod.Close:
        return util.gray_erode(util.gray_dilate(img, kernel), kernel)





