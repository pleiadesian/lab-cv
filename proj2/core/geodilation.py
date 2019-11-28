"""
@ File:     dilation.py
@ Author:   pleiadesian
@ Datetime: 2019-11-26 20:31
"""
import cv2
import numpy as np
import core.util as util


def geodesic_dilation(mask, kernel, n=3):
    """
    Conditional dilation in binary image
    :param mask: mask graph
    :param kernel: kernel for binary dilation
    :param n: repeat n times of dilation at most
    :return: geodesic dilation graph
    """
    m = np.reshape(mask, [1, mask.shape[0] * mask.shape[1]])
    mean = m.sum()/(mask.shape[0] * mask.shape[1])
    ret, mask = cv2.threshold(mask, mean, 255, cv2.THRESH_BINARY)
    
    marker = util.bin_erode(mask, kernel)  # use eroded mask as marker

    last_marker = marker
    curr_marker = marker
    for N in range(n):
        curr_marker = util.bin_dilate(last_marker, kernel)
        curr_marker = intersect(curr_marker, mask)
        if not np.any(last_marker != curr_marker):
            return curr_marker
        last_marker = curr_marker
    return curr_marker


def intersect(img1, img2):
    """
    :param img1
    :param img2
    :return: intersection of img1 and img2
    """
    img_h, img_w = np.shape(img1)
    inter_mask = np.zeros((img_h, img_w), np.float32)
    for i in range(img_h):
        for j in range(img_w):
            if img1[i, j] == img2[i, j] == 255:
                inter_mask[i, j] = 255
            else:
                inter_mask[i, j] = 0
    return inter_mask
