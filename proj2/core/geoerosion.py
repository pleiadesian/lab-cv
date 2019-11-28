"""
@ File:     erosion.py
@ Author:   pleiadesian
@ Datetime: 2019-11-26 21:13
"""
import cv2
import numpy as np
import core.util as util


def geodesic_erosion(mask, kernel, n=3):
    """
    Conditional erosion in binary image
    :param mask: mask graph
    :param kernel: kernel for binary erosion
    :param n: repeat n times of erosion at most
    :return: geodesic erosion graph
    """
    m = np.reshape(mask, [1, mask.shape[0] * mask.shape[1]])
    mean = m.sum()/(mask.shape[0] * mask.shape[1])
    ret, mask = cv2.threshold(mask, mean, 255, cv2.THRESH_BINARY)

    marker = util.bin_dilate(mask, kernel)  # use dilated mask as marker

    last_marker = marker
    curr_marker = marker
    for N in range(n):
        curr_marker = util.bin_erode(last_marker, kernel)
        curr_marker = union(curr_marker, mask)
        if not np.any(last_marker != curr_marker):
            return curr_marker
        last_marker = curr_marker
    return curr_marker


def union(img1, img2):
    """
    :param img1
    :param img2
    :return: union of img1 and img2
    """
    img_h, img_w = np.shape(img1)
    union_mask = np.zeros((img_h, img_w), np.float32)
    for i in range(img_h):
        for j in range(img_w):
            if img1[i, j] == 255 or img2[i, j] == 255:
                union_mask[i, j] = 255
            else:
                union_mask[i, j] = 0
    return union_mask
