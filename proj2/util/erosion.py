"""
@ File:     erosion.py
@ Author:   pleiadesian
@ Datetime: 2019-11-26 21:13
"""
import numpy as np


def geodesic_erosion(marker, mask, n=100):
    """
    :param marker: marker graph
    :param mask: mask graph
    :param n: repeat n times of erosion at most
    :return: geodesic erosion graph
    """
    marker_h, marker_w = np.shape(marker)
    image_gerosion = np.zeros((marker_h, marker_w), np.float32)
    last_mask = np.zeros((marker_h, marker_w), np.float32)
    union_mask = np.zeros((marker_h, marker_w), np.float32)
    for N in range(n):
        image_erosion = bin_erosion(marker)
        union_mask = union(image_erosion, mask)
        for i in range(marker_h):
            for j in range(marker_w):
                if union_mask[i, j] == 1:
                    image_gerosion[i, j] = 255
                else:
                    image_gerosion[i, j] = 0
        if last_mask == union_mask:
            break
        last_mask = union_mask
    return image_gerosion, union_mask


def bin_erosion(image):
    """
    :param image
    :return: image after binary erosion
    """
    img_h, img_w = np.shape(image)
    image_erosion = np.zeros((img_h, img_w), np.float32)
    for i in range(img_h):
        for j in range(img_w):
            window = image[i:i + 3, j:j + 3]  # TODO: modify to arbitrary SE
            image_erosion[i, j] = 255
            for pix in window:
                if pix == 0:
                    image_erosion[i, j] = 0
                    break
    return image_erosion


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
                union_mask[i, j] = 1
            else:
                union_mask[i, j] = 0
    return union_mask
