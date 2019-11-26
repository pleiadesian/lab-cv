"""
@ File:     dilation.py
@ Author:   pleiadesian
@ Datetime: 2019-11-26 20:31
"""
import numpy as np


def geodesic_dilation(marker, mask, n=1000):
    """
    :param marker: marker graph
    :param mask: mask graph
    :param n: repeat n times of dilation at most
    :return: geodesic dilation graph
    """
    marker_h, marker_w = np.shape(marker)
    image_gdilation = np.zeros((marker_h, marker_w), np.float32)
    last_mask = np.zeros((marker_h, marker_w), np.float32)
    inter_mask = np.zeros((marker_h, marker_w), np.float32)
    for N in range(n):
        image_dilation = bin_dilation(marker)
        inter_mask = intersect(image_dilation, mask)
        for i in range(marker_h):
            for j in range(marker_w):
                if inter_mask[i, j] == 1:
                    image_gdilation[i, j] = 255
                else:
                    image_gdilation[i, j] = 0
        if last_mask == inter_mask:
            break
        last_mask = inter_mask
    return image_gdilation, inter_mask


def bin_dilation(image):
    """
    :param image
    :return: image after binary dilation
    """
    img_h, img_w = np.shape(image)
    image_dilation = np.zeros((img_h, img_w), np.float32)
    pad_image = np.pad(image, ((1, 1), (1, 1)))
    for i in range(img_h):
        for j in range(img_w):
            window = pad_image[i:i + 3, j:j + 3]  # TODO: modify to arbitrary SE
            if np.sum(window) != 0:
                image_dilation[i, j] = 255
            else:
                image_dilation[i, j] = 0
    return image_dilation


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
                inter_mask[i, j] = 1
            else:
                inter_mask[i, j] = 0
    return inter_mask
