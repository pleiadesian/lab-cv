# -*- coding: utf-8 -*-
import cv2
import numpy as np
from util import convolution as conv


# main filter
def mean_filter(image, kernel=np.ones((3, 3))):
    return conv.img_convolve(image, kernel) * (1.0 / kernel.size)


# median filter
def median_filter(image, kernel=np.ones((3, 3))):
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    # convolution window size
    convolve_h = int(img_h + 2 * padding_h)
    convolve_w = int(img_w + 2 * padding_w)

    # padding image for edge
    img_padding = np.zeros((convolve_h, convolve_w))
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]

    # output image after filter
    image_filter = np.zeros(image.shape)

    # convolution
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_filter[i - padding_h][j - padding_w] = int(np.median(img_padding[i - padding_h:i + padding_h + 1,
                                                                      j - padding_w:j + padding_w + 1] * kernel))
    return image_filter


# Gaussian kernel generator
def gaussian_mat(sigma):
    img_h = img_w = 2 * sigma + 1
    mat = np.zeros((img_h, img_w))
    for x in range(-sigma, sigma + 1):
        for y in range(-sigma, sigma + 1):
            mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    return mat
