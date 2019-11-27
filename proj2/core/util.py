"""
@ File:     util.py
@ Author:   pleiadesian
@ Datetime: 2019-11-27 10:26
"""
import numpy as np


# convolution
def img_convolve(image, kernel, conv_func=None):
    """
    image convolution
    :param image
    :param kernel: kernel not rotated
    :param conv_func: if exists, do corresponding func on window
    :return: image after convolution
    """
    # rotate kernel
    kernel = np.rot90(kernel, 2)

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

    # output image after convolution
    image_convolve = np.zeros(image.shape, dtype=np.float32)

    # convolution
    if conv_func is None:
        for i in range(padding_h, padding_h + img_h):
            for j in range(padding_w, padding_w + img_w):
                image_convolve[i - padding_h][j - padding_w] = int(
                    np.sum(img_padding[i - padding_h:i + padding_h + 1,
                           j - padding_w:j + padding_w + 1] * kernel))
    else:
        for i in range(padding_h, padding_h + img_h):
            for j in range(padding_w, padding_w + img_w):
                image_convolve[i - padding_h][j - padding_w] = conv_func(img_padding[i - padding_h:i + padding_h + 1,
                                                                         j - padding_w:j + padding_w + 1], kernel)

    return image_convolve
