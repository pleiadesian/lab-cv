# -*- coding: utf-8 -*-
import numpy as np


# convolution
def img_convolve(image, kernel):
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
    image_convolve = np.zeros(image.shape)

    # convolution
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(np.sum(img_padding[i - padding_h:i + padding_h + 1,
                                                                      j - padding_w:j + padding_w + 1] * kernel))

    return image_convolve


# robert operation
def robert_edge(image):
    roberts = np.array([[-1, -1],
                        [1, 1]])
    r, c = image.shape
    image_convolve = np.zeros((r, c))
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                img_window = image[x:x + 2, y:y + 2]
                robert_window = roberts * img_window
                image_convolve[x, y] = abs(robert_window.sum())
    return image_convolve


# Sobel Edge
def sobel_edge_x(image):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    return img_convolve(image, sobel_x)


# Sobel Edge
def sobel_edge_y(image):
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    return img_convolve(image, sobel_y)


# Prewitt Edge
def prewitt_edge(image):
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

    img_x = img_convolve(image, prewitt_x)
    img_y = img_convolve(image, prewitt_y)

    img_prediction = np.zeros(img_x.shape)
    for i in range(img_prediction.shape[0]):
        for j in range(img_prediction.shape[1]):
            img_prediction[i][j] = max(img_x[i][j], img_y[i][j])
    return img_prediction
