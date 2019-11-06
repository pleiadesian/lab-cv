# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

'''
# @Author  : wzl      
# @ID      : F1710w0249
# @File    : convolution.py
'''


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


# 均值滤波
# def imgAverageFilter(image, kernel):
#     '''
#     :param image: 图片矩阵
#     :param kernel: 滤波窗口
#     :return:均值滤波后的矩阵
#     '''
#     return imgConvolve(image, kernel) * (1.0 / kernel.size)


# 高斯滤波
# def imgGaussian(sigma):
#     '''
#     :param sigma: σ标准差
#     :return: 高斯滤波器的模板
#     '''
#     img_h = img_w = 2 * sigma + 1
#     gaussian_mat = np.zeros((img_h, img_w))
#     for x in range(-sigma, sigma + 1):
#         for y in range(-sigma, sigma + 1):
#             gaussian_mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
#     return gaussian_mat


# robert operation
def robert_edge(image, roberts):
    r, c = image.shape
    image_convolve = np.zeros((r, c))
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                img_window = image[x:x + 2, y:y + 2]
                robert_window = roberts * img_window
                temp = abs(robert_window.sum())
                image_convolve[x, y] = abs(robert_window.sum())
    return image_convolve


# Sobel Edge
def sobel_edge(image, sobel):
    return img_convolve(image, sobel)


# Prewitt Edge
def prewitt_edge(image, prewitt_x, prewitt_y):
    img_x = img_convolve(image, prewitt_x)
    img_y = img_convolve(image, prewitt_y)

    img_prediction = np.zeros(img_x.shape)
    for i in range(img_prediction.shape[0]):
        for j in range(img_prediction.shape[1]):
            img_prediction[i][j] = max(img_x[i][j], img_y[i][j])
    return img_prediction


# # 滤波3x3
# kernel_3x3 = np.ones((3, 3))
# # 滤波5x5
# kernel_5x5 = np.ones((5, 5))

# Roberts operator
roberts_oper = np.array([[-1, -1],
                        [1, 1]])

# Sobel operator
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Prewitt operator
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])

# # ######################均值滤波################################
# # 读图片
# image = cv2.imread('balloonGrayNoisy.jpg', cv2.IMREAD_GRAYSCALE)
# # 均值滤波
# img_k3 = imgAverageFilter(image, kernel_3x3)
#
# # 写图片
# cv2.imwrite('average_3x3.jpg', img_k3)
# # 均值滤波
# img_k5 = imgAverageFilter(image, kernel_5x5)
# # 写图片
# cv2.imwrite('average_5x5.jpg', img_k5)

# ######################高斯滤波################################
# image = cv2.imread('balloonGrayNoisy.jpg', cv2.IMREAD_GRAYSCALE)
# img_gaus1 = imgAverageFilter(image, imgGaussian(1))
# cv2.imwrite('gaussian1.jpg', img_gaus1)
# img_gaus2 = imgAverageFilter(image, imgGaussian(2))
# cv2.imwrite('gaussian2.jpg', img_gaus2)
# img_gaus3 = imgAverageFilter(image, imgGaussian(3))
# cv2.imwrite('gaussian3.jpg', img_gaus3)

image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('gray.jpg', image)

# Roberts operation
image_roberts = robert_edge(image, roberts_oper)
cv2.imwrite('roberts.jpg', image_roberts)

# Sobel operation
img_sobel_x = sobel_edge(image, sobel_x)
cv2.imwrite('sobel_x.jpg', img_sobel_x)
img_sobel_y = sobel_edge(image, sobel_y)
cv2.imwrite('sobel_y.jpg', img_sobel_y)

# Prewitt operation
img_prewitt = prewitt_edge(image, prewitt_x, prewitt_y)
cv2.imwrite('prewitt.jpg', img_prewitt)