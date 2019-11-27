"""
@ File:     edgedetect.py
@ Author:   pleiadesian
@ Datetime: 2019-11-26 20:31
"""
import numpy as np
import core.util as util


def gaussian_filter(image, sigma=3):
    """
    Smooth the image with a Gaussian filter
    :param image
    :param sigma: sigma for Gaussian matrix (default=3)
    :return: image processed by Gaussian filter
    """
    kernel = gaussian_mat(sigma)
    img_temp = util.img_convolve(image, kernel) * (1.0 / kernel.size)
    return img_temp


def gradient(image):
    """
    Compute gradient magnitude and direction
    :param image
    :return: magnitude and direction of image gradient
    """
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], np.float32)
    kernel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], np.float32)

    ix = util.img_convolve(image, kernel_x)
    iy = util.img_convolve(image, kernel_y)

    magnitude = np.hypot(ix, iy)  # (ix ** 2 + iy ** 2) ** 0.5
    degree = np.arctan2(ix, iy)
    return magnitude, degree


def nonmax_suppress(image, degree):
    """
    non-maxima suppression
    :param image
    :param degree: direction of image gradient
    :return: image after non-maxium suppression
    """
    img_h, img_w = image.shape
    image_supress = np.zeros((img_h, img_w), dtype=np.float32)

    for i in range(img_h):
        for j in range(img_w):
            if degree[i][j] < 0:
                degree[i][j] += 360

            if ((j + 1) < image_supress.shape[1]) and ((j - 1) >= 0) and ((i + 1) < image_supress.shape[0]) and ((i - 1) >= 0):
                if (degree[i][j] >= 337.5 or degree[i][j] < 22.5) or (157.5 <= degree[i][j] < 202.5):
                    if image[i][j] >= image[i][j + 1] and image[i][j] >= image[i][j - 1]:
                        image_supress[i][j] = image[i][j]
                if (22.5 <= degree[i][j] < 67.5) or (202.5 <= degree[i][j] < 247.5):
                    if image[i][j] >= image[i - 1][j + 1] and image[i][j] >= image[i + 1][j - 1]:
                        image_supress[i][j] = image[i][j]
                if (67.5 <= degree[i][j] < 112.5) or (247.5 <= degree[i][j] < 292.5):
                    if image[i][j] >= image[i - 1][j] and image[i][j] >= image[i + 1][j]:
                        image_supress[i][j] = image[i][j]
                if (112.5 <= degree[i][j] < 157.5) or (292.5 <= degree[i][j] < 337.5):
                    if image[i][j] >= image[i - 1][j - 1] and image[i][j] >= image[i + 1][j + 1]:
                        image_supress[i][j] = image[i][j]
    return image_supress


def threshold(image, low=20, high=30, weak_curve=50, strong_curve=255):
    """
    Hysteresis thresholding
    :param image
    :param low: lower threshold
    :param high: higher threshold
    :param weak_curve: gray scale of weak curve
    :param strong_curve: gray scale of strong curve
    :return: image after threshold
    """
    strong_index = np.where(image > high)
    weak_index = np.where((image >= low) & (image <= high))
    zero_index = np.where(image < low)

    image[strong_index[0], strong_index[1]] = strong_curve
    image[weak_index[0], weak_index[1]] = weak_curve
    image[zero_index[0], zero_index[1]] = 0

    return image


def keep_contour(image, weak=50, strong=255):
    """
    Keep only pixels along these contours
    :param image
    :param weak: gray scale of weak curve
    :param strong: gray scale of strong curve
    :return: image keeping only strong curve
    """
    img_h, img_w = image.shape
    for i in range(img_h):
        for j in range(img_w):
            if image[i, j] == weak:
                if image[i + 1, j] == strong or image[i - 1, j] == strong or image[i, j + 1] == strong or\
                        image[i, j - 1] == strong or image[i + 1, j + 1] == strong or image[i - 1, j - 1] == strong:
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image


def gaussian_mat(sigma):
    """
    Gaussian kernel generator
    :param sigma
    :return: Gaussian matrix for Gaussian filter
    """
    img_h = img_w = 2 * sigma + 1
    mat = np.zeros((img_h, img_w), dtype=np.float32)
    for x in range(-sigma, sigma + 1):
        for y in range(-sigma, sigma + 1):
            mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    return mat

