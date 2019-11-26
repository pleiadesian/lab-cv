"""
@ File:     edgedetect.py
@ Author:   pleiadesian
@ Datetime: 2019-11-26 20:31
"""
import numpy as np


# Smooth the image with a Gaussian filter
def gaussian_filter(image, sigma):
    kernel = gaussian_mat(sigma)
    img_temp = img_convolve(image, kernel) * (1.0 / kernel.size)
    return img_temp


# Compute gradient magnitude and direction
def gradient(image):
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], np.float32)
    kernel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], np.float32)

    ix = img_convolve(image, kernel_x)
    iy = img_convolve(image, kernel_y)

    magnitude = np.hypot(ix, iy)  # (ix ** 2 + iy ** 2) ** 0.5
    degree = np.arctan2(ix, iy)
    return magnitude, degree


# non-maxima suppression
def nonmax_suppress(image, degree):
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


# Hysteresis thresholding
def threshold(image, low, high, weak_curve, strong_curve):
    strong_index = np.where(image > high)
    weak_index = np.where((image >= low) & (image <= high))
    zero_index = np.where(image < low)

    image[strong_index[0], strong_index[1]] = strong_curve
    image[weak_index[0], weak_index[1]] = weak_curve
    image[zero_index[0], zero_index[1]] = 0

    return image


# Keep only pixels along these contours
def keep_contour(image, weak, strong):
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
    image_convolve = np.zeros(image.shape, dtype=np.float32)

    # convolution
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h + 1,
                       j - padding_w:j + padding_w + 1] * kernel))

    return image_convolve


# Gaussian kernel generator
def gaussian_mat(sigma):
    img_h = img_w = 2 * sigma + 1
    mat = np.zeros((img_h, img_w), dtype=np.float32)
    for x in range(-sigma, sigma + 1):
        for y in range(-sigma, sigma + 1):
            mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    return mat


def round_angle(degree):
    angle = np.rad2deg(degree) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif 22.5 <= angle < 67.5:
        angle = 45
    elif 67.5 <= angle < 112.5:
        angle = 90
    elif 112.5 <= angle < 157.5:
        angle = 135
    return angle


