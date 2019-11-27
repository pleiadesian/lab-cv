"""
@ File:     App.py
@ Author:   pleiadesian
@ Datetime: 2019-11-26 20:31
"""
import sys
import cv2
import main
import numpy as np
from core import edgedetect
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QImage, QPixmap


class ImageProc():
    def __init__(self):
        pass

    def edge_detection(self, image, sigma, low_thresh, high_thresh, weak_curve, strong_curve):
        """
        :param image:
        :param sigma: sigma for Gaussian filter
        :param low_thresh: lower threshold for thresholding
        :param high_thresh: higher threshold for thresholding
        :param weak_curve: weak curve gray scale
        :param strong_curve: strong curve gray scale
        :return: image after canny edge detection
        """
        img = image
        img = edgedetect.gaussian_filter(img, sigma)
        img, degree = edgedetect.gradient(img)
        img = edgedetect.nonmax_suppress(img, degree)
        img = edgedetect.threshold(img, low_thresh, high_thresh, weak_curve, strong_curve)
        img = edgedetect.keep_contour(img, weak_curve, strong_curve)
        return img


class PictureView(QMainWindow, main.Ui_MainWindow):
    def __init__(self, parent=None):
        super(PictureView, self).__init__(parent)
        self.setupUi(self)
        img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
        x = int(img.shape[1] / 4) * 4
        y = int(img.shape[0] / 4) * 4
        img = cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)
        imageProc = ImageProc()
        img = imageProc.edge_detection(img, 3, 20, 30, 50, 255)
        img = img.astype(np.uint8)  # only uint8 is compatible with QImage
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.zoomscale = 1
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    picView = PictureView()
    picView.show()
    app.exec_()
