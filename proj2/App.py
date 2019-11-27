"""
@ File:     App.py
@ Author:   pleiadesian
@ Datetime: 2019-11-26 20:31
"""
import sys
import cv2
import main
import ast
import numpy as np
from core import edgedetect, morphgradient
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QFileDialog
from PyQt5.QtGui import QImage, QPixmap


class ImageProc:
    def __init__(self):
        pass

    @staticmethod
    def edge_detection(image, sigma, low_thresh, high_thresh, weak_curve, strong_curve):
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

    @staticmethod
    def morph_gradient(image, kernel, method):
        return morphgradient.morph_gradient(image, kernel, method)


class PictureView(QMainWindow, main.Ui_MainWindow):
    def __init__(self, parent=None):
        super(PictureView, self).__init__(parent)
        self.setupUi(self)
        self.image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
        self.originImage = self.image
        self.imgwidth = 500
        self.imgheight = 500
        self.display_image()

    def choose_image(self):
        fname = QFileDialog.getOpenFileName(self, 'open file', '/')[0]
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            self.image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            self.originImage = self.image
            self.display_image()

    def edge_detector(self):
        self.sigmaLabel.setEnabled(True)
        self.sigmaText.setEnabled(True)
        self.threshLabel.setEnabled(True)
        self.threshHighLabel.setEnabled(True)
        self.threshHighText.setEnabled(True)
        self.threshLowLabel.setEnabled(True)
        self.threshLowText.setEnabled(True)
        self.edgeOkButton.setEnabled(True)

    def morph_gradient(self):
        self.standardButton.setEnabled(True)
        self.externalButton.setEnabled(True)
        self.internalButton.setEnabled(True)
        self.kernelLabel.setEnabled(True)
        self.kernelText.setEnabled(True)

    def display_origin(self):
        self.image = self.originImage
        self.display_image()

    def display_image(self):
        img = cv2.resize(self.image, (self.imgheight, self.imgwidth), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.uint8)  # only uint8 is compatible with QImage
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.zoomscale = 1
        frame = QImage(img, self.imgheight, self.imgwidth, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)

    def display_edge(self):
        sigma = 3
        low_thresh = 20
        high_thresh = 30
        weak_curve = 50
        strong_curve = 255
        if self.sigmaText.text().isdigit() and int(self.sigmaText.text()) > 1:
            sigma = self.sigmaText.text()
        if self.threshLowText.text().isdigit() and int(self.threshLowText.text()) > 1:
            sigma = self.threshLowText.text()
        if self.threshHighText.text().isdigit() and int(self.threshHighText.text()) > 1:
            sigma = self.threshHighText.text()
        self.image = ImageProc.edge_detection(self.originImage, sigma, low_thresh, high_thresh, weak_curve, strong_curve)
        self.display_image()
        self.sigmaLabel.setEnabled(False)
        self.sigmaText.setEnabled(False)
        self.threshLabel.setEnabled(False)
        self.threshHighLabel.setEnabled(False)
        self.threshHighText.setEnabled(False)
        self.threshLowLabel.setEnabled(False)
        self.threshLowText.setEnabled(False)
        self.edgeOkButton.setEnabled(False)

    def display_standard_grad(self):
        kernel = np.ones((3, 3), np.float32)
        if self.kernelText.toPlainText():
            kernel_l = ast.literal_eval(self.kernelText.toPlainText())
            if kernel_l.ndim == 2 and np.all(kernel_l == 0 | kernel_l == 1):
                kernel = np.array(kernel_l).astype(np.float32)
        self.image = ImageProc.morph_gradient(self.originImage, kernel, morphgradient.GradientMethod.Standard)
        self.display_image()
        self.standardButton.setEnabled(False)
        self.externalButton.setEnabled(False)
        self.internalButton.setEnabled(False)
        self.kernelLabel.setEnabled(False)
        self.kernelText.setEnabled(False)

    def display_external_grad(self):
        kernel = np.ones((3, 3), np.float32)
        if self.kernelText.toPlainText():
            kernel_l = ast.literal_eval(self.kernelText.toPlainText())
            if kernel_l.ndim == 2 and np.all(kernel_l == 0 | kernel_l == 1):
                kernel = np.array(kernel_l).astype(np.float32)
        self.image = ImageProc.morph_gradient(self.originImage, kernel, morphgradient.GradientMethod.External)
        self.display_image()
        self.standardButton.setEnabled(False)
        self.externalButton.setEnabled(False)
        self.internalButton.setEnabled(False)
        self.kernelLabel.setEnabled(False)
        self.kernelText.setEnabled(False)

    def display_internal_grad(self):
        kernel = np.ones((3, 3), np.float32)
        if self.kernelText.toPlainText():
            kernel_l = ast.literal_eval(self.kernelText.toPlainText())
            if kernel_l.ndim == 2 and np.all(kernel_l == 0 | kernel_l == 1):
                kernel = np.array(kernel_l).astype(np.float32)
        self.image = ImageProc.morph_gradient(self.originImage, kernel, morphgradient.GradientMethod.Internal)
        self.display_image()
        self.standardButton.setEnabled(False)
        self.externalButton.setEnabled(False)
        self.internalButton.setEnabled(False)
        self.kernelLabel.setEnabled(False)
        self.kernelText.setEnabled(False)

    def geo_dilation(self):
        pass

    def geo_erosion(self):
        pass

    def open_reconstruct(self):
        pass

    def close_reconstruct(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    picView = PictureView()
    picView.show()
    app.exec_()
