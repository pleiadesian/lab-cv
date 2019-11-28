"""
@ File:     App.py
@ Author:   pleiadesian
@ Datetime: 2019-11-26 20:31
"""
import sys
import cv2
import main
import ast
import os
import numpy as np
from enum import Enum
from core import canny_edgedetect, edgedetect, geodilation, geoerosion, reconstruct, morphgradient
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QFileDialog
from PyQt5.QtGui import QImage, QPixmap


class ButtonMethod(Enum):
    EdgeDetect = 0
    Gradient = 1
    Geodilation = 2
    Geoerosion = 3
    ReconstOpen = 4
    ReconstClose = 5


class ImageProc:
    def __init__(self):
        pass

    @staticmethod
    def canny_edge_detection(image, sigma, low_thresh, high_thresh, weak_curve, strong_curve):
        """
        not used in this project
        :param image:
        :param sigma: sigma for Gaussian filter
        :param low_thresh: lower threshold for thresholding
        :param high_thresh: higher threshold for thresholding
        :param weak_curve: weak curve gray scale
        :param strong_curve: strong curve gray scale
        :return: image after canny edge detection
        """
        img = image
        img = canny_edgedetect.gaussian_filter(img, sigma)
        img, degree = canny_edgedetect.gradient(img)
        img = canny_edgedetect.nonmax_suppress(img, degree)
        img = canny_edgedetect.threshold(img, low_thresh, high_thresh, weak_curve, strong_curve)
        img = canny_edgedetect.keep_contour(img, weak_curve, strong_curve)
        return img

    @staticmethod
    def edge_detection(image, kernel, method):
        return edgedetect.edge_detect(image, kernel, method)

    @staticmethod
    def geo_dilation(image, kernel, n):
        return geodilation.geodesic_dilation(image, kernel, n)

    @staticmethod
    def geo_erosion(image, kernel, n):
        return geoerosion.geodesic_erosion(image, kernel, n)

    @staticmethod
    def opening_by_reconstruct(image, kernel, n):
        return reconstruct.opening_by_reconstruct(image, kernel, n)

    @staticmethod
    def closing_by_reconstruct(image, kernel, n):
        return reconstruct.closing_by_reconstruct(image, kernel, n)

    @staticmethod
    def morph_gradient(image, kernel, method):
        return morphgradient.morph_gradient(image, kernel, method)


class PictureView(QMainWindow, main.Ui_MainWindow):
    def __init__(self, parent=None):
        super(PictureView, self).__init__(parent)
        self.setupUi(self)
        # self.image = cv2.imread(os.getcwd()+"/lena.png", cv2.IMREAD_GRAYSCALE)
        self.image = np.ones((500, 500), np.float32)
        self.originImage = self.image
        self.imgwidth = 500
        self.imgheight = 500
        self.buttonMethod = ButtonMethod.EdgeDetect
        self.display_image()

    def choose_image(self):
        fname = QFileDialog.getOpenFileName(self, 'open file', '/')[0]
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            self.image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            self.originImage = self.image
            self.display_image()

    def generate_kernel(self):
        self.kernelText.setPlainText("[[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],"
                                     "[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],"
                                     "[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],"
                                     "[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1]]")

    def edge_detector(self):
        self.kernelText.setEnabled(True)
        self.kernelLabel.setEnabled(True)
        self.standardButton.setEnabled(True)
        self.externalButton.setEnabled(True)
        self.internalButton.setEnabled(True)
        self.buttonMethod = ButtonMethod.EdgeDetect
        # self.sigmaLabel.setEnabled(True)
        # self.sigmaText.setEnabled(True)
        # self.threshLabel.setEnabled(True)
        # self.threshHighLabel.setEnabled(True)
        # self.threshHighText.setEnabled(True)
        # self.threshLowLabel.setEnabled(True)
        # self.threshLowText.setEnabled(True)
        # self.edgeOkButton.setEnabled(True)

    def geo_dilation(self):
        self.kernelText.setEnabled(True)
        self.kernelLabel.setEnabled(True)
        self.repeatLabel.setEnabled(True)
        self.repeatText.setEnabled(True)
        self.okButton.setEnabled(True)
        self.buttonMethod = ButtonMethod.Geodilation

    def geo_erosion(self):
        self.kernelText.setEnabled(True)
        self.kernelLabel.setEnabled(True)
        self.repeatLabel.setEnabled(True)
        self.repeatText.setEnabled(True)
        self.okButton.setEnabled(True)
        self.buttonMethod = ButtonMethod.Geoerosion

    def open_reconstruct(self):
        self.kernelText.setEnabled(True)
        self.kernelLabel.setEnabled(True)
        self.repeatLabel.setEnabled(True)
        self.repeatText.setEnabled(True)
        self.okButton.setEnabled(True)
        self.buttonMethod = ButtonMethod.ReconstOpen

    def close_reconstruct(self):
        self.kernelText.setEnabled(True)
        self.kernelLabel.setEnabled(True)
        self.repeatLabel.setEnabled(True)
        self.repeatText.setEnabled(True)
        self.okButton.setEnabled(True)
        self.buttonMethod = ButtonMethod.ReconstClose

    def morph_gradient(self):
        self.kernelText.setEnabled(True)
        self.kernelLabel.setEnabled(True)
        self.standardButton.setEnabled(True)
        self.externalButton.setEnabled(True)
        self.internalButton.setEnabled(True)
        self.buttonMethod = ButtonMethod.Gradient

    def display_origin(self):
        self.image = self.originImage
        self.display_image()

    def display_binary(self):
        h = self.originImage.shape[0]
        w = self.originImage.shape[1]
        m = np.reshape(self.originImage, [1, h * w])
        mean = m.sum() / (h * w)
        ret, self.image = cv2.threshold(self.originImage, mean, 255, cv2.THRESH_BINARY)
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
        
    def display_ok(self):
        kernel = np.ones((3, 3), np.float32)
        if self.kernelText.toPlainText():
            kernel_l = np.array(ast.literal_eval(self.kernelText.toPlainText())).astype(np.float32)
            if kernel_l.ndim == 2:
                kernel_l[kernel_l != 0] = 1
                kernel = kernel_l
        n = 3
        if self.repeatText.text().isdigit() and int(self.repeatText.text()) > 0:
            n = int(self.repeatText.text())
        if self.buttonMethod == ButtonMethod.Geodilation:
            self.image = ImageProc.geo_dilation(self.originImage, kernel, n)
        elif self.buttonMethod == ButtonMethod.Geoerosion:
            self.image = ImageProc.geo_erosion(self.originImage, kernel, n)
        elif self.buttonMethod == ButtonMethod.ReconstOpen:
            self.image = ImageProc.opening_by_reconstruct(self.originImage, kernel, n)
        elif self.buttonMethod == ButtonMethod.ReconstClose:
            self.image = ImageProc.closing_by_reconstruct(self.originImage, kernel, n)
        self.display_image()
        self.kernelText.setEnabled(False)
        self.kernelLabel.setEnabled(False)
        self.repeatLabel.setEnabled(False)
        self.repeatText.setEnabled(False)
        self.okButton.setEnabled(False)

    def display_standard_grad(self):
        kernel = np.ones((3, 3), np.float32)
        if self.kernelText.toPlainText():
            kernel_l = np.array(ast.literal_eval(self.kernelText.toPlainText())).astype(np.float32)
            if kernel_l.ndim == 2:
                kernel = kernel_l
        if self.buttonMethod == ButtonMethod.EdgeDetect:
            kernel[kernel != 0] = 1
            self.image = ImageProc.edge_detection(self.originImage, kernel, edgedetect.EdgeDetectMethod.Standard)
        else:
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
            kernel_l = np.array(ast.literal_eval(self.kernelText.toPlainText())).astype(np.float32)
            if kernel_l.ndim == 2:
                kernel = np.array(kernel_l).astype(np.float32)
        if self.buttonMethod == ButtonMethod.EdgeDetect:
            kernel[kernel != 0] = 1
            self.image = ImageProc.edge_detection(self.originImage, kernel, edgedetect.EdgeDetectMethod.External)
        else:
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
            kernel_l = np.array(ast.literal_eval(self.kernelText.toPlainText())).astype(np.float32)
            if kernel_l.ndim == 2:
                kernel = np.array(kernel_l).astype(np.float32)
        if self.buttonMethod == ButtonMethod.EdgeDetect:
            kernel[kernel != 0] = 1
            self.image = ImageProc.edge_detection(self.originImage, kernel, edgedetect.EdgeDetectMethod.Internal)
        else:
            self.image = ImageProc.morph_gradient(self.originImage, kernel, morphgradient.GradientMethod.Internal)
        self.display_image()
        self.standardButton.setEnabled(False)
        self.externalButton.setEnabled(False)
        self.internalButton.setEnabled(False)
        self.kernelLabel.setEnabled(False)
        self.kernelText.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    picView = PictureView()
    picView.show()
    app.exec_()
