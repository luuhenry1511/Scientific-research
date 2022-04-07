# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:44:17 2018

@author: Kien Minh
"""

from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
                             QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import cv2
import os
# import matplotlib library
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


haar_face_cascade = cv2.CascadeClassifier(
    'E:/2020/phantichthongke_BTH/HT/NCKH_CNN/FaceCNN/haarcascade_frontalface_default.xml')


def detect_faces_and_crop(f_cascade, colored_img, scaleFactor=1.07):
    test2 = cv2.imread(colored_img)
    # just making a copy of image passed, so that passed image is not changed
    img_copy = test2.copy()

    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=6);
    # print the number of faces found

    # go over list of faces and draw them as rectangles on original colored img
    # print('face size: ',len(faces))
    index = 0
    for (x, y, w, h) in faces:
        # how to crop image
        cropped = img_copy[y:y + h, x:x + w]
        cv2.imwrite('D:/' + str(index) + '.jpg', cropped)
        # img = Image.fromarray(cropped)
        # img = img.resize((150,150))
        img = image.load_img('D:/' + str(index) + '.jpg', target_size=(150, 150))
        x1 = image.img_to_array(img)
        x1 = np.expand_dims(x1, axis=0)
        os.remove('D:/' + str(index) + '.jpg')
        # load the model we saved
        model = load_model('E:/2020/phantichthongke_BTH/HT/NCKH/NCKH_BAINOP/KQ_train/Luu_train.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        # predicting images
        images = np.vstack([x1])
        classes = model.predict_classes(images, batch_size=16)  # label cua y
        if classes[0] == 0:
            prediction = 'lưu'
        elif classes[0] == 1:
            prediction = 'duy'

        else:
            prediction = 'Khong xac dinh'
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (126, 126, 129), 2)
        cv2.putText(img_copy, prediction, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.imwrite('D:/temp.jpg', img_copy)


class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Image Recognition")
        self.resize(500, 400)

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath())
        detect_faces_and_crop(haar_face_cascade, fileName)
        fileName = 'D:/temp.jpg'
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Recognition",
                                        "Cannot load %s." % fileName)
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()
        os.remove('D:/temp.jpg')

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Image Recognition",
                          "<p>The <b>Image Recognition</b> example shows how to combine "
                          "QLabel and QScrollArea to display an image. QLabel is "
                          "typically used for displaying text, but it can also display "
                          "an image. QScrollArea provides a scrolling view around "
                          "another widget. If the child widget exceeds the size of the "
                          "frame, QScrollArea automatically provides scroll bars.</p>"
                          "<p>The example demonstrates how QLabel's ability to scale "
                          "its contents (QLabel.scaledContents), and QScrollArea's "
                          "ability to automatically resize its contents "
                          "(QScrollArea.widgetResizable), can be used to implement "
                          "zooming and scaling features.</p>"
                          "<p>In addition the example shows how to use QPainter to "
                          "print an image.</p>")

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                               triggered=self.open)

        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P",
                                enabled=False, triggered=self.print_)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                               triggered=self.close)

        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++",
                                 enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-",
                                  enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                                     enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                                      checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                                  triggered=QApplication.instance().aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())

'''      prediction = 'an'
        elif classes[0] == 1:
            prediction = 'bao'
        elif classes[0] == 2:
            prediction = 'chuyen'
        else:
            prediction = 'ko co'
'''