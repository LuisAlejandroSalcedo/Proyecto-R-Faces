import sys
import cv2
import numpy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog
from PyQt5.QtCore import Qt

import train
import detect
import config
import imutils
import argparse
import reconocer

class FaceGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.img = None
        self.label = QLabel()
        self.UI()

    def UI(self):
        self.label.setText('Proyecto R-Face')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet('border: gray; border-style:solid; border-width: 1px;')
        btn_open = QPushButton('Abir Imagen...')
        btn_open.clicked.connect(self.abrirImagen)
        btn_procesar = QPushButton('Comenzar proceso de reconocimiento facial')
        btn_procesar.clicked.connect(self.r_faces)
        top_bar = QHBoxLayout()
        top_bar.addWidget(btn_open)
        top_bar.addWidget(btn_procesar)
        root = QVBoxLayout(self)
        root.addLayout(top_bar)
        root.addWidget(self.label)
        self.resize(540, 574)
        self.setWindowTitle('Proyecto R-Face')

    def abrirImagen(self):
        filename, _ = QFileDialog.getOpenFileName(None, 'Elegir Imagen', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        self.img = filename
        if filename:
            with open(filename, "rb") as file:
                data = numpy.array(bytearray(file.read()))
                self.image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                self.mostrarImagen()

    def r_faces(self):
        if self.image is not None:
            faceCascade = cv2.CascadeClassifier('cascades/face.xml')
            eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
            faceSize = config.DEFAULT_FACE_SIZE
            threshold = 500
            recognizer = train.trainRecognizer('train', faceSize, showFaces=True)
            capture = cv2.imread(self.img)
            while True:
                self.image = imutils.resize(capture, height=500)
                for (label, confidence, (x, y, w, h)) in reconocer.RecognizeFace(self.image, faceCascade, eyeCascade, faceSize, threshold):
                    cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(self.image, "{} = {}".format(recognizer.getLabelInfo(label), int(confidence)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)
                self.mostrarImagen()
                break

    def mostrarImagen(self):
        size = self.image.shape
        step = self.image.size / size[0]
        qformat = QImage.Format_Indexed8

        if len(size) == 3:
            if size[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, size[1], size[0], step, qformat)
        img = img.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(img))
        self.resize(self.label.pixmap().size())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FaceGUI()
    win.show()
    sys.exit(app.exec_())