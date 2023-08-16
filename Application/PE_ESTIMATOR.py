import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout,QProgressBar,QSizePolicy, QFrame, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets
import pydicom
import pydicom.data
import numpy as np
from PIL import Image, ImageQt

import tensorflow as tf


class CTPAImage(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("\n\n DROP THE CTPA HERE \n\n")
        self.setStyleSheet(''' 
            QLabel{
                    background:black;
                    color:white;
                }

            ''')
        
    def setPixmap(self, image):
        super().setPixmap(image)

    def updateText(self, text):
        self.setText("\n\n "+text+" \n\n")

class Title(QLabel):
    def __init__(self,text):
        super().__init__()
        self.setText(text)
        self.setStyleSheet(''' 
            QLabel{
                    font-size: 18px; 
                }

            ''')
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


class SmallText(QLabel):
    def __init__(self,text):
        super().__init__()
        self.setText(text)
        #self.setAlignment(Qt.AlignBottom)
        self.setStyleSheet(''' 
            QLabel{
                    font-size: 12px; 
                }

            ''')
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def updateText(self, text):
        self.setText(text)


class PredictionBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setGeometry(25,25,300,300)
        self.setMaximum(100)
        self.setValue(0)
    def updateValue(self,value):
        self.setValue(value)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.currentCTPA=None
        self.modelPath=r".\model.h5"
        self.AI=None

        self.resize(400,400)
        self.setAcceptDrops(True)
        
        self.show_warning()

        mainLayout=QVBoxLayout()
        self.CTPAViewer=CTPAImage()
        mainLayout.addWidget(self.CTPAViewer)

        horizontal_bar = QFrame()
        horizontal_bar.setFrameShape(QFrame.HLine)  # Set the frame shape to a horizontal line
        horizontal_bar.setFrameShadow(QFrame.Sunken)  # Set the frame shadow to a sunken style
        mainLayout.addWidget(horizontal_bar)

        self.IMG_ID_TITLE = Title("DICOM FILE NAME")
        mainLayout.addWidget(self.IMG_ID_TITLE)

       

        self.IMG_ID = SmallText("INSERT DICOM FILE")
        mainLayout.addWidget(self.IMG_ID)

        horizontal_bar2 = QFrame()
        horizontal_bar2.setFrameShape(QFrame.HLine)  # Set the frame shape to a horizontal line
        horizontal_bar2.setFrameShadow(QFrame.Sunken)  # Set the frame shadow to a sunken style
        mainLayout.addWidget(horizontal_bar2)

        self.PE_PRED_LABEL_PROJECTS_BAR = Title("PROBABILITY OF PE EXISTANCE")
        mainLayout.addWidget( self.PE_PRED_LABEL_PROJECTS_BAR)
       
       


        self.PEPrediction=PredictionBar()
        mainLayout.addWidget(self.PEPrediction)
        self.setLayout(mainLayout)
        self.load_model()
    
    def dragEnterEvent(self, event):
        filename = event.mimeData().urls()[0].fileName()
        if filename[-3:] == "dcm":
            self.CTPAViewer.setText("BON")
            event.accept()
        else:
            self.IMG_ID.updateText("Insert DICOM FILE")
            self.PEPrediction.updateValue(0);
            self.CTPAViewer.setText("FORMAT INVALID. ONLY DICOM FILES ARE ACCEPTED")
            event.ignore()

    def dragMoveEvent(self, event):
        filename = event.mimeData().urls()[0].fileName()
        if filename[-3:] == "dcm":
            self.CTPAViewer.setText("VALID FORMAT")
            event.accept()
        else:
            self.IMG_ID.updateText("Insert DICOM FILE")
            self.PEPrediction.updateValue(0);
            self.CTPAViewer.setText("FORMAT INVALID. ONLY DICOM FILES ARE ACCEPTED")
            event.ignore()
    def dropEvent(self, event):
        url=event.mimeData().urls()[0]
        filename = url.fileName()
        path= url.path()
        if filename[-3:] == "dcm":
            self.IMG_ID.updateText(filename)
            self.CTPAViewer.setText("PROCESSING")
            self.extactCTPA(path,filename)
            event.accept()
            self.set_image(self.currentCTPA)
            self.assess_image()
            self.PEPrediction.updateValue(int(self.results[0][0]*100))
        else:
            self.IMG_ID.updateText("Insert DICOM FILE")
            self.CTPAViewer.setText("FORMAT INVALID. ONLY DICOM FILES ARE ACCEPTED")

    def resizeEvent(self, event):
        if self.currentCTPA:
            img = ImageQt.ImageQt(self.currentCTPA)  # convert to QImage
            qimage = QtGui.QImage(img)
            qpixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = qpixmap.scaled(self.CTPAViewer.width()-10, self.CTPAViewer.height()-10,
                                           aspectRatioMode=QtCore.Qt.KeepAspectRatio)
            self.CTPAViewer.setPixmap(scaled_pixmap)

    def show_warning(self):
        msg=QMessageBox()
        msg.setWindowTitle("WARNING")
        msg.setText("The software is designed to assist radiologists in their diagnostic processes and should not be considered a primary method of diagnosis. The percentage displayed represents the probability of the presence of a Pulmonary Embolism in the image inputted. It is important to note that any misdiagnosis resulting from the misuse of the application shall not be attributed to the provider of the software. \nBy acknowledging these terms and pressing OK, you confirm your understanding and agreement with the intended purpose of this application.")
        msg.setIcon(QMessageBox.Critical)
        x = msg.exec_()


    def set_image(self, image):
        pixmap = self.get_pixmap(image, self.CTPAViewer.size())
        self.CTPAViewer.setPixmap(pixmap)
        x = msg.exec_()
    

    def set_image(self, image):
        img= ImageQt.ImageQt(image)
        self.CTPAViewer.setPixmap(QPixmap.fromImage(img))
    
    def extactCTPA(self,img_path, filename):
        base=img_path[1:-len(filename)]
        pass_dicom=filename #get the last item in the list which is an string and eliminate last char.
        filename=pydicom.data.data_manager.get_files(base,pass_dicom)[0]
        img = pydicom.dcmread(filename)
        myImg=img.pixel_array.astype(float)
        rescaled_image=(np.maximum(myImg,0)/myImg.max())*255
        final_image=np.uint8(rescaled_image)
        self.currentCTPA=Image.fromarray(final_image)



    def load_model(self):
        self.AI=tf.keras.models.load_model(
                self.modelPath, compile=False)

    def assess_image(self):
        img_rgb = self.currentCTPA.convert('RGB')
        img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
        tensor_img = tf.convert_to_tensor(img_array)
        tensor_img = tf.reshape(tensor_img, [1, 512,512, 3])
        self.results=self.AI.predict(tensor_img)
  

if __name__ == '__main__':
    app =QApplication(sys.argv)
    mainW = MainWindow()
    mainW.setWindowTitle("Pulmonary Embolism Detection")
    mainW.show()
    sys.exit(app.exec_())


